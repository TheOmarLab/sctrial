"""Persistent R subprocess session for benchmark runners.

Eliminates per-call R startup overhead by keeping one Rscript process
alive per Python worker process. Libraries load once on session creation;
subsequent calls only pay for file I/O and actual computation.

Each spawn-based multiprocessing worker gets its own session dict, so
there is no shared state across workers and no fork-corruption risk.
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Bootstrap R: starts a read-eval loop that sources scripts sent via stdin
# and signals completion with "DONE\n" on stdout.
_BOOTSTRAP_R = """\
con <- stdin()
repeat {
  script_path <- readLines(con, n=1L)
  if (!length(script_path) || trimws(script_path) == "QUIT") break
  tryCatch(
    source(trimws(script_path), local=FALSE),
    error = function(e) message("R ERROR: ", conditionMessage(e))
  )
  cat("DONE\\n")
  flush(stdout())
}
"""

# Per-process session registry: key -> _RSession
_sessions: dict[str, _RSession] = {}


class _RSession:
    """A long-lived Rscript subprocess with a file-based command protocol.

    Protocol:
      Python → R stdin : absolute path of an .R script to source, followed by newline
      R → Python stdout: "DONE\\n" after the script finishes
      Python → R stdin : "QUIT\\n" to shut down
    """

    def __init__(self, init_r: str | None = None) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        td = Path(self._tmpdir.name)

        bootstrap = td / "bootstrap.R"
        bootstrap.write_text(_BOOTSTRAP_R)

        self._proc = subprocess.Popen(
            ["Rscript", "--vanilla", str(bootstrap)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        if init_r:
            init_script = td / "init.R"
            init_script.write_text(init_r)
            self._send(str(init_script), timeout=600)

        atexit.register(self.close)

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def run(self, script_path: str, timeout: float = 1800) -> None:
        """Source *script_path* in the persistent R session and wait for completion."""
        if not self.is_alive():
            raise RuntimeError("R session has exited unexpectedly")
        self._send(script_path, timeout=timeout)

    def _send(self, script_path: str, timeout: float) -> None:
        self._proc.stdin.write((script_path.strip() + "\n").encode())
        self._proc.stdin.flush()

        done = threading.Event()

        def _reader() -> None:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    return  # EOF — process died
                if line.strip() == b"DONE":
                    done.set()
                    return

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        if not done.wait(timeout=timeout):
            self._proc.kill()
            raise TimeoutError(f"R session timed out after {timeout}s on {script_path}")

    def close(self) -> None:
        if self.is_alive():
            try:
                self._proc.stdin.write(b"QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        try:
            self._tmpdir.cleanup()
        except Exception:
            pass


def get_session(key: str, init_r: str) -> _RSession:
    """Return the live session for *key*, creating/restarting it if needed."""
    sess = _sessions.get(key)
    if sess is None or not sess.is_alive():
        logger.debug("Starting persistent R session: %s", key)
        _sessions[key] = _RSession(init_r)
    return _sessions[key]
