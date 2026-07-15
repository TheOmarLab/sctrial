"""Persistent R subprocess session for benchmark runners.

Eliminates per-call R startup overhead by keeping one Rscript process
alive per Python worker process. Libraries load once on session creation;
subsequent calls only pay for file I/O and actual computation.

Each spawn-based multiprocessing worker gets its own session dict, so
there is no shared state across workers and no fork-corruption risk.

Protocol
--------
1. Python spawns ``Rscript --vanilla bootstrap.R``.
   bootstrap.R embeds the caller-supplied init code (library loading etc.)
   at the top.  After init, R prints ``READY\\n`` to stdout and enters a
   read-eval loop that waits for commands on stdin.
2. For each analysis call, Python writes an absolute script path + ``\\n``
   to R's stdin.  R sources the script and prints ``DONE\\n`` on finish.
3. To shut down, Python sends ``QUIT\\n``; R exits cleanly.

Stdin is **never** used during the init phase — libraries load as plain R
script execution, which is reliable across non-interactive / SLURM contexts.
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

# $init_r is substituted with the caller's library-loading code.
_BOOTSTRAP_TEMPLATE = Template("""\
$init_r
cat("READY\\n")
flush(stdout())

.con <- file("stdin", open = "r")
repeat {
  .path <- readLines(.con, n = 1L)
  if (!length(.path) || trimws(.path) == "QUIT") break
  .err <- tryCatch(
    { source(trimws(.path), local = FALSE); NULL },
    error = function(e) conditionMessage(e)
  )
  if (!is.null(.err)) {
    cat(paste0("FAIL: ", gsub("\\n", " | ", .err), "\\n"))
  } else {
    cat("DONE\\n")
  }
  flush(stdout())
}
close(.con)
""")

# Per-process session registry: key -> _RSession
_sessions: dict[str, _RSession] = {}


class _RSession:
    """A long-lived Rscript subprocess with a file-based command protocol."""

    def __init__(self, init_r: str = "") -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        td = Path(self._tmpdir.name)

        bootstrap = td / "bootstrap.R"
        bootstrap.write_text(_BOOTSTRAP_TEMPLATE.substitute(init_r=init_r))

        self._proc = subprocess.Popen(
            ["Rscript", "--vanilla", str(bootstrap)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Drain stderr in background so the pipe never blocks.
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Wait for R to finish loading libraries and print READY.
        self._wait_ready(timeout=600)

        atexit.register(self.close)

    # ------------------------------------------------------------------
    def _drain_stderr(self) -> None:
        for raw in self._proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            self._stderr_lines.append(line)
            if line.startswith("R ERROR:") or "Error" in line:
                logger.warning("R stderr: %s", line)
            else:
                logger.debug("R stderr: %s", line)

    def _last_stderr(self, n: int = 30) -> str:
        return "\n".join(self._stderr_lines[-n:]) or "(no stderr output)"

    def _wait_ready(self, timeout: float) -> None:
        """Block until R prints READY\\n (init complete) or crash/timeout."""
        ready = threading.Event()
        crashed = threading.Event()

        def _reader() -> None:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    crashed.set()
                    ready.set()
                    return
                if line.strip() == b"READY":
                    ready.set()
                    return

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        if not ready.wait(timeout=timeout):
            self._proc.kill()
            raise TimeoutError(
                f"R session init timed out after {timeout}s\n"
                f"Last R stderr:\n{self._last_stderr()}"
            )
        if crashed.is_set():
            rc = self._proc.wait()
            raise RuntimeError(
                f"R session crashed during init (exit code {rc})\n"
                f"Last R stderr:\n{self._last_stderr()}"
            )

    # ------------------------------------------------------------------
    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def run(self, script_path: str, timeout: float = 1800) -> None:
        """Source *script_path* in the persistent R session."""
        if not self.is_alive():
            raise RuntimeError("R session has exited unexpectedly")
        self._send(script_path, timeout=timeout)

    def _send(self, script_path: str, timeout: float) -> None:
        self._proc.stdin.write((script_path.strip() + "\n").encode())
        self._proc.stdin.flush()

        done = threading.Event()
        crashed = threading.Event()

        r_error: list[str] = []

        def _reader() -> None:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    crashed.set()
                    done.set()
                    return
                decoded = line.decode(errors="replace").strip()
                if decoded == "DONE":
                    done.set()
                    return
                if decoded.startswith("FAIL:"):
                    r_error.append(decoded[5:].strip())
                    done.set()
                    return

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        if not done.wait(timeout=timeout):
            self._proc.kill()
            raise TimeoutError(
                f"R session timed out after {timeout}s on {script_path}\n"
                f"Last R stderr:\n{self._last_stderr()}"
            )
        if crashed.is_set():
            rc = self._proc.wait()
            raise RuntimeError(
                f"R session crashed (exit code {rc}) on {script_path}\n"
                f"Last R stderr:\n{self._last_stderr()}"
            )
        if r_error:
            raise RuntimeError(f"R error on {script_path}: {r_error[0]}")

    # ------------------------------------------------------------------
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
