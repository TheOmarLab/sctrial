"""A loopback-only, same-origin server for a single local study."""

from __future__ import annotations

import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from ._study import LocalStudy


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, study: LocalStudy, document: bytes, port: int):
        self.study = study
        self.offline_document = document
        self.session = secrets.token_urlsafe(32)
        marker = b"<head>"
        if marker not in document:
            raise ValueError("The installed workspace document is incomplete.")
        setup = json.dumps({"session": self.session}, separators=(",", ":")).encode("ascii")
        self.document = document.replace(
            marker, marker + b"<script>window.SCTRIAL_LOCAL=" + setup + b";</script>", 1
        )
        super().__init__(("127.0.0.1", port), WorkspaceHandler)
        self.host_header = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.host_header}"


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer

    def log_request(self, code="-", size="-"):
        pass  # Local use should not leave request paths in the terminal.

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; img-src data: blob:; font-src data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Navigating away can cancel an in-flight response.
            self.close_connection = True

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send(status, data, "application/json; charset=utf-8")

    def _trusted(self, *, api: bool = False, mutation: bool = False) -> bool:
        origin = self.headers.get("Origin")
        host_ok = self.headers.get("Host") == self.server.host_header
        origin_ok = origin == self.server.origin or (origin is None and not mutation)
        token = self.headers.get("X-sctrial-session", "")
        token_ok = not api or hmac.compare_digest(token.encode(), self.server.session.encode())
        if not (host_ok and origin_ok and token_ok):
            self._json(403, {"error": "Open the local workspace from its launcher."})
            return False
        return True

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if not self._trusted(api=route == "/local/schema"):
            return
        if route == "/":
            self._send(200, self.server.document, "text/html; charset=utf-8")
        elif route == "/sctrial-workbook.html":
            self._send(200, self.server.offline_document, "text/html; charset=utf-8")
        elif route == "/local/schema":
            self._json(200, self.server.study.schema())
        else:
            self._json(404, {"error": "Unknown workspace route."})

    def do_POST(self) -> None:
        if not self._trusted(api=True, mutation=True):
            return
        actions = {"/local/study": self.server.study.prepare, "/local/fit": self.server.study.fit}
        action = actions.get(urlsplit(self.path).path)
        if action is None:
            self._json(404, {"error": "Unknown workspace route."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Send the settings as JSON."})
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 256 * 1024:
                self._json(413, {"error": "The settings are missing or exceed 256 KiB."})
                return
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("The request must contain a settings object.")
            self._json(200, action(request))
        except (ValueError, UnicodeError) as error:
            self._json(400, {"error": str(error)})
        except OSError:
            self._json(409, {"error": "The local file could not be read. Reopen the study."})
        except Exception:
            self._json(500, {"error": "The analysis could not complete with these settings."})


def create_server(study: LocalStudy, document: bytes, port: int = 0) -> WorkspaceServer:
    return WorkspaceServer(study, document, port)
