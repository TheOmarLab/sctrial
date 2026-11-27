"""A loopback-only, same-origin server for a single local study."""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from ._study import LocalStudy


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, study: LocalStudy | None, document: bytes, port: int):
        self.study = study
        self.offline_document = document
        self.session = secrets.token_urlsafe(32)
        marker = b"<head>"
        if marker not in document:
            raise ValueError("The installed workspace document is incomplete.")
        self.atlas = files("sctrial.interactive").joinpath("atlas")
        atlas_page = self.atlas.joinpath("index.html").read_bytes()
        self.document = atlas_page
        self.workbook = document
        self.file_stamp = None
        if study is not None:
            info = study.path.stat()
            self.file_stamp = info.st_ino, info.st_size, info.st_mtime_ns
            config = {
                "session": self.session,
                "file": {"name": study.path.name, "size": info.st_size, "url": "/local/h5ad"},
            }
            value = json.dumps(config, ensure_ascii=True).replace("<", r"\u003c").encode("ascii")
            setup = marker + b"<script>window.SCTRIAL_LOCAL=" + value + b";</script>"
            self.document = atlas_page.replace(marker, setup, 1)
            self.workbook = document.replace(marker, setup, 1)
        super().__init__(("127.0.0.1", port), WorkspaceHandler)
        self.host_header = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.host_header}"


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer

    def log_request(self, code="-", size="-"):
        pass  # Local use should not leave request paths in the terminal.

    def _send(self, status: int, data: bytes, content_type: str, *, atlas: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        policy = (
            (
                "default-src 'none'; script-src 'self' 'wasm-unsafe-eval' 'unsafe-inline'; "
                "worker-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                "font-src 'self' data:; connect-src 'self'; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
            )
            if atlas
            else (
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; img-src data: blob:; font-src data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
            )
        )
        self.send_header("Content-Security-Policy", policy)
        try:
            self.end_headers()
            if self.command != "HEAD":
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

    def _source(self, head_only: bool = False) -> None:
        # Browser workers read only the required byte ranges of the launcher file.
        study = self.server.study
        if study is None:
            self._json(404, {"error": "No local file is open."})
            return
        try:
            with study.path.open("rb") as source:
                info = os.fstat(source.fileno())
                if (info.st_ino, info.st_size, info.st_mtime_ns) != self.server.file_stamp:
                    self._json(409, {"error": "The source file changed. Reopen the study."})
                    return
                size = info.st_size
                if head_only:
                    self.send_response(200)
                    length = size
                else:
                    match = re.fullmatch(
                        r"bytes=(\d{1,20})-(\d{1,20})", self.headers.get("Range", "")
                    )
                    first, end = (int(match[1]), int(match[2])) if match else (-1, -1)
                    if not 0 <= first <= end < size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {first}-{end}/{size}")
                    source.seek(first)
                    length = end - first + 1
                self.send_header("Content-Length", str(length))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                while not head_only and length:
                    block = source.read(min(length, 256 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    length -= len(block)
        except (FileNotFoundError, PermissionError):
            self._json(409, {"error": "The local file is unavailable. Reopen the study."})
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self) -> None:
        if not self._trusted(api=True):
            return
        if urlsplit(self.path).path == "/local/h5ad":
            self._source(head_only=True)
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if not self._trusted(api=route.startswith("/local/")):
            return
        if route in {"/", "/index.html"}:
            self._send(200, self.server.document, "text/html; charset=utf-8", atlas=True)
        elif route == "/workspace.html":
            self._send(200, self.server.workbook, "text/html; charset=utf-8")
        elif route == "/sctrial-workbook.html":
            self._send(200, self.server.offline_document, "text/html; charset=utf-8")
        elif route == "/local/schema" and self.server.study is not None:
            self._json(200, self.server.study.schema())
        elif route == "/local/h5ad":
            self._source()
        else:
            self._atlas_asset(route)

    def _atlas_asset(self, route: str) -> None:
        # Serve only shipped atlas resources, keeping user file paths out of URL routing.
        types = {
            "js": "text/javascript",
            "mjs": "text/javascript",
            "css": "text/css",
            "wasm": "application/wasm",
            "json": "application/json",
            "ttf": "font/ttf",
            "bin": "application/octet-stream",
            "gmt": "text/plain",
            "txt": "text/plain",
        }
        parts = route.removeprefix("/").split("/")
        ext = parts[-1].rsplit(".", 1)[-1]
        if ext in types and all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) for p in parts):
            item = self.server.atlas
            for component in parts:
                item = item.joinpath(component)
            if item.is_file():
                self._send(200, item.read_bytes(), types[ext], atlas=True)
                return
        self._json(404, {"error": "Unknown workspace route."})

    def do_POST(self) -> None:
        if not self._trusted(api=True, mutation=True):
            return
        if self.server.study is None:
            self._json(404, {"error": "Launch the worksheet with an H5AD to use local model fits."})
            return
        actions = {
            "/local/study": self.server.study.prepare,
            "/local/fit": self.server.study.fit,
            "/local/genes": self.server.study.find_genes,
        }
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


def create_server(study: LocalStudy | None, document: bytes, port: int = 0) -> WorkspaceServer:
    return WorkspaceServer(study, document, port)
