"""Launch a private browser workspace for one local H5AD file."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from ._document import build_workspace
from ._server import create_server
from ._study import LocalStudy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study", nargs="?", type=Path, help="Annotated H5AD to open in the cell atlas"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Print the URL without opening a browser"
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Loopback port (default: choose an available port)"
    )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    try:
        study = LocalStudy(args.study) if args.study else None
        document = build_workspace()
        server = create_server(study, document, args.port)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot open the workspace: {error}\n")
    with server:
        print(f"Study workspace: {server.origin}/", flush=True)
        if not args.no_browser:
            webbrowser.open(server.origin + "/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
