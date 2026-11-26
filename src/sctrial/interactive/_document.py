"""Assemble a self-contained workspace for both the local server and offline use."""

from __future__ import annotations

from importlib.resources import files

_STYLES = ("font", "dialog", "cohort", "results", "pins", "canvas", "mapping", "studio")
_SCRIPTS = ("csv", "cohort", "results", "pins", "canvas", "mapping", "studio")


def build_workspace() -> bytes:
    """Return the offline document with its data, scripts, styles and font embedded."""
    assets = files("sctrial.interactive").joinpath("assets")

    def read(name: str) -> str:
        return assets.joinpath(name).read_text(encoding="utf-8")

    style = "\n".join(read(name + ".css") for name in _STYLES)
    scripts = "\n".join(read(name + ".js") for name in _SCRIPTS)
    return (
        read("shell.html")
        .replace("/*STYLE*/", style)
        .replace("/*SCRIPT*/", scripts)
        .replace("/*FIXTURE*/", read("example.json").strip())
        .encode("utf-8")
    )
