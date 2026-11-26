"""Render the feature clips: stills, MP4 and GIF, frame by frame.

Each scene of ``clips/web`` is a pure function of time, so a clip is rendered by asking the page for
frame after frame and handing the frames to ffmpeg. Nothing is recorded from the screen, and two renders
of one scene give the same frames.

Needs ``ffmpeg`` and the ``playwright`` Python package with its browser (``playwright install chromium``).

    python clips/render.py --stills 1,3,5,7,9          # PNG stills of every scene, for inspection
    python clips/render.py --scene did                 # one clip: MP4 and GIF
    python clips/render.py                             # every clip
    python clips/render.py --reel square --no-gif      # every scene under its title, 1080 x 1080, MP4
    python clips/render.py --reel wide --only did,inference,power --gif-width 800
    python clips/render.py --scene did --gif-only --gif-width 640 --gif-fps 12 --out docs/source/_static/clips
    python clips/render.py --captions                  # the title and caption of every scene
    python clips/render.py --fill                      # how much of the frame the last frame of each scene uses
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import io
import subprocess
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
FPS = 30
FLOOR = 884  # height of the wall in a 1920 x 1080 frame, as in web/studio.js

FRAME_JS = """t => { window.__clip.render(t);
                     return document.getElementById('canvas').toDataURL('image/png').slice(22); }"""


def _serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *args, **kwargs: None  # type: ignore[method-assign]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _scene_ids(page, port: int) -> list[str]:
    page.goto(f"http://127.0.0.1:{port}/web/index.html?capture=1")
    return page.evaluate("window.__clip.scenes")


def _open(page, port: int, query: str) -> float:
    page.goto(f"http://127.0.0.1:{port}/web/index.html?{query}&capture=1")
    page.evaluate("window.__clip.ready")  # raises when a scene or its values did not load
    return float(page.evaluate("window.__clip.duration"))


def _frame(page, t: float) -> bytes:
    return base64.b64decode(page.evaluate(FRAME_JS, t))


def render_stills(page, port: int, name: str, query: str, times: list[float], out: Path) -> None:
    _open(page, port, query)
    for t in times:
        path = out / f"{name}_t{t:04.1f}.png"
        path.write_bytes(_frame(page, t))
        print("still", path.name)


def fill_report(page, port: int, name: str, query: str) -> bool:
    """Measure how much of the frame the last frame of a scene uses, against the empty room.

    Reports the bounding box of the content as shares of the width and of the wall height, and the
    share of a 6 x 4 grid over the wall that holds content. Needs Pillow and NumPy.
    """
    import numpy as np
    from PIL import Image

    duration = _open(page, port, query)
    frame = lambda t: np.asarray(Image.open(io.BytesIO(_frame(page, t))).convert("RGB"), dtype=int)  # noqa: E731
    content = np.abs(frame(duration - 0.8) - frame(duration)).sum(axis=2) > 24
    wall = content[:FLOOR]
    xs, ys = np.flatnonzero(content.any(axis=0)), np.flatnonzero(wall.any(axis=1))
    if not len(xs) or not len(ys):
        raise RuntimeError(f"{name}: the frame before the loop point equals the empty room (the scene does not fade)")
    width, height = (xs[-1] - xs[0] + 1) / content.shape[1], (ys[-1] - ys[0] + 1) / FLOOR
    cells = [wall[r * FLOOR // 4:(r + 1) * FLOOR // 4, c * content.shape[1] // 6:(c + 1) * content.shape[1] // 6].mean() > 0.004
             for r in range(4) for c in range(6)]
    grid = sum(cells) / len(cells)
    ok = width >= 0.85 and height >= 0.75 and grid >= 2 / 3
    print(f"{name:18s} width {width:4.0%}  wall height {height:4.0%}  grid {grid:4.0%}  "
          f"margins left {xs[0]:4d} right {content.shape[1] - 1 - xs[-1]:4d} top {ys[0]:4d}  {'ok' if ok else 'TOO EMPTY'}")
    return ok


def render_clip(page, port: int, name: str, query: str, out: Path, gif: dict | None, keep_mp4: bool) -> None:
    duration = _open(page, port, query)
    with tempfile.TemporaryDirectory() as scratch:
        mp4 = (out if keep_mp4 else Path(scratch)) / f"{name}.mp4"
        encode = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png",
             "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(mp4)],
            stdin=subprocess.PIPE,
        )
        assert encode.stdin is not None
        for i in range(round(duration * FPS)):
            encode.stdin.write(_frame(page, i / FPS))
        encode.stdin.close()
        if encode.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {name}")
        sizes = [f"{mp4.stat().st_size / 1e6:.2f} MB mp4"] if keep_mp4 else []
        if gif:
            target = out / f"{name}.gif"
            palette = (f"fps={gif['fps']},scale={gif['width']}:-1:flags=lanczos,split[a][b];"
                       f"[a]palettegen=max_colors={gif['colors']}:stats_mode=diff[p];"
                       "[b][p]paletteuse=dither=bayer:bayer_scale=4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex", palette,
                            str(target)], check=True)
            sizes.append(f"{target.stat().st_size / 1e6:.2f} MB gif")
    print(f"clip {name}: {duration:.1f} s, {', '.join(sizes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", action="append", help="scene id; may be given several times (default: all)")
    parser.add_argument("--reel", choices=["square", "wide"], help="a reel of scenes under their titles")
    parser.add_argument("--only", help="comma-separated scene ids of the reel (default: all)")
    parser.add_argument("--stills", help="comma-separated times in seconds; writes PNG stills and no video")
    parser.add_argument("--captions", action="store_true", help="print the title and caption of every scene")
    parser.add_argument("--fill", action="store_true", help="report how much of the frame each scene uses")
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--gif-colors", type=int, default=160)
    parser.add_argument("--no-gif", action="store_true", help="write the MP4 only")
    parser.add_argument("--gif-only", action="store_true", help="write the GIF only")
    parser.add_argument("--out", type=Path, default=ROOT / "out", help="output folder")
    args = parser.parse_args()
    if args.no_gif and args.gif_only:
        parser.error("--no-gif and --gif-only exclude each other")
    if args.captions and args.reel:
        parser.error("--captions lists the scenes; it does not apply to a reel")
    args.out.mkdir(parents=True, exist_ok=True)
    gif = None if args.no_gif else {"width": args.gif_width, "fps": args.gif_fps, "colors": args.gif_colors}

    server, port = _serve()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1350}, device_scale_factor=1)
            errors: list[str] = []
            full = True
            page.on("pageerror", lambda error: errors.append(str(error)))
            if args.reel:
                only = f"&only={args.only}" if args.only else ""
                jobs = [(f"reel_{args.reel}", f"reel={args.reel}{only}")]
            else:
                jobs = [(scene, f"scene={scene}") for scene in args.scene or _scene_ids(page, port)]
            for name, query in jobs:
                if args.fill:
                    full = fill_report(page, port, name, query) and full
                elif args.captions:
                    _open(page, port, query)
                    print(f"{page.evaluate('window.__clip.title')}\n    {page.evaluate('window.__clip.caption')}\n")
                elif args.stills:
                    render_stills(page, port, name, query, [float(t) for t in args.stills.split(",")], args.out)
                else:
                    render_clip(page, port, name, query, args.out, gif, keep_mp4=not args.gif_only)
                if errors:
                    raise RuntimeError(f"{name}: {errors[0]}")
            browser.close()
    finally:
        server.shutdown()
    if not full:
        raise SystemExit("at least one scene leaves too much of the frame empty")


if __name__ == "__main__":
    main()
