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
"""

from __future__ import annotations

import argparse
import base64
import functools
import http.server
import subprocess
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
FPS = 30

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
            page.on("pageerror", lambda error: errors.append(str(error)))
            if args.reel:
                only = f"&only={args.only}" if args.only else ""
                jobs = [(f"reel_{args.reel}", f"reel={args.reel}{only}")]
            else:
                jobs = [(scene, f"scene={scene}") for scene in args.scene or _scene_ids(page, port)]
            for name, query in jobs:
                if args.captions:
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


if __name__ == "__main__":
    main()
