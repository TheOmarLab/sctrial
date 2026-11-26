# Study-design clips

Ten 10-second silent clips that illustrate how sctrial handles different study designs and analysis
properties: the design definition, participant-level inference, two-arm and single-arm longitudinal designs, cross-sectional
designs, repeated visits, small cohorts, incomplete and unbalanced data, cell-type-specific analysis,
and power. The clips are schematic. A sphere stands for one participant and a small dot for one cell;
positions illustrate a comparison and are not data.

| Folder or file | Content |
|---|---|
| `web/studio.js` | The look (room, floor, spheres, type) and the shared drawing functions |
| `web/scenes/` | One file per clip. A scene is a pure function of time: `draw(ctx, t, state)` |
| `web/player.js`, `web/player.css` | The list of clips beside one player; also serves single frames to the renderer |
| `web/index.html` | Standalone page. The documentation embeds the same player on its Study Designs page |
| `render.py` | Renders stills, MP4 and GIF frame by frame (Playwright and ffmpeg) |

## Play

```bash
python -m http.server 8765 --directory clips
```

Then open `http://localhost:8765/web/index.html`.

## Render

```bash
pip install playwright && playwright install chromium   # ffmpeg must be on the path
python clips/render.py --stills 2,5,9                   # PNG stills of every clip
python clips/render.py --scene two-arm                  # one clip: clips/out/two-arm.mp4 and .gif
python clips/render.py                                  # every clip
python clips/render.py --reel square --no-gif           # every clip under its title, 1080 x 1080
python clips/render.py --captions                       # the title and caption of every clip
```

Frames are requested one at a time from the scene, so two renders give the same frames.

The animations of the repository README are written by:

```bash
python clips/render.py --reel wide --only two-arm,single-arm,cross-sectional,repeated-visits,small-cohorts \
    --gif-width 800 --gif-fps 12 --gif-colors 128
cp clips/out/reel_wide.gif docs/source/_static/clips/reel.gif
python clips/render.py --scene participants --scene incomplete --scene cell-types --scene power \
    --gif-only --gif-width 640 --gif-fps 12 --gif-colors 96 --out docs/source/_static/clips
```

## Add a clip

Write `web/scenes/NN_name.js` that calls `Studio.register({ id, title, caption, reel, prepare, draw })`
and add its file name to `SCENES` in `web/player.js`. Titles and captions are read by statisticians
and clinical investigators: state what is estimated or tested in the terms of the field, and keep
explanation in the caption, not in the clip.
