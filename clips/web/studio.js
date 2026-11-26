/* sctrial study-design clips: the Studio look.
 *
 * A matte grey room with a floor, soft shadows and one amber accent. Every scene draws on a
 * 1920 x 1080 canvas through the helpers below and is a pure function of time, so a clip can be
 * played live or rendered frame by frame. The scenes are schematic: they illustrate study designs
 * and show no data.
 */
(function (global) {
  "use strict";

  const W = 1920, H = 1080, FLOOR = 884;
  const INK = "#2a2a28", SOFT = "#7a7872", RULE = "#55544f", ON_FLOOR = "#5d5b56";
  const TONES = {
    accent: { hi: "#FFE39A", mid: "#F0B02A", lo: "#C98500", flat: "#E3A008", text: "#B87A00", line: "rgba(217,148,0,.55)" },
    grey:   { hi: "#FBFBFA", mid: "#C9C8C4", lo: "#93928D", flat: "#9d9c97", text: "#77756f", line: "rgba(70,70,66,.34)" },
    dark:   { hi: "#8d8c88", mid: "#5c5b57", lo: "#3b3a37", flat: "#4a4946", text: "#3b3a37", line: "rgba(40,40,38,.55)" },
  };
  const FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif';

  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const seg = (t, a, b) => clamp((t - a) / (b - a));
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const easeOut = (x) => 1 - Math.pow(1 - x, 3);
  const lerp = (a, b, k) => a + (b - a) * k;
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  /* The height of a schematic outcome level between 0 (low) and 1 (high). */
  const level = (v) => 800 - v * 630;

  /* Sideways shifts that keep circles of radius r at the heights ys from overlapping (a swarm). */
  function swarm(ys, r) {
    const gap = 2 * r + 4, placed = [], dx = new Array(ys.length).fill(0);
    const order = ys.map((y, i) => i).sort((a, b) => ys[a] - ys[b]);
    for (const i of order) {
      const free = (x) => placed.every((q) => Math.hypot(q.x - x, q.y - ys[i]) >= gap);
      let x = 0;
      for (let step = 1; !free(x); step++) {
        const side = placed.length % 2 ? -1 : 1;
        x = free(side * step * 4) ? side * step * 4 : -side * step * 4;
      }
      dx[i] = x; placed.push({ x, y: ys[i] });
    }
    return dx;
  }

  /* Deterministic generator for the positions of grains. */
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function room(ctx, width, height, floor, depth = 1) {
    let g = ctx.createLinearGradient(0, 0, 0, floor);
    g.addColorStop(0, "#EFEEEA"); g.addColorStop(1, "#DFDDD7");
    ctx.fillStyle = g; ctx.fillRect(0, 0, width, floor);
    g = ctx.createLinearGradient(0, floor, 0, height);
    g.addColorStop(0, "#CBC9C2"); g.addColorStop(1, "#D9D7D1");
    ctx.fillStyle = g; ctx.fillRect(0, floor, width, height - floor);
    g = ctx.createLinearGradient(0, floor - 30 * depth, 0, floor + 6 * depth);
    g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, "rgba(0,0,0,.10)");
    ctx.fillStyle = g; ctx.fillRect(0, floor - 30 * depth, width, 36 * depth);
  }

  /* A standing translucent panel with its contact shadow on the floor: one visit. */
  function panel(ctx, x, alpha = 1, width = 280, top = 96) {
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha *= alpha;
    const g = ctx.createLinearGradient(x - width / 2, 0, x + width / 2, 0);
    g.addColorStop(0, "rgba(255,255,255,0)"); g.addColorStop(0.5, "rgba(255,255,255,.6)"); g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g; ctx.fillRect(x - width / 2, top, width, FLOOR - top);
    ctx.fillStyle = "rgba(0,0,0,.08)";
    ctx.beginPath(); ctx.ellipse(x, FLOOR + 8, width * 0.62, 11, 0, 0, 7); ctx.fill();
    ctx.restore();
  }

  /* A matte sphere: one participant, or one participant-level value. `shadow` is the strength of
     its shadow on the floor, 0 to 1. */
  function sphere(ctx, x, y, r, tone = "grey", alpha = 1, shadow = 1) {
    if (alpha <= 0 || r <= 0) return;
    const c = TONES[tone];
    ctx.save(); ctx.globalAlpha *= alpha;
    if (shadow > 0) {
      const lift = clamp((FLOOR - y) / (FLOOR - 120));
      ctx.fillStyle = `rgba(0,0,0,${(0.17 - 0.11 * lift) * Math.min(1, shadow)})`;
      ctx.beginPath(); ctx.ellipse(x + r * 0.35, FLOOR + 7, r * (1.05 + 0.5 * lift), r * 0.3, 0, 0, 7); ctx.fill();
    }
    const g = ctx.createRadialGradient(x - r * 0.33, y - r * 0.4, r * 0.12, x, y, r);
    g.addColorStop(0, c.hi); g.addColorStop(0.55, c.mid); g.addColorStop(1, c.lo);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
    ctx.restore();
  }

  /* The outline of a missing observation. */
  function ring(ctx, x, y, r, color = INK, alpha = 1) {
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha *= alpha; ctx.strokeStyle = color; ctx.lineWidth = 3.5; ctx.setLineDash([7, 7]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke(); ctx.restore();
  }

  /* A grain: a small matte dot without shading, for cells. */
  function grain(ctx, x, y, r, color) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  }

  function line(ctx, x0, y0, x1, y1, color, width = 3, dash = null, alpha = 1) {
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha *= alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.restore();
  }

  /* A line through `points` ([[x, y], ...]), drawn up to the fraction `upto` of its length in points. */
  function poly(ctx, points, color, width = 4, upto = 1, alpha = 1) {
    const reach = clamp(upto) * (points.length - 1), whole = Math.floor(reach);
    if (reach <= 0 || alpha <= 0) return;
    ctx.save(); ctx.globalAlpha *= alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i <= whole; i++) ctx.lineTo(points[i][0], points[i][1]);
    if (whole < points.length - 1) {
      const k = reach - whole, a = points[whole], b = points[whole + 1];
      ctx.lineTo(lerp(a[0], b[0], k), lerp(a[1], b[1], k));
    }
    ctx.stroke(); ctx.restore();
  }

  /* A vertical measure from y0 to y1 with end caps, grown by `k` from y0. */
  function bracket(ctx, x, y0, y1, k = 1, color = TONES.accent.text, alpha = 1) {
    if (k <= 0 || alpha <= 0) return;
    const y = lerp(y0, y1, k);
    line(ctx, x, y0, x, y, color, 8, null, alpha);
    line(ctx, x - 18, y0, x + 18, y0, color, 8, null, alpha);
    line(ctx, x - 18, y, x + 18, y, color, 8, null, alpha);
  }

  function text(ctx, s, x, y, size, o = {}) {
    const alpha = o.alpha === undefined ? 1 : o.alpha;
    if (alpha <= 0) return;
    ctx.save(); ctx.globalAlpha *= clamp(alpha);
    ctx.fillStyle = o.color || INK; ctx.textAlign = o.align || "left"; ctx.textBaseline = o.baseline || "middle";
    ctx.font = `${o.weight || 500} ${size}px ${FONT}`;
    if (o.spacing) ctx.letterSpacing = o.spacing;
    ctx.fillText(s, x, y); ctx.restore();
  }
  function width(ctx, s, size, weight = 500) {
    ctx.save(); ctx.font = `${weight} ${size}px ${FONT}`;
    const w = ctx.measureText(s).width; ctx.restore();
    return w;
  }

  /* The name of a visit, a group or an axis on the floor; `row` 1 is the upper line. */
  function floorLabel(ctx, s, x, alpha = 1, row = 1, color = ON_FLOOR) {
    text(ctx, s, x, FLOOR + (row === 1 ? 70 : 132), row === 1 ? 46 : 38, { align: "center", color, alpha });
  }

  /* Words in the upper left corner that light up one after another: `on` is the progress of each. */
  function steps(ctx, words, on, alpha = 1) {
    let x = 130;
    words.forEach((word, i) => {
      text(ctx, word, x, 96, 48, { color: on[i] > 0 ? INK : SOFT, alpha: alpha * (0.4 + 0.6 * on[i]) });
      const w = width(ctx, word, 48);
      line(ctx, x, 132, x + w * on[i], 132, TONES.accent.flat, 6, null, on[i] * 0.9);
      x += w + 70;
    });
  }

  const scenes = [];
  function register(scene) { scenes.push(scene); }

  /* One frame of a scene at time t (seconds). The room stays; the content fades at the loop point. */
  function frame(ctx, scene, t, state) {
    const d = scene.duration || 10;
    ctx.clearRect(0, 0, W, H); room(ctx, W, H, FLOOR);
    ctx.save(); ctx.globalAlpha = 1 - seg(t, d - 0.55, d);
    scene.draw(ctx, t, state);
    ctx.restore();
  }

  /* A reel: a title card, scenes in turn under their titles, a closing card.
   *
   * A scene is drawn into a taller room, so its wall and floor carry on above and below it: at
   * 1080 / 1920 of its size in the square layout, at full size under a title band in the wide one.
   * Each scene shows the window of its own time given by `scene.reel` ([from, to] in seconds).
   */
  const REELS = {
    square: { W: 1080, H: 1080, top: 318, k: 1080 / 1920, title: [150, 64], dots: 238, card: 500, name: 32 },
    wide: { W: 1920, H: 1220, top: 140, k: 1, title: [84, 56], dots: 0, card: 560, name: 0 },
  };
  const REEL_TIMES = { each: 4.6, intro: 1.8, outro: 2.8 };
  const reelDuration = (list = scenes) => REEL_TIMES.intro + list.length * REEL_TIMES.each + REEL_TIMES.outro;

  function reelFrame(ctx, T, states, words, L = REELS.square, list = scenes) {
    const centre = L.W / 2, total = reelDuration(list), { each, intro, outro } = REEL_TIMES;
    ctx.clearRect(0, 0, L.W, L.H); room(ctx, L.W, L.H, L.top + FLOOR * L.k, L.k);
    if (T < intro) {
      const a = seg(T, 0.1, 0.6) * (1 - seg(T, intro - 0.35, intro));
      text(ctx, words.name, centre, L.card - 40, 118, { align: "center", alpha: a, weight: 600, spacing: "-2px" });
      text(ctx, words.line, centre, L.card + 68, 38, { align: "center", alpha: a, color: SOFT });
      return;
    }
    if (T >= total - outro) {
      const u = T - (total - outro), a = seg(u, 0.1, 0.6) * (1 - seg(u, outro - 0.4, outro));
      text(ctx, words.install, centre, L.card - 40, 76, { align: "center", alpha: a, weight: 600 });
      text(ctx, words.link, centre, L.card + 60, 38, { align: "center", alpha: a, color: TONES.accent.text });
      return;
    }
    const at = T - intro, index = Math.min(list.length - 1, Math.floor(at / each));
    const scene = list[index], u = at - index * each, d = scene.duration || 10;
    const [from, to] = scene.reel || [d * 0.5, d - 0.6];
    const a = seg(u, 0, 0.3) * (1 - seg(u, each - 0.3, each));
    text(ctx, scene.title, centre, L.title[0], L.title[1], { align: "center", alpha: a, weight: 600, spacing: "-0.5px" });
    if (L.dots) {
      list.forEach((s, i) => {                               // one dot per clip
        ctx.globalAlpha = 1;
        grain(ctx, centre + (i - (list.length - 1) / 2) * 26, L.dots, i === index ? 6.5 : 4.5,
              i === index ? TONES.accent.flat : "rgba(60,60,56,.28)");
      });
    }
    ctx.save(); ctx.translate(0, L.top); ctx.scale(L.k, L.k); ctx.globalAlpha = a;
    scene.draw(ctx, lerp(from, to, u / each), states[scene.id]);
    ctx.restore();
    if (L.name) text(ctx, words.name, centre, L.H - 52, L.name, { align: "center", color: ON_FLOOR, weight: 600 });
  }

  global.Studio = {
    W, H, FLOOR, INK, SOFT, RULE, ON_FLOOR, TONES, REELS,
    clamp, seg, ease, easeOut, lerp, mean, level, swarm, rng,
    panel, sphere, ring, grain, line, poly, bracket, text, width, floorLabel, steps,
    scenes, register, frame, reelDuration, reelFrame,
  };
})(window);
