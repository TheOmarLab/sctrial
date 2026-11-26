/* Cell-type-specific analysis: the cells of each sample are split by cell type and the design is
 * applied within each type. Schematic: the grains and the estimates illustrate and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, grain, sphere, line, floorLabel } = S;
  const TYPES = [
    { name: "T cells", color: "#4a4946", n: 130, y: 250, est: 0.62, half: 0.34 },
    { name: "B cells", color: "#8f8e89", n: 80, y: 490, est: 0.10, half: 0.44 },
    { name: "myeloid cells", color: "#c2c0b9", n: 105, y: 730, est: -0.48, half: 0.32 },
  ];
  const XC = 390, ZERO = 1370, UNIT = 360;

  S.register({
    id: "cell-types",
    title: "Cell-type-specific analysis",
    caption: "The same design-matched comparison is applied within each cell type, with one value per participant, visit and cell type, and one estimate with its confidence interval per cell type. The change in cell-type composition can be tested in the same framework.",
    reel: [2.8, 9.4],
    prepare() {
      const rand = S.rng(9), grains = [];
      const disc = (cx, cy, r) => { const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * r; return [cx + Math.cos(a) * d, cy + Math.sin(a) * d]; };
      TYPES.forEach((type, k) => {
        for (let i = 0; i < type.n; i++) grains.push({ k, color: type.color, delay: rand(), mixed: disc(XC, 490, 230), sorted: disc(XC, type.y, 36 + type.n * 0.48) });
      });
      return { grains };
    },
    draw(ctx, t, d) {
      const appear = seg(t, 0, 0.8);
      for (const g of d.grains) {
        const k = ease(seg(t, 1.2 + g.k * 0.35 + g.delay * 0.5, 2.8 + g.k * 0.35 + g.delay * 0.5));
        ctx.globalAlpha = appear;
        grain(ctx, lerp(g.mixed[0], g.sorted[0], k), lerp(g.mixed[1], g.sorted[1], k), 4.6, g.color);
      }
      ctx.globalAlpha = 1;
      floorLabel(ctx, "cells of one sample", XC, seg(t, 0.2, 0.8) * (1 - seg(t, 2.6, 3.1)));
      floorLabel(ctx, "by cell type", XC, seg(t, 3.1, 3.7));

      const axis = ease(seg(t, 4.2, 5.0));
      line(ctx, ZERO, 150, ZERO, lerp(150, 830, axis), S.RULE, 3.5, [6, 13], axis);
      floorLabel(ctx, "estimate within each cell type", ZERO, axis);
      text(ctx, "no effect", ZERO, 104, 38, { align: "center", color: S.SOFT, alpha: axis });
      TYPES.forEach((type, k) => {
        text(ctx, type.name, XC + 230, type.y, 46, { alpha: seg(t, 3.0 + k * 0.2, 3.6 + k * 0.2) });
        const e = ease(seg(t, 5.0 + k * 0.6, 5.9 + k * 0.6)), x = ZERO + type.est * UNIT, h = type.half * UNIT * e;
        if (e <= 0) return;
        line(ctx, XC + 560, type.y, lerp(XC + 560, x - h - 44, e), type.y, S.TONES.grey.line, 3, [4, 11], 0.9);
        line(ctx, x - h, type.y, x + h, type.y, S.TONES.accent.text, 8);
        line(ctx, x - h, type.y - 18, x - h, type.y + 18, S.TONES.accent.text, 8);
        line(ctx, x + h, type.y - 18, x + h, type.y + 18, S.TONES.accent.text, 8);
        sphere(ctx, x, type.y, 23 * e, "accent", e, 0);
      });
    },
  });
})(window.Studio);
