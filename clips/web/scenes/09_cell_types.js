/* Cell-type-specific analysis: the cells of each sample are split by cell type, every participant
 * contributes one value per cell type, and the design is applied to those values within each type.
 * Schematic: the grains and the values illustrate and show no data. The interval drawn in a row is
 * the t interval of the six participant values drawn in that row.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, grain, sphere, line, floorLabel } = S;
  // `values`: one schematic value per participant; `own`: the participant whose sample is shown on the left
  const TYPES = [
    { name: "T cells", color: "#4a4946", n: 130, y: 250, own: 2, values: [0.19, 0.37, 0.55, 0.72, 0.85, 1.03] },
    { name: "B cells", color: "#8f8e89", n: 80, y: 490, own: 3, values: [-0.45, -0.27, 0.05, 0.19, 0.37, 0.70] },
    { name: "myeloid cells", color: "#c2c0b9", n: 105, y: 730, own: 1, values: [-0.94, -0.68, -0.55, -0.38, -0.25, -0.09] },
  ];
  const XC = 270, ZERO = 1370, UNIT = 400, RP = 18, UP = 30, DOWN = 36;
  const T975 = 2.571;                                       // 0.975 quantile of the t distribution, 5 degrees of freedom

  S.register({
    id: "cell-types",
    title: "Cell-type-specific analysis",
    caption: "The same design-matched comparison is applied within each cell type, with one value per participant, visit and cell type, and one estimate with its confidence interval per cell type. The change in cell-type composition can be tested in the same framework.",
    reel: [2.6, 9.4],
    prepare() {
      const rand = S.rng(9), grains = [];
      const disc = (cx, cy, rx, ry = rx) => { const a = rand() * Math.PI * 2, d = Math.sqrt(rand()); return [cx + Math.cos(a) * d * rx, cy + Math.sin(a) * d * ry]; };
      TYPES.forEach((type, k) => {
        for (let i = 0; i < type.n; i++) grains.push({ k, color: type.color, delay: rand(), mixed: disc(S.W / 2, 490, 700, 300), sorted: disc(XC, type.y, 48 + type.n * 0.55) });
      });
      const types = TYPES.map((type) => {
        const n = type.values.length, est = S.mean(type.values);
        const sd = Math.sqrt(type.values.reduce((s, v) => s + (v - est) ** 2, 0) / (n - 1));
        const half = T975 * sd / Math.sqrt(n), xs = type.values.map((v) => ZERO + v * UNIT);
        return { ...type, est, half, xs, from: Math.min(xs[0] - RP, ZERO + (est - half) * UNIT) };
      });
      return { grains, types };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent, base = ctx.globalAlpha, appear = base * seg(t, 0, 0.8);
      for (const g of d.grains) {
        const k = ease(seg(t, 0.9 + g.k * 0.25 + g.delay * 0.4, 2.2 + g.k * 0.25 + g.delay * 0.4));
        ctx.globalAlpha = appear;
        grain(ctx, lerp(g.mixed[0], g.sorted[0], k), lerp(g.mixed[1], g.sorted[1], k), 5.2, g.color);
      }
      ctx.globalAlpha = base;
      floorLabel(ctx, "cells of one sample", S.W / 2, seg(t, 0.2, 0.8) * (1 - seg(t, 1.8, 2.3)));
      floorLabel(ctx, "by cell type", XC, seg(t, 2.6, 3.2));

      const axis = ease(seg(t, 3.2, 4.0));
      line(ctx, ZERO, 150, ZERO, lerp(150, 830, axis), S.RULE, 3.5, [6, 13], axis);
      text(ctx, "no effect", ZERO, 104, 38, { align: "center", color: S.SOFT, alpha: axis });
      floorLabel(ctx, "one value per participant", ZERO, seg(t, 4.3, 4.9));
      floorLabel(ctx, "estimate with its confidence interval", ZERO, seg(t, 6.3, 6.9), 2, A.text);

      d.types.forEach((type, k) => {
        text(ctx, type.name, XC + 250, type.y, 48, { alpha: seg(t, 2.6 + k * 0.2, 3.2 + k * 0.2) });
        line(ctx, XC + 600, type.y, lerp(XC + 600, type.from - 44, axis), type.y, S.TONES.grey.line, 3, [4, 11], 0.9 * axis);
        // the value of the sample shown on the left leaves its cells, then the other participants appear
        type.xs.forEach((x, j) => {
          if (j === type.own) {
            const e = ease(seg(t, 3.6 + k * 0.15, 4.6 + k * 0.15));
            sphere(ctx, lerp(XC, x, e), type.y - UP - 40 * Math.sin(Math.PI * e), RP, "grey", seg(e, 0, 0.15), 0);
          } else {
            const order = j < type.own ? j : j - 1, at = 4.8 + k * 0.1 + order * 0.14;
            sphere(ctx, x, type.y - UP, RP, "grey", S.easeOut(seg(t, at, at + 0.4)), 0);
          }
        });
        // the estimate and its confidence interval, from the participant values of the row
        const e = ease(seg(t, 6.3 + k * 0.5, 7.2 + k * 0.5)), x = ZERO + type.est * UNIT, h = type.half * UNIT * e, y = type.y + DOWN;
        if (e <= 0) return;
        line(ctx, x - h, y, x + h, y, A.text, 8);
        line(ctx, x - h, y - 18, x - h, y + 18, A.text, 8);
        line(ctx, x + h, y - 18, x + h, y + 18, A.text, 8);
        sphere(ctx, x, y, 21 * e, "accent", e, 0);
      });
    },
  });
})(window.Studio);
