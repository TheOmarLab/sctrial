/* Cell-type-specific analysis, shown for a paired design: within each cell type every participant has
 * a value at baseline and at follow-up, each pair is reduced to one change from baseline, and the
 * mean change is drawn with its confidence interval. Schematic: the grains and the values illustrate
 * and show no data. The interval of a row is the t interval of the six changes drawn in that row.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, grain, sphere, line, floorLabel } = S;
  // per participant: a schematic baseline level, the change from baseline, and the line the pair
  // takes while the pairs are shown. The sample on the left is the baseline of the pair on line 0.
  const TYPES = [
    { name: "T cells", color: "#4a4946", n: 130, y: 250,
      base: [0.10, -0.22, 0.26, -0.06, -0.30, 0.18], change: [0.19, 0.37, 0.55, 0.72, 0.85, 1.03], slot: [2, 5, 0, 3, 1, 4] },
    { name: "B cells", color: "#8f8e89", n: 80, y: 490,
      base: [0.22, -0.10, 0.30, -0.26, 0.06, -0.18], change: [-0.45, -0.27, 0.05, 0.19, 0.37, 0.70], slot: [4, 1, 3, 0, 5, 2] },
    { name: "myeloid cells", color: "#c2c0b9", n: 105, y: 730,
      base: [0.02, 0.26, -0.14, 0.30, -0.26, 0.14], change: [-0.94, -0.68, -0.55, -0.38, -0.25, -0.09], slot: [1, 4, 2, 5, 0, 3] },
  ];
  const XC = 270, ZERO = 1370, MID = 1350, UNIT = 400, GAP = 28, RB = 11, RS = 14, RP = 18, UP = 30, DOWN = 36;
  const T975 = 2.571;                                       // 0.975 quantile of the t distribution, 5 degrees of freedom

  S.register({
    id: "cell-types",
    title: "Cell-type-specific analysis",
    caption: "The same design-matched comparison is applied within each cell type, with one value per participant, visit and cell type. The example shows a paired design: the two visits of each participant are reduced to one change from baseline, and the mean change is reported with its confidence interval for each cell type. The change in cell-type composition can be tested in the same framework.",
    reel: [2.6, 9.6],
    prepare() {
      const rand = S.rng(9), grains = [];
      const disc = (cx, cy, rx, ry = rx) => { const a = rand() * Math.PI * 2, d = Math.sqrt(rand()); return [cx + Math.cos(a) * d * rx, cy + Math.sin(a) * d * ry]; };
      TYPES.forEach((type, k) => {
        for (let i = 0; i < type.n; i++) grains.push({ k, color: type.color, delay: rand(), mixed: disc(S.W / 2, 490, 700, 300), sorted: disc(XC, type.y, 48 + type.n * 0.55) });
      });
      const types = TYPES.map((type) => {
        const n = type.change.length, est = S.mean(type.change);
        const sd = Math.sqrt(type.change.reduce((s, v) => s + (v - est) ** 2, 0) / (n - 1)), half = T975 * sd / Math.sqrt(n);
        // while the pairs are shown they are centred in the plot; afterwards every baseline sits on the zero line
        const after = type.base.map((b, j) => b + type.change[j]), all = type.base.concat(after);
        const origin = MID - (Math.min(...all) + Math.max(...all)) / 2 * UNIT;
        const pairs = type.base.map((b, j) => ({
          xb: origin + b * UNIT, xf: origin + after[j] * UNIT, xc: ZERO + type.change[j] * UNIT,
          y: type.y + (type.slot[j] - 2.5) * GAP, first: type.slot[j] === 0, order: type.slot[j],
        }));
        return { ...type, est, half, pairs, guide: Math.min(Math.min(...pairs.map((p) => p.xc)) - RP - 44, ZERO - 34) };
      });
      return { grains, types };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent, G = S.TONES.grey, base = ctx.globalAlpha, appear = base * seg(t, 0, 0.8);
      for (const g of d.grains) {
        const k = ease(seg(t, 0.8 + g.k * 0.2 + g.delay * 0.3, 1.9 + g.k * 0.2 + g.delay * 0.3));
        ctx.globalAlpha = appear;
        grain(ctx, lerp(g.mixed[0], g.sorted[0], k), lerp(g.mixed[1], g.sorted[1], k), 5.2, g.color);
      }
      ctx.globalAlpha = base;
      floorLabel(ctx, "cells of one sample", S.W / 2, seg(t, 0.2, 0.8) * (1 - seg(t, 1.5, 2.0)));
      floorLabel(ctx, "by cell type", XC, seg(t, 2.2, 2.8));
      floorLabel(ctx, "each participant at baseline and follow-up", ZERO, seg(t, 3.3, 3.9) * (1 - seg(t, 5.2, 5.6)));
      floorLabel(ctx, "change from baseline, per participant", ZERO, seg(t, 5.7, 6.3));
      floorLabel(ctx, "mean change, with its confidence interval", ZERO, seg(t, 7.2, 7.8), 2, A.text);

      // the line of no change: every baseline is moved onto it
      const zero = ease(seg(t, 5.0, 5.7));
      line(ctx, ZERO, 150, ZERO, lerp(150, 830, zero), S.RULE, 3.5, [6, 13], zero);
      text(ctx, "no change", ZERO, 104, 38, { align: "center", color: S.SOFT, alpha: zero });

      const settle = ease(seg(t, 6.5, 7.2));
      d.types.forEach((type, k) => {
        text(ctx, type.name, XC + 250, type.y, 48, { alpha: seg(t, 2.2 + k * 0.15, 2.8 + k * 0.15) });
        line(ctx, XC + 600, type.y, lerp(XC + 600, type.guide, settle), type.y, G.line, 3, [4, 11], 0.9 * settle);
        const slide = ease(seg(t, 5.3 + k * 0.1, 6.3 + k * 0.1));
        type.pairs.forEach((p) => {
          // the pair appears: for the sample shown on the left, its baseline value leaves the cells
          const start = p.first ? 2.7 + k * 0.12 : 3.9 + k * 0.08 + p.order * 0.1;
          const shown = p.first ? ease(seg(t, start, start + 0.8)) : S.easeOut(seg(t, start, start + 0.35));
          const grown = p.first ? ease(seg(t, start + 0.8, start + 1.2)) : shown;
          if (shown <= 0) return;
          const xb = lerp(p.xb, ZERO, slide), xf = lerp(p.xf, p.xc, slide), y = lerp(p.y, type.y - UP, settle);
          const bx = p.first ? lerp(XC, xb, shown) : xb, by = p.first ? lerp(type.y - UP, y, shown) - 40 * Math.sin(Math.PI * shown) : y;
          const pale = (p.first ? seg(shown, 0, 0.15) : shown) * (1 - settle);
          if (grown > 0) line(ctx, bx, by, lerp(bx, xf, grown), y, G.text, 4, null, 0.75 * grown * (1 - settle));
          sphere(ctx, bx, by, RB, "grey", 0.7 * pale, 0);                                   // baseline
          if (grown > 0) sphere(ctx, lerp(bx, xf, grown), y, lerp(RS, RP, settle), "grey", grown, 0);   // follow-up, then the change
          if (p.first && k === 0) {
            const named = (1 - seg(t, 5.1, 5.5));
            text(ctx, "baseline", p.xb, p.y - 44, 38, { align: "center", color: S.SOFT, alpha: seg(t, 3.5, 4.0) * named });
            text(ctx, "follow-up", p.xf, p.y - 44, 38, { align: "center", color: S.SOFT, alpha: seg(t, 3.9, 4.4) * named });
          }
        });
        // the mean change and its confidence interval, from the changes of the row
        const e = ease(seg(t, 7.2 + k * 0.4, 8.0 + k * 0.4)), x = ZERO + type.est * UNIT, h = type.half * UNIT * e, y = type.y + DOWN;
        if (e <= 0) return;
        line(ctx, x - h, y, x + h, y, A.text, 8);
        line(ctx, x - h, y - 18, x - h, y + 18, A.text, 8);
        line(ctx, x + h, y - 18, x + h, y + 18, A.text, 8);
        sphere(ctx, x, y, 21 * e, "accent", e, 0);
      });
    },
  });
})(window.Studio);
