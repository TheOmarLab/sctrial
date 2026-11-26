/* Participant-level inference: the cells of one participant at one visit become one value.
 * Schematic: the grains stand for cells and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, level, text, grain, sphere, panel, line, floorLabel } = S;
  const PANELS = [590, 1330], PER_SAMPLE = 34, R = 26;
  const GREY = S.TONES.grey.flat, AMBER = S.TONES.accent.flat;
  const ARMS = [
    { treated: false, base: [0.22, 0.40, 0.58], change: [0.04, 0.07, 0.03] },
    { treated: true, base: [0.28, 0.46, 0.64], change: [0.22, 0.26, 0.20] },
  ];

  S.register({
    id: "participants",
    title: "Participant-level inference",
    caption: "Cells from one participant are correlated and are not independent replicates. Each participant contributes one value per visit, and standard errors, tests and intervals rest on the number of participants.",
    reel: [2.6, 9.4],
    prepare() {
      const rand = S.rng(2), samples = [], grains = [];
      ARMS.forEach((arm, a) => arm.base.forEach((b, i) => [0, 1].forEach((v) => {
        const slot = a * 3 + i, x = PANELS[v] + (slot - 2.5) * 108, y = level(b + v * arm.change[i]);
        samples.push({ treated: arm.treated, visit: v, x, y });
        for (let k = 0; k < PER_SAMPLE; k++) {
          const ang = rand() * Math.PI * 2, r = Math.sqrt(rand()) * 42;
          grains.push({ s: samples.length - 1, treated: arm.treated, delay: rand(),
                        blob: [x + Math.cos(ang) * r, 470 + Math.sin(ang) * r],
                        strip: [x + (rand() - 0.5) * 36, y + (rand() + rand() + rand() - 1.5) * 72] });
        }
      })));
      return { samples, grains };
    },
    draw(ctx, t, d) {
      const base = ctx.globalAlpha, appear = base * seg(t, 0, 0.8), swap = seg(t, 4.3, 4.9);
      text(ctx, "cells", S.W - 130, 96, 48, { align: "right", alpha: seg(t, 0.2, 0.8) * (1 - seg(t, 4.0, 4.3)) });
      text(ctx, "one value per participant and visit", S.W - 130, 96, 48, { align: "right", alpha: swap, color: S.TONES.accent.text });

      const held = ease(seg(t, 5.2, 6.4));
      panel(ctx, PANELS[0], held, 720); panel(ctx, PANELS[1], held, 720);
      floorLabel(ctx, "baseline", PANELS[0], seg(t, 0.3, 1.0));
      floorLabel(ctx, "follow-up", PANELS[1], seg(t, 0.3, 1.0));

      // the cells of each sample: a cluster, then the spread of their values, then their mean
      for (const g of d.grains) {
        const s = d.samples[g.s];
        const k1 = ease(seg(t, 1.3 + g.delay * 0.5, 2.9 + g.delay * 0.5));
        const k2 = ease(seg(t, 3.5 + g.delay * 0.4, 4.6 + g.delay * 0.4));
        if (k2 >= 1) continue;
        const x = lerp(lerp(g.blob[0], g.strip[0], k1), s.x, k2), y = lerp(lerp(g.blob[1], g.strip[1], k1), s.y, k2);
        ctx.globalAlpha = appear * (1 - 0.85 * k2);
        grain(ctx, x, y, 4.4, g.treated ? AMBER : GREY);
      }
      ctx.globalAlpha = base;
      d.samples.forEach((s, i) => {
        const born = S.easeOut(seg(t, 4.0 + i * 0.03, 4.9 + i * 0.03));
        if (born <= 0) return;
        line(ctx, s.x, S.FLOOR, s.x, lerp(S.FLOOR, s.y + R, held), s.treated ? S.TONES.accent.line : S.TONES.grey.line, 3.5, null, held);
        sphere(ctx, s.x, s.y, 7 + (R - 7) * born, s.treated ? "accent" : "grey", born, 0.3 * born);
      });
    },
  });
})(window.Studio);
