/* One design definition: the cells of a study are organised by participant, visit and arm.
 * Schematic: the grains stand for cells and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, grain, floorLabel } = S;
  const PEOPLE = 8, PER_SAMPLE = 40, PANELS = [650, 1400], ROWS = [330, 660];
  const GREY = S.TONES.grey.flat, AMBER = S.TONES.accent.flat;

  S.register({
    id: "design",
    title: "One design definition",
    caption: "The participant, the visit and the arm of every cell are declared once. Each analysis then applies the comparison that matches the design.",
    reel: [4.2, 9.4],
    prepare() {
      const rand = S.rng(1), grains = [];
      const disc = (cx, cy, rx, ry = rx) => { const a = rand() * Math.PI * 2, d = Math.sqrt(rand()); return [cx + Math.cos(a) * d * rx, cy + Math.sin(a) * d * ry]; };
      for (let i = 0; i < PEOPLE; i++) {
        for (let v = 0; v < 2; v++) {
          for (let k = 0; k < PER_SAMPLE; k++) {
            grains.push({ treated: i >= PEOPLE / 2, delay: rand(), order: i / PEOPLE, cloud: disc(S.W / 2, 470, 720, 290),
                          person: disc(260 + i * 200, 470, 78), sample: disc(PANELS[v] + (i % 4 - 1.5) * 156, ROWS[Math.floor(i / 4)], 58) });
          }
        }
      }
      return { grains };
    },
    draw(ctx, t, d) {
      const on = [seg(t, 1.3, 1.8), seg(t, 3.3, 3.8), seg(t, 5.3, 5.8)];
      S.steps(ctx, ["participant", "visit", "arm"], on, seg(t, 0.2, 0.8));
      const appear = seg(t, 0, 0.9);
      for (const g of d.grains) {
        const k1 = ease(seg(t, 1.3 + g.order * 0.5 + g.delay * 0.4, 2.7 + g.order * 0.5 + g.delay * 0.4));
        const k2 = ease(seg(t, 3.3 + g.delay * 0.5, 4.8 + g.delay * 0.5));
        const x = lerp(lerp(g.cloud[0], g.person[0], k1), g.sample[0], k2);
        const y = lerp(lerp(g.cloud[1], g.person[1], k1), g.sample[1], k2);
        const tint = g.treated ? seg(t, 5.3 + g.delay * 0.4, 6.0 + g.delay * 0.4) : 0;
        ctx.globalAlpha = appear;
        grain(ctx, x, y, 4.6, GREY);
        if (tint > 0) { ctx.globalAlpha = appear * tint; grain(ctx, x, y, 4.6, AMBER); }
      }
      ctx.globalAlpha = 1;
      const a = seg(t, 4.6, 5.3), b = seg(t, 6.0, 6.7);
      floorLabel(ctx, "baseline", PANELS[0], a);
      floorLabel(ctx, "follow-up", PANELS[1], a);
      text(ctx, "control", 300, ROWS[0], 46, { align: "right", color: S.TONES.grey.text, alpha: b });
      text(ctx, "treated", 300, ROWS[1], 46, { align: "right", color: S.TONES.accent.text, alpha: b });
    },
  });
})(window.Studio);
