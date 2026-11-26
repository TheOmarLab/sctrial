/* One design definition: the cells of a study are organised by participant, visit and arm.
 * Schematic: the grains stand for cells and show no data. A participant is one cluster, named below it
 * at both visits; the visit is the left or right block; the arm is the row and the colour.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, grain, floorLabel } = S;
  const PEOPLE = 8, PER_SAMPLE = 56, PANELS = [690, 1480], ROWS = [320, 660];
  const GREY = S.TONES.grey.flat, AMBER = S.TONES.accent.flat;
  const person = (i) => [260 + i * 200, 490];
  const sample = (i, v) => [PANELS[v] + (i % 4 - 1.5) * 172, ROWS[Math.floor(i / 4)]];

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
            grains.push({ treated: i >= PEOPLE / 2, delay: rand(), order: i / PEOPLE, cloud: disc(S.W / 2, 490, 740, 300),
                          person: disc(...person(i), 80), sample: disc(...sample(i, v), 78) });
          }
        }
      }
      return { grains };
    },
    draw(ctx, t, d) {
      const on = [seg(t, 1.3, 1.8), seg(t, 3.3, 3.8), seg(t, 5.3, 5.8)];
      S.steps(ctx, ["participant", "visit", "arm"], on, seg(t, 0.2, 0.8));
      const base = ctx.globalAlpha, appear = base * seg(t, 0, 0.9);
      for (const g of d.grains) {
        const k1 = ease(seg(t, 1.3 + g.order * 0.5 + g.delay * 0.4, 2.7 + g.order * 0.5 + g.delay * 0.4));
        const k2 = ease(seg(t, 3.3 + g.delay * 0.5, 4.8 + g.delay * 0.5));
        const x = lerp(lerp(g.cloud[0], g.person[0], k1), g.sample[0], k2);
        const y = lerp(lerp(g.cloud[1], g.person[1], k1), g.sample[1], k2);
        const tint = g.treated ? seg(t, 5.3 + g.delay * 0.4, 6.0 + g.delay * 0.4) : 0;
        ctx.globalAlpha = appear;
        grain(ctx, x, y, 5, GREY);
        if (tint > 0) { ctx.globalAlpha = appear * tint; grain(ctx, x, y, 5, AMBER); }
      }
      ctx.globalAlpha = base;
      // the name of a participant stands under its cluster, then under each of its two samples
      const together = 1 - seg(t, 3.3, 3.6), apart = seg(t, 4.7, 5.2);
      for (let i = 0; i < PEOPLE; i++) {
        const name = `P${i + 1}`, style = { align: "center", color: S.SOFT };
        const [x, y] = person(i);
        text(ctx, name, x, y + 124, 34, { ...style, alpha: seg(t, 2.1 + i / PEOPLE * 0.5, 2.7 + i / PEOPLE * 0.5) * together });
        for (let v = 0; v < 2; v++) {
          const [sx, sy] = sample(i, v);
          text(ctx, name, sx, sy + 122, 34, { ...style, alpha: apart });
        }
      }
      const a = seg(t, 4.6, 5.3), b = seg(t, 6.0, 6.7);
      floorLabel(ctx, "baseline", PANELS[0], a);
      floorLabel(ctx, "follow-up", PANELS[1], a);
      text(ctx, "control", 330, ROWS[0], 48, { align: "right", color: S.TONES.grey.text, alpha: b });
      text(ctx, "treated", 330, ROWS[1], 48, { align: "right", color: S.TONES.accent.text, alpha: b });
    },
  });
})(window.Studio);
