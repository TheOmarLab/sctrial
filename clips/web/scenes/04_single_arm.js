/* Single-arm longitudinal design: each participant is compared with their own baseline.
 * Schematic: the positions illustrate the comparison and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, level, text, sphere, panel, line, bracket, floorLabel } = S;
  const X0 = 310, X1 = 760, XD = 1190, R = 28, RD = 20, ZERO = 650, UNIT = 780;
  const BASE = [0.16, 0.26, 0.36, 0.45, 0.54, 0.63, 0.72], CHANGE = [0.21, 0.12, 0.27, 0.06, 0.19, 0.23, -0.03];
  const HALF = 0.075;                                       // schematic half-width of the interval

  S.register({
    id: "single-arm",
    title: "Single-arm longitudinal design",
    caption: "Without a control arm, each participant serves as their own control. The estimand is the mean within-participant change between two visits, tested against no change.",
    reel: [3.6, 9.4],
    prepare() {
      const people = BASE.map((b, i) => ({ y0: level(b), y1: level(b + CHANGE[i]), yd: ZERO - CHANGE[i] * UNIT }));
      S.swarm(people.map((p) => p.y0), R).forEach((dx, i) => { people[i].x0 = X0 + dx; });
      S.swarm(people.map((p) => p.y1), R).forEach((dx, i) => { people[i].x1 = X1 + dx; });
      S.swarm(people.map((p) => p.yd), RD).forEach((dx, i) => { people[i].xd = XD + dx; });
      const mean = ZERO - S.mean(CHANGE) * UNIT;
      return { people, mean, lo: mean + HALF * UNIT, hi: mean - HALF * UNIT };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent;
      panel(ctx, X0); panel(ctx, X1);
      floorLabel(ctx, "baseline", X0, seg(t, 0.1, 0.7));
      floorLabel(ctx, "follow-up", X1, seg(t, 1.4, 2.1));

      const shift = ease(seg(t, 4.3, 5.8));
      d.people.forEach((p, i) => {
        const appear = S.easeOut(seg(t, 0.25 + i * 0.09, 0.9 + i * 0.09));
        const go = ease(seg(t, 1.4 + i * 0.22, 3.0 + i * 0.22));
        const x = lerp(p.x0, p.x1, go), y = lerp(p.y0, p.y1, go);
        if (go > 0) line(ctx, p.x0, p.y0, x, y, A.line, 4);
        sphere(ctx, p.x0, p.y0, R, "accent", appear * (1 - 0.5 * seg(go, 0, 0.2)), 0.22);     // the participant's own baseline
        if (go > 0) sphere(ctx, x, y, R, "accent", 1, 0.22);
        if (shift > 0) sphere(ctx, lerp(p.x1, p.xd, shift), lerp(p.y1, p.yd, shift), lerp(R, RD, shift), "accent", shift, 0);
      });

      const zero = seg(t, 4.3, 5.0);
      line(ctx, XD - 150, ZERO, XD + 150, ZERO, S.RULE, 3.5, [6, 13], zero);
      text(ctx, "no change", XD - 172, ZERO, 38, { align: "right", color: S.SOFT, alpha: zero });
      floorLabel(ctx, "within-participant change", XD, zero);

      const m = ease(seg(t, 6.1, 7.0));
      if (m > 0) {
        line(ctx, XD - 115 * m, d.mean, XD + 115 * m, d.mean, A.flat, 11);
        bracket(ctx, XD + 180, d.lo, d.hi, m);
      }
      text(ctx, "mean change", XD + 226, d.mean - 26, 50, { alpha: seg(t, 6.7, 7.4) });
      text(ctx, "with its confidence interval", XD + 228, d.mean + 30, 34, { color: S.SOFT, alpha: seg(t, 7.2, 7.9) });
    },
  });
})(window.Studio);
