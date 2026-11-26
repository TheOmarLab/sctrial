/* Single-arm longitudinal design: each participant is compared with their own baseline.
 * Schematic: the positions illustrate the comparison and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, level, text, sphere, panel, line, bracket, floorLabel } = S;
  const X0 = 330, X1 = 930, XD = 1510, R = 30, RD = 23, ZERO = 660, UNIT = 1050;
  const BASE = [0.05, 0.17, 0.29, 0.41, 0.53, 0.65, 0.77], CHANGE = [0.21, 0.12, 0.27, 0.06, 0.19, 0.20, -0.04];
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
      panel(ctx, X0, 1, 300); panel(ctx, X1, 1, 300); panel(ctx, XD, 1, 300);
      floorLabel(ctx, "baseline", X0, seg(t, 0.1, 0.7));
      floorLabel(ctx, "follow-up", X1, seg(t, 1.4, 2.1));

      const shift = ease(seg(t, 4.3, 5.8));
      d.people.forEach((p, i) => {
        const appear = S.easeOut(seg(t, 0.25 + i * 0.09, 0.9 + i * 0.09));
        const go = ease(seg(t, 1.4 + i * 0.22, 3.0 + i * 0.22));
        const x = lerp(p.x0, p.x1, go), y = lerp(p.y0, p.y1, go);
        if (go > 0) line(ctx, p.x0, p.y0, x, y, A.line, 4);
        sphere(ctx, p.x0, p.y0, R, "accent", appear * (1 - 0.5 * seg(go, 0, 0.2)), 0.22);    // the participant's own baseline
        if (go > 0) sphere(ctx, x, y, R, "accent", 1, 0.22);
        if (shift > 0) sphere(ctx, lerp(p.x1, p.xd, shift), lerp(p.y1, p.yd, shift), lerp(R, RD, shift), "accent", shift, 0);
      });

      const zero = seg(t, 4.3, 5.0);
      line(ctx, XD - 170, ZERO, XD + 170, ZERO, S.RULE, 3.5, [6, 13], zero);
      text(ctx, "no change", XD - 196, ZERO, 40, { align: "right", color: S.SOFT, alpha: zero });
      floorLabel(ctx, "change", XD, zero);

      const m = ease(seg(t, 6.1, 7.0));
      if (m > 0) {
        line(ctx, XD - 130 * m, d.mean, XD + 130 * m, d.mean, A.flat, 12);
        bracket(ctx, XD + 220, d.lo, d.hi, m);
      }
      text(ctx, "mean change, with its confidence interval", S.W - 130, 96, 48, { align: "right", color: A.text, alpha: seg(t, 6.7, 7.4) });
    },
  });
})(window.Studio);
