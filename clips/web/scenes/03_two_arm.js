/* Two-arm longitudinal design: the change in the treated arm against the change in the control arm.
 * Schematic: the positions illustrate the comparison and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, level, text, sphere, panel, line, bracket, floorLabel } = S;
  const X0 = 480, X1 = 1100, R = 28;
  // schematic outcome levels: baseline and change of each participant
  const ARMS = [
    { tone: "grey", base: [0.16, 0.28, 0.40, 0.52, 0.64], change: [0.05, 0.08, 0.02, 0.07, 0.04] },
    { tone: "accent", base: [0.22, 0.34, 0.46, 0.58, 0.70], change: [0.24, 0.28, 0.21, 0.26, 0.22] },
  ];

  S.register({
    id: "two-arm",
    title: "Two-arm longitudinal design",
    caption: "Two arms are observed at baseline and at follow-up. The estimand is the difference-in-differences: the mean change in the treated arm minus the mean change in the control arm, estimated from one value per participant and visit.",
    reel: [3.6, 9.4],
    prepare() {
      const people = ARMS.flatMap((arm, a) => arm.base.map((b, i) => ({ arm: a, tone: arm.tone, y0: level(b), y1: level(b + arm.change[i]) })));
      S.swarm(people.map((p) => p.y0), R).forEach((dx, i) => { people[i].x0 = X0 + dx; });
      S.swarm(people.map((p) => p.y1), R).forEach((dx, i) => { people[i].x1 = X1 + dx; });
      const mean = (a, key) => S.mean(people.filter((p) => p.arm === a).map((p) => p[key]));
      const m = { c0: mean(0, "y0"), c1: mean(0, "y1"), t0: mean(1, "y0"), t1: mean(1, "y1") };
      m.expected = m.t0 + (m.c1 - m.c0);                   // the treated arm, had it changed as the control arm
      return { people, m };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent, G = S.TONES.grey, m = d.m;
      panel(ctx, X0); panel(ctx, X1);
      floorLabel(ctx, "baseline", X0, seg(t, 0.1, 0.7));
      floorLabel(ctx, "follow-up", X1, seg(t, 1.5, 2.2));

      const dim = 1 - 0.45 * seg(t, 4.2, 5.0);
      d.people.forEach((p, i) => {
        const appear = S.easeOut(seg(t, 0.25 + i * 0.07, 0.9 + i * 0.07));
        const go = ease(seg(t, 1.5 + i * 0.09, 3.6 + i * 0.09));
        const x = lerp(p.x0, p.x1, go), y = lerp(p.y0, p.y1, go);
        if (go > 0) line(ctx, p.x0, p.y0, x, y, S.TONES[p.tone].line, 4, null, dim);
        sphere(ctx, p.x0, p.y0, R, p.tone, appear * (1 - 0.4 * seg(go, 0, 0.15)) * dim, 0.22);
        if (go > 0) sphere(ctx, x, y, R, p.tone, dim, 0.22);
      });
      const named = seg(t, 0.9, 1.5);
      text(ctx, "treated", X0 - 170, m.t0 - 44, 46, { align: "right", color: A.text, alpha: named });
      text(ctx, "control", X0 - 170, m.c0 + 44, 46, { align: "right", color: G.text, alpha: named });

      // the mean change of each arm
      const k = ease(seg(t, 4.2, 5.4));
      if (k > 0) {
        line(ctx, X0, m.c0, lerp(X0, X1, k), lerp(m.c0, m.c1, k), G.flat, 11);
        line(ctx, X0, m.t0, lerp(X0, X1, k), lerp(m.t0, m.t1, k), A.flat, 11);
      }
      // the treated arm, had it changed as the control arm
      const c = ease(seg(t, 5.6, 6.7));
      if (c > 0) line(ctx, X0, m.t0, lerp(X0, X1, c), lerp(m.t0, m.expected, c), S.RULE, 4.5, [6, 15]);
      bracket(ctx, X1 + 170, m.expected, m.t1, ease(seg(t, 6.9, 7.8)));
      text(ctx, "difference-in-differences", X1 + 218, (m.expected + m.t1) / 2, 46, { alpha: seg(t, 7.4, 8.1) });
    },
  });
})(window.Studio);
