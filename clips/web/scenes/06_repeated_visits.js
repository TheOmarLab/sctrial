/* Repeated visits: two arms followed over four visits, each later visit compared with baseline.
 * Schematic: the positions illustrate the comparison and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, level, text, sphere, panel, line, poly, arrow, floorLabel } = S;
  const XS = [390, 800, 1210, 1620], NAMES = ["baseline", "visit 2", "visit 3", "visit 4"], R = 20;
  const ARMS = [
    { tone: "grey", base: [0.18, 0.30, 0.42, 0.54], path: [0, 0.02, 0.03, 0.03],
      wobble: [[0, 0.02, -0.01, 0.01], [0, -0.02, 0.01, -0.01], [0, 0.01, 0.02, 0.0], [0, -0.01, -0.02, 0.0]] },
    { tone: "accent", base: [0.22, 0.34, 0.38, 0.50], path: [0, 0.15, 0.27, 0.34],
      wobble: [[0, 0.03, -0.02, 0.02], [0, -0.03, 0.02, -0.02], [0, 0.02, 0.03, 0.01], [0, -0.02, -0.03, -0.01]] },
  ];

  S.register({
    id: "repeated-visits",
    title: "Repeated visits",
    caption: "With more than two visits, a difference-in-differences is estimated at each later visit relative to baseline, or the difference between the arms in the trend over time is tested. When two or more visits precede treatment, parallel trends before treatment can be examined.",
    reel: [3.4, 9.4],
    prepare() {
      const people = ARMS.flatMap((arm, a) => arm.base.map((b, i) => ({ arm: a, tone: arm.tone,
        ys: arm.path.map((p, k) => level(b + p + arm.wobble[i][k])) })));
      XS.forEach((x, k) => S.swarm(people.map((p) => p.ys[k]), R).forEach((dx, i) => { (people[i].xs || (people[i].xs = []))[k] = x + dx; }));
      const means = [0, 1].map((a) => XS.map((x, k) => [x, S.mean(people.filter((p) => p.arm === a).map((p) => p.ys[k]))]));
      // at each later visit: the treated arm, had it changed from baseline as the control arm did
      const expected = XS.map((x, k) => means[1][0][1] + (means[0][k][1] - means[0][0][1]));
      return { people, means, expected };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent, G = S.TONES.grey, start = (k) => 1.4 + (k - 1) * 1.25;
      XS.forEach((x, k) => {
        panel(ctx, x, 1, 240);
        floorLabel(ctx, NAMES[k], x, k === 0 ? seg(t, 0.1, 0.7) : seg(t, start(k) - 0.2, start(k) + 0.4));
      });
      const dim = 1 - 0.45 * seg(t, 5.3, 6.1);
      d.people.forEach((p, i) => {
        sphere(ctx, p.xs[0], p.ys[0], R, p.tone, S.easeOut(seg(t, 0.2 + i * 0.07, 0.8 + i * 0.07)) * dim, 0.22);
        for (let k = 1; k < XS.length; k++) {
          const go = ease(seg(t, start(k) + i * 0.03, start(k) + 0.95 + i * 0.03));
          if (go <= 0) break;
          const x = lerp(p.xs[k - 1], p.xs[k], go), y = lerp(p.ys[k - 1], p.ys[k], go);
          line(ctx, p.xs[k - 1], p.ys[k - 1], x, y, S.TONES[p.tone].line, 3.2, null, dim);
          sphere(ctx, x, y, R, p.tone, dim, 0.22);
        }
      });
      // the mean of each arm at each visit
      const drawn = ease(seg(t, 5.3, 6.7));
      poly(ctx, d.means[0], G.flat, 11, drawn);
      poly(ctx, d.means[1], A.flat, 11, drawn);
      text(ctx, "control", XS[0] - 150, d.means[0][0][1] + 46, 44, { align: "right", color: G.text, alpha: seg(t, 0.9, 1.5) });
      text(ctx, "treated", XS[0] - 150, d.means[1][0][1] - 46, 44, { align: "right", color: A.text, alpha: seg(t, 0.9, 1.5) });
      // each later visit against baseline: the change in the treated arm minus the change in the control arm
      for (let k = 1; k < XS.length; k++) {
        arrow(ctx, XS[k] + 140, d.expected[k], d.means[1][k][1], ease(seg(t, 6.8 + k * 0.4, 7.5 + k * 0.4)));
      }
      // the key of the arrows
      const key = seg(t, 7.4, 8.1);
      arrow(ctx, 146, 66, 126, 1, A.text, key);
      text(ctx, "DiD vs baseline", 186, 96, 48, { color: A.text, alpha: key });
    },
  });
})(window.Studio);
