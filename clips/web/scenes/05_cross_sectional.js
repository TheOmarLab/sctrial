/* Cross-sectional design: two groups of participants compared at a single time point.
 * Schematic: the positions illustrate the comparison and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, level, text, sphere, panel, line, arrow, floorLabel } = S;
  const XC = 960, XA = 500, XB = 1340, R = 30;
  const GROUPS = [
    { tone: "grey", values: [0.05, 0.15, 0.23, 0.30, 0.37, 0.44, 0.52, 0.63] },
    { tone: "accent", values: [0.40, 0.53, 0.62, 0.70, 0.79, 0.94] },
  ];

  S.register({
    id: "cross-sectional",
    title: "Cross-sectional design",
    caption: "Each participant is sampled once. Two groups are compared at that time point on participant-level values, and the difference is reported in standard deviation units.",
    reel: [3.2, 9.4],
    prepare() {
      const people = GROUPS.flatMap((g, k) => g.values.map((v) => ({ group: k, tone: g.tone, y: level(v) })));
      S.swarm(people.map((p) => p.y), R).forEach((dx, i) => { people[i].x = XC + dx; });
      [XA, XB].forEach((x, k) => {
        const own = people.filter((p) => p.group === k);
        S.swarm(own.map((p) => p.y), R).forEach((dx, i) => { own[i].xg = x + dx; });
      });
      const mean = (k) => S.mean(people.filter((p) => p.group === k).map((p) => p.y));
      return { people, a: mean(0), b: mean(1) };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent, G = S.TONES.grey, split = ease(seg(t, 2.0, 3.6));
      panel(ctx, XC, 1 - split, 380); panel(ctx, XA, split, 380); panel(ctx, XB, split, 380);
      floorLabel(ctx, "one visit", XC, seg(t, 0.2, 0.8) * (1 - seg(t, 1.9, 2.4)));
      floorLabel(ctx, "group 1", XA, seg(t, 2.6, 3.4));
      floorLabel(ctx, "group 2", XB, seg(t, 2.6, 3.4), 1, A.text);

      const dim = 1 - 0.4 * seg(t, 4.0, 4.8), turn = seg(t, 2.2, 3.2);
      d.people.forEach((p, i) => {
        const appear = S.easeOut(seg(t, 0.3 + i * 0.05, 1.0 + i * 0.05));
        const x = lerp(p.x, p.xg, split), y = p.y - (1 - appear) * 60;
        sphere(ctx, x, y, R, "grey", appear * dim * (p.group ? 1 - turn : 1), p.group ? 0 : 0.22);
        if (p.group) sphere(ctx, x, y, R, "accent", appear * dim * turn, 0.22);
      });

      const k = ease(seg(t, 4.0, 5.0));
      if (k > 0) {
        line(ctx, XA - 170 * k, d.a, XA + 170 * k, d.a, G.flat, 12);
        line(ctx, XB - 170 * k, d.b, XB + 170 * k, d.b, A.flat, 12);
      }
      const c = ease(seg(t, 5.2, 6.0)), xb = XB + 430;
      if (c > 0) {
        line(ctx, XA + 170, d.a, lerp(XA + 170, xb + 18, c), d.a, S.RULE, 3.5, [6, 13]);
        line(ctx, XB + 170, d.b, lerp(XB + 170, xb + 18, c), d.b, S.RULE, 3.5, [6, 13]);
      }
      arrow(ctx, xb, d.a, d.b, ease(seg(t, 6.0, 6.8)));
      text(ctx, "difference between groups, in SD units", 130, 96, 48, { color: A.text, alpha: seg(t, 6.5, 7.2) });
    },
  });
})(window.Studio);
