/* Incomplete and unbalanced data: participants observed at one visit, and participants of an arm that
 * the design does not name, do not enter the paired comparison; the arms may differ in size.
 * Schematic: the positions illustrate the rule and show no pattern of change.
 */
(function (S) {
  "use strict";
  const { seg, ease, level, text, sphere, ring, panel, line, floorLabel } = S;
  const X0 = 310, X1 = 900, R = 30, XL = 1230;
  // six control and four treated participants: after the exclusions, five and three
  const PEOPLE = [
    { tone: "grey", b: 0.14, c: 0.10 }, { tone: "grey", b: 0.26, c: -0.05, missing: 1 }, { tone: "grey", b: 0.38, c: 0.13 },
    { tone: "grey", b: 0.50, c: -0.06 }, { tone: "grey", b: 0.62, c: 0.09 }, { tone: "grey", b: 0.76, c: 0.02 },
    { tone: "accent", b: 0.20, c: 0.15 }, { tone: "accent", b: 0.44, c: -0.07 },
    { tone: "accent", b: 0.56, c: 0.11, missing: 0 }, { tone: "accent", b: 0.69, c: 0.12 },
    { tone: "dark", b: 0.04, c: 0.07, other: true },
  ];

  S.register({
    id: "incomplete",
    title: "Incomplete and unbalanced data",
    caption: "A paired comparison uses the participants observed at both visits in the two arms named by the design, and each result reports the number of participants analyzed. Participants of any other arm are excluded with a warning. The arms may differ in size.",
    reel: [2.6, 9.4],
    prepare() {
      const people = PEOPLE.map((p) => ({ ...p, y0: level(p.b), y1: level(p.b + p.c), out: p.missing !== undefined || Boolean(p.other) }));
      S.swarm(people.map((p) => p.y0), R).forEach((dx, i) => { people[i].x0 = X0 + dx; });
      S.swarm(people.map((p) => p.y1), R).forEach((dx, i) => { people[i].x1 = X1 + dx; });
      return { people };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent;
      panel(ctx, X0, 1, 320); panel(ctx, X1, 1, 320);
      floorLabel(ctx, "baseline", X0, seg(t, 0.1, 0.7));
      floorLabel(ctx, "follow-up", X1, seg(t, 0.1, 0.7));
      const found = seg(t, 2.4, 3.1), other = seg(t, 4.6, 5.3), kept = seg(t, 6.2, 7.0);
      d.people.forEach((p, i) => {
        const appear = S.easeOut(seg(t, 0.3 + i * 0.06, 0.9 + i * 0.06)), drawn = ease(seg(t, 1.2 + i * 0.05, 2.1 + i * 0.05));
        const fade = p.other ? 1 - 0.78 * other : p.out ? 1 - 0.78 * seg(t, 3.4, 4.2) : 1;
        if (p.missing === undefined) {
          line(ctx, p.x0, p.y0, p.x0 + (p.x1 - p.x0) * drawn, p.y0 + (p.y1 - p.y0) * drawn, p.out ? S.TONES.dark.line : S.TONES[p.tone].line,
               p.out ? 3.5 : 3.5 + 3 * kept, null, fade);
        }
        if (p.missing !== 0) sphere(ctx, p.x0, p.y0, R, p.tone, appear * fade, 0.22 * fade); else ring(ctx, p.x0, p.y0, R, S.INK, found);
        if (p.missing !== 1) sphere(ctx, p.x1, p.y1, R, p.tone, appear * fade, 0.22 * fade); else ring(ctx, p.x1, p.y1, R, S.INK, found);
      });
      // what is left out, and what is analysed
      text(ctx, "excluded", XL, 250, 38, { color: S.SOFT, alpha: found });
      ring(ctx, XL + 28, 334, 24, S.INK, found);
      text(ctx, "observed at one visit", XL + 82, 334, 46, { alpha: found });
      sphere(ctx, XL + 28, 424, 24, "dark", other, 0);
      text(ctx, "arm not in the design", XL + 82, 424, 46, { alpha: other });
      text(ctx, "analysis set", XL, 574, 38, { color: A.text, alpha: kept });
      line(ctx, XL + 18, 664, XL + 124, 634, S.TONES.grey.line, 6, null, kept);
      sphere(ctx, XL + 18, 664, 17, "grey", kept, 0); sphere(ctx, XL + 124, 634, 17, "grey", kept, 0);
      text(ctx, "complete pairs", XL + 168, 650, 46, { alpha: kept });
    },
  });
})(window.Studio);
