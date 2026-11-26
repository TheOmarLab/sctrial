/* Power and sample size: power as a function of the number of participants per arm, and the smallest
 * number that reaches the target. Schematic: the curve is the normal-approximation power of a
 * two-group comparison for an illustrative effect; no axis carries a number.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, sphere, line, poly, floorLabel } = S;
  const XA = 340, XB = 1560, Y0 = S.FLOOR - 60, Y1 = 170, N0 = 3, N1 = 26, EFFECT = 1.0, TARGET = 0.8, STEP = 25;
  // standard normal distribution function (Abramowitz and Stegun 26.2.17)
  const phi = (z) => {
    const k = 1 / (1 + 0.2316419 * Math.abs(z)), d = 0.3989422804 * Math.exp(-z * z / 2);
    const p = d * k * (0.31938153 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
    return z >= 0 ? 1 - p : p;
  };
  const power = (n) => phi(EFFECT * Math.sqrt(n / 2) - 1.959964);

  S.register({
    id: "power",
    title: "Power and sample size",
    caption: "For each design, power is computed from the expected effect, the standard deviation of the outcome and the number of participants. The single-arm and two-group comparisons offer a normal approximation or an exact t reference distribution; the two-arm difference-in-differences uses the normal approximation. The smallest number of participants that reaches the target power and the minimum detectable effect are returned.",
    reel: [2.6, 9.4],
    prepare() {
      const x = (n) => XA + ((n - N0) / (N1 - N0)) * (XB - XA), y = (p) => Y0 - p * (Y0 - Y1);
      let need = N0;
      while (power(need) < TARGET) need++;
      const curve = [];
      for (let n = N0; n <= N1 + 1e-9; n += 0.25) curve.push([x(n), y(power(n)), n]);
      return { x, y, curve, need, upto: curve.findIndex((p) => p[2] >= need) / (curve.length - 1) };
    },
    draw(ctx, t, d) {
      const A = S.TONES.accent, frame = seg(t, 0.2, 0.9);
      line(ctx, XA - 40, Y0, XB + 40, Y0, S.RULE, 2.5, null, frame * 0.5);
      line(ctx, XA - 40, Y0, XA - 40, Y1 - 40, S.RULE, 2.5, null, frame * 0.5);
      text(ctx, "power", XA - 66, Y1 - 40, 44, { align: "right", color: S.SOFT, alpha: frame });
      floorLabel(ctx, "participants per arm", (XA + XB) / 2, frame);
      const target = ease(seg(t, 0.9, 1.7));
      line(ctx, XA - 40, d.y(TARGET), lerp(XA - 40, XB + 40, target), d.y(TARGET), S.RULE, 3.5, [6, 13], target);
      text(ctx, "target", XB + 66, d.y(TARGET), 44, { alpha: target });

      // the curve up to the sample size that reaches the target, then beyond it
      const run = ease(seg(t, 1.8, 5.6)) * d.upto, further = ease(seg(t, 6.6, 7.8));
      if (further > 0) poly(ctx, d.curve, S.TONES.grey.flat, 8, d.upto + further * (1 - d.upto));
      poly(ctx, d.curve, S.INK, 8, run);
      const at = run * (d.curve.length - 1), i = Math.min(d.curve.length - 2, Math.floor(at)), k = at - i;
      const px = lerp(d.curve[i][0], d.curve[i + 1][0], k), py = lerp(d.curve[i][1], d.curve[i + 1][1], k);
      const n = lerp(d.curve[i][2], d.curve[i + 1][2], k), reached = seg(t, 5.6, 6.1);
      if (run > 0) {
        // the cohort at this point of the curve: the two arms side by side
        for (let j = 0; j < Math.floor(n + 1e-6); j++) {
          sphere(ctx, px - 16, Y0 - 18 - j * STEP, 11.5, "grey", 1, 0);
          sphere(ctx, px + 16, Y0 - 18 - j * STEP, 11.5, "accent", 1, 0);
        }
        sphere(ctx, px, py, 19, "dark", 1 - reached, 0);
        sphere(ctx, px, py, 19 + 4 * reached, "accent", reached, 0);
      }
      if (reached > 0) line(ctx, px, py + 26, px, Y0 - 26 - d.need * STEP, A.text, 3.5, [6, 13], reached);
      text(ctx, "required sample size", d.x(d.need) + 70, Y0 - 190, 50, { alpha: seg(t, 5.9, 6.6) });
      text(ctx, "smallest cohort that reaches the target", d.x(d.need) + 72, Y0 - 132, 34, { color: S.SOFT, alpha: seg(t, 6.4, 7.1) });
    },
  });
})(window.Studio);
