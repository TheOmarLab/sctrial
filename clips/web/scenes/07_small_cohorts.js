/* Small cohorts: the reference distribution of the test statistic is built by flipping the sign of
 * each participant's contribution, over every sign assignment when the participants are few.
 * Schematic: six participants give 2^6 = 64 assignments; the values illustrate and show no data.
 */
(function (S) {
  "use strict";
  const { seg, ease, lerp, text, sphere, line, floorLabel } = S;
  const CHANGE = [0.21, 0.12, 0.27, 0.06, 0.19, 0.23], ROW = 190, XA = 150, XB = 1770, BINS = 17, DOT = 27;
  const BASE = S.FLOOR - 8, SLOW = 4, PITCH = 2 * DOT + 4;

  S.register({
    id: "small-cohorts",
    title: "Small cohorts",
    caption: "With few participants, large-sample approximations are unreliable. The test statistic is compared with its distribution under the null hypothesis, obtained with a wild cluster bootstrap that flips the sign of each participant's residuals. When the participants are few, every sign assignment is enumerated and the test is exact.",
    reel: [2.4, 9.4],
    prepare() {
      const n = CHANGE.length, all = 1 << n, limit = S.mean(CHANGE) * 1.06, height = new Array(BINS).fill(0);
      // every assignment of signs; the observed one (all positive) comes first
      const order = Array.from({ length: all }, (_, c) => all - 1 - c);
      const when = (k) => (k < SLOW ? 1.2 + k * 0.7 : 4.0 + 2.2 * Math.pow((k - SLOW) / (all - SLOW), 0.8));
      const draws = order.map((code, k) => {
        const signs = CHANGE.map((_, i) => ((code >> i) & 1 ? 1 : -1));
        const value = S.mean(CHANGE.map((v, i) => v * signs[i]));
        const bin = Math.min(BINS - 1, Math.floor(((value + limit) / (2 * limit)) * BINS));
        return { signs, x: XA + (bin + 0.5) * ((XB - XA) / BINS), y: BASE - height[bin]++ * PITCH - DOT, at: when(k), observed: k === 0 };
      });
      const people = CHANGE.map((_, i) => S.W / 2 + (i - (n - 1) / 2) * 200);
      return { draws, people, all, top: BASE - Math.max(...height) * PITCH };
    },
    draw(ctx, t, d) {
      let current = -1;
      for (let k = 0; k < d.all && d.draws[k].at <= t; k++) current = k;
      const signs = current >= 0 && t < 6.6 ? d.draws[current].signs : d.draws[0].signs;
      d.people.forEach((x, i) => {
        const appear = S.easeOut(seg(t, 0.2 + i * 0.06, 0.8 + i * 0.06));
        sphere(ctx, x, ROW, 34, "accent", appear, 0);
        text(ctx, signs[i] > 0 ? "+" : "−", x, ROW - 80, 60, { align: "center", alpha: appear * seg(t, 0.8, 1.2), color: signs[i] > 0 ? S.SOFT : S.INK });
      });
      floorLabel(ctx, `${current + 1} of ${d.all} sign assignments`, S.W / 2, seg(t, 1.1, 1.4), 2);
      for (const g of d.draws) {
        if (g.at > t) break;
        const k = ease(seg(t, g.at, g.at + 0.55));
        sphere(ctx, lerp(S.W / 2, g.x, k), lerp(ROW + 62, g.y, k * k), DOT, g.observed ? "accent" : "grey", 1, 0);
      }
      floorLabel(ctx, "bootstrap null distribution", S.W / 2, seg(t, 1.4, 2.1));
      const o = ease(seg(t, 6.8, 7.5)), obs = d.draws[0];
      if (o > 0) {
        line(ctx, obs.x, obs.y - DOT - 14, obs.x, lerp(obs.y - DOT - 14, d.top + 70, o), S.INK, 4.5);
        text(ctx, "observed", obs.x, d.top + 34, 44, { align: "center", alpha: o });
      }
      text(ctx, "exact test", S.W - 130, 96, 48, { align: "right", color: S.TONES.accent.text, alpha: seg(t, 7.6, 8.3) });
    },
  });
})(window.Studio);
