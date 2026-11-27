(() => {
  'use strict';

  function prepare(study) {
    const listed = study.populations;
    // The native adapter and the bundled study put a derived pooled population first.
    // CSV population labels are user-defined, including any literal "All cells" label.
    const hasPool = study.source === 'h5ad' || study.study?.source === 'Synthetic study';
    const populations = hasPool && listed.length > 1 ? listed.slice(1) : [...listed];
    const allowed = new Set(populations);
    const visits = [...study.visits];
    const arms = [...new Set(study.participants.map((person) => person.arm))];
    const rows = study.observations.filter((row) => allowed.has(row.population) && row.cells > 0);

    // Index just the active arm and feature. Changing visits shares the same scale and
    // does no new aggregation, even with many participant-level outcomes in a CSV.
    let active = null;
    function summarize(featureId, arm) {
      if (active && active.featureId === featureId && active.arm === arm) return active;
      const byVisit = new Map();
      for (const row of rows) {
        if (arm != null && arm !== row.arm) continue;
        let groups = byVisit.get(row.visit);
        if (!groups) byVisit.set(row.visit, (groups = new Map()));
        let group = groups.get(row.population);
        if (!group) {
          group = { count: 0, participants: 0, n: 0, value: null };
          groups.set(row.population, group);
        }
        group.count += row.cells;
        // Importers validate one observation per participant, population and visit.
        group.participants++;
        const score = featureId == null ? null : row.scores[featureId];
        if (!Number.isFinite(score)) continue;
        group.n++;
        if (group.n === 1) group.value = score;
        else {
          // Divide first for opposite-sign extremes to avoid overflow in subtraction.
          const delta = score - group.value;
          group.value = Number.isFinite(delta)
            ? group.value + delta / group.n
            : group.value * ((group.n - 1) / group.n) + score / group.n;
        }
      }

      let min = Infinity;
      let max = -Infinity;
      for (const groups of byVisit.values()) {
        for (const { value } of groups.values()) {
          if (value == null) continue;
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
      active = { featureId, arm, byVisit, range: min === Infinity ? null : [min, max] };
      return active;
    }

    return {
      populations,
      visits,
      arms,
      groups({ featureId = null, visit, arm = null }) {
        const atVisit = summarize(featureId, arm).byVisit.get(visit);
        return populations.map((name) => ({
          name,
          isPooled: hasPool && name === listed[0],
          ...(atVisit?.get(name) || { count: 0, participants: 0, n: 0, value: null }),
        }));
      },
      domain({ featureId = null, arm = null }) {
        const { range } = summarize(featureId, arm);
        return range && [...range];
      },
    };
  }

  window.SctrialAtlasData = { prepare };
})();
