(() => {
  'use strict';

  function emptyGroup(name, isPooled) {
    return {
      name,
      isPooled,
      count: 0,
      participants: 0,
      n: 0,
      value: null,
      values: [],
      min: null,
      q1: null,
      median: null,
      q3: null,
      max: null,
      cellBins: null,
    };
  }

  // Linear interpolation also works for opposite-sign endpoints near the limits of Number.
  function between(a, b, t) {
    const difference = b - a;
    return Number.isFinite(difference) ? a + difference * t : a * (1 - t) + b * t;
  }

  function quantile(sorted, p) {
    const position = (sorted.length - 1) * p;
    const index = Math.floor(position);
    return between(sorted[index], sorted[Math.ceil(position)], position - index);
  }

  function prepare(study) {
    const listed = study.populations;
    // CSV population labels are supplied by the user. Only the native adapter
    // and bundled study explicitly designate their first population as pooled.
    const hasPool = study.source === 'h5ad' || study.study?.source === 'Synthetic study';
    const pooledName = hasPool ? listed[0] : null;
    const populations = hasPool && listed.length > 1 ? listed.slice(1) : [...listed];
    const allowed = new Set(populations);
    const visits = [...study.visits];
    const arms = [...new Set(study.participants.map((person) => person.arm))];
    const rows = study.observations.filter(
      (row) => row.cells > 0 && (allowed.has(row.population) || row.population === pooledName),
    );
    const cellDistributions = study.cellDistributions;
    function cellDomain({ featureId }) {
      const limits = cellDistributions?.features?.[featureId];
      return Array.isArray(limits) && limits.length === 2 && limits.every(Number.isFinite)
        ? [...limits]
        : null;
    }

    // Native files and the bundled study supply common-bin histograms of measured
    // cells by arm, visit and disjoint type. A participant-summary CSV has none.
    function collectCellBins(byVisit, featureId, arm) {
      const domain = cellDomain({ featureId });
      if (!domain) return;
      function add(group, counts, zero) {
        if (!group) return;
        if (!group.cellBins) {
          group.cellBins = { counts: Array(counts.length).fill(0), domain, zero: 0, measured: 0 };
        }
        const combined = group.cellBins;
        for (let i = 0; i < counts.length; i++) {
          combined.counts[i] += counts[i];
          combined.measured += counts[i];
        }
        combined.zero += zero;
      }
      for (const entry of cellDistributions.groups) {
        if (arm != null && entry.arm !== arm) continue;
        const counts = entry.bins?.[featureId];
        if (!allowed.has(entry.population) || !Array.isArray(counts) || !counts.length) continue;
        const groups = byVisit.get(entry.visit)?.groups;
        if (!groups) continue;
        const zero = entry.zeros?.[featureId] ?? 0;
        add(groups.get(entry.population), counts, zero);
        if (hasPool && entry.population !== pooledName) add(groups.get(pooledName), counts, zero);
      }
    }

    // Retain only the current feature's distributions. Moving between visits can
    // reuse the sorted values without accumulating a copy per feature.
    let current = null;
    function summarize(featureId, arm) {
      if (current && current.featureId === featureId && current.arm === arm) return current;
      const byVisit = new Map();
      for (const row of rows) {
        if (arm != null && arm !== row.arm) continue;
        let visit = byVisit.get(row.visit);
        if (!visit) {
          visit = { groups: new Map(), totalCells: 0, people: new Set() };
          byVisit.set(row.visit, visit);
        }
        if (allowed.has(row.population)) {
          visit.totalCells += row.cells;
          visit.people.add(row.id);
        }
        let group = visit.groups.get(row.population);
        if (!group) {
          group = emptyGroup(row.population, row.population === pooledName);
          visit.groups.set(row.population, group);
        }
        group.count += row.cells;
        // The adapters supply a single row per participant, visit and population.
        group.participants++;
        const score = featureId == null ? null : row.scores[featureId];
        if (!Number.isFinite(score)) continue;
        group.values.push(score);
        group.n++;
        group.value = group.n === 1 ? score : between(group.value, score, 1 / group.n);
      }

      for (const visit of byVisit.values()) {
        for (const group of visit.groups.values()) {
          if (!group.n) continue;
          const values = group.values.sort((a, b) => a - b);
          group.min = values[0];
          group.max = values[values.length - 1];
          group.q1 = quantile(values, 0.25);
          group.median = quantile(values, 0.5);
          group.q3 = quantile(values, 0.75);
        }
      }
      collectCellBins(byVisit, featureId, arm);
      current = { featureId, arm, byVisit };
      return current;
    }

    // Distributions share the feature's full participant-value scale across arms
    // and visits, including the tails of each cell type. Missing values stay missing.
    let extent = null;
    function domain({ featureId = null, includePooled = false }) {
      if (featureId == null) return null;
      if (extent?.featureId !== featureId) {
        let min = Infinity;
        let max = -Infinity;
        let pooledMin = Infinity;
        let pooledMax = -Infinity;
        for (const row of rows) {
          const score = row.scores[featureId];
          if (!Number.isFinite(score)) continue;
          if (row.population === pooledName) {
            pooledMin = Math.min(pooledMin, score);
            pooledMax = Math.max(pooledMax, score);
          }
          if (allowed.has(row.population)) {
            min = Math.min(min, score);
            max = Math.max(max, score);
          }
        }
        extent = {
          featureId,
          range: min === Infinity ? null : [min, max],
          withPool:
            min === Infinity && pooledMin === Infinity
              ? null
              : [Math.min(min, pooledMin), Math.max(max, pooledMax)],
        };
      }
      const range = includePooled ? extent.withPool : extent.range;
      return range && [...range];
    }

    function groupAt(atVisit, name) {
      const group = atVisit?.groups.get(name) ?? emptyGroup(name, name === pooledName);
      const totalCells = atVisit?.totalCells ?? 0;
      // A derived pool is available for molecular measurements but never counted
      // alongside its constituent types in the observed composition.
      const fraction = allowed.has(name) ? (totalCells ? group.count / totalCells : 0) : null;
      return { ...group, totalCells, fraction };
    }

    return {
      populations,
      visits,
      arms,
      domain,
      cellDomain,
      groups({ featureId = null, visit, arm = null }) {
        const atVisit = summarize(featureId, arm).byVisit.get(visit);
        return populations.map((name) => groupAt(atVisit, name));
      },
      pooled({ featureId = null, visit, arm = null }) {
        if (pooledName == null) return null;
        return groupAt(summarize(featureId, arm).byVisit.get(visit), pooledName);
      },
      cohort({ visit, arm = null }) {
        const atVisit = summarize(current?.featureId ?? null, arm).byVisit.get(visit);
        return { count: atVisit?.totalCells ?? 0, participants: atVisit?.people.size ?? 0 };
      },
    };
  }

  window.SctrialAtlasData = { prepare };
})();
