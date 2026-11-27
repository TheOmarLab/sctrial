(() => {
  'use strict';
  const icons = {
    dashboard:
      '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    people:
      '<circle cx="9" cy="8" r="3"/><path d="M3 20v-2a6 6 0 0 1 12 0v2m0-15a3 3 0 0 1 0 6m2 3a5 5 0 0 1 4 5"/>',
    chart:
      '<path d="M3 4v16h18M6 14h11m-6-5h9"/><circle cx="10" cy="14" r="2"/><circle cx="16" cy="9" r="2"/>',
    sliders:
      '<path d="M3 6h7m4 0h7M3 12h12m4 0h2M3 18h3m4 0h11"/><circle cx="12" cy="6" r="2"/><circle cx="17" cy="12" r="2"/><circle cx="8" cy="18" r="2"/>',
    folder: '<path d="M3 6h7l2 2h8a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1Z"/>',
    layers: '<path d="m3 8 9-5 9 5-9 5Zm0 4 9 5 9-5M3 16l9 5 9-5"/>',
    chevrons: '<path d="m8 9 4-4 4 4m-8 6 4 4 4-4"/>',
    search: '<circle cx="10" cy="10" r="6.5"/><path d="m15 15 5.5 5.5"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
    download: '<path d="M12 3v11m-4-4 4 4 4-4M4 16v4h16v-4"/>',
    menu: '<path d="M3 5h18M3 12h18M3 19h18"/>',
    close: '<path d="m6 6 12 12m0-12L6 18"/>',
    paired: '<circle cx="6" cy="15" r="3"/><circle cx="18" cy="8" r="3"/><path d="m8.5 13.4 7-4.2"/>',
    cells:
      '<circle cx="8" cy="8" r="4"/><circle cx="16.5" cy="14.5" r="4.5"/><circle cx="6" cy="18" r="2"/>',
    calendar:
      '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v5m10-5v5M3 11h18m-12 4h2"/>',
    arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    file: '<path d="M5 3h9l5 5v13H5Zm9 0v6h5M8 14h8m-8 3h6"/>',
    code: '<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-14-2 18"/>',
    check: '<path d="m4 12 5 5L20 6"/>',
    upload: '<path d="M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4"/>',
  };
  Object.assign(icons, {
    down: '<path d="m6 9 6 6 6-6"/>',
    studio: '<path d="M3 19V5m0 11 5-5 4 3 8-9"/><circle cx="8" cy="11" r="1.5"/><path d="M3 20h18"/>',
    stack: '<rect x="5" y="7" width="14" height="13" rx="2"/><path d="M8 4h8m-6 8h4"/>',
    pin: '<path d="m8 3 9 2-2 5 3 3-1 2-6-1-5 7 2-8-4-1 1-2 3-1Z"/>',
    open: '<path d="M3 6h7l2 2h8v3M3 6v13h15l3-7H8l-3 7"/>',
    person: '<circle cx="12" cy="7" r="3"/><path d="M5 21v-3a7 7 0 0 1 14 0v3"/>',
    trend: '<path d="m3 17 5-6 5 3 7-9"/>',
    points:
      '<circle cx="5" cy="14" r="1.5"/><circle cx="11" cy="9" r="1.5"/><circle cx="17" cy="15" r="1.5"/><circle cx="19" cy="5" r="1.5"/>',
    crosshair: '<circle cx="12" cy="12" r="6"/><path d="M12 2v5m0 10v5M2 12h5m10 0h5"/>',
    expand: '<path d="M4 9V4h5m6 0h5v5M4 15v5h5m6 0h5v-5"/>',
    shrink: '<path d="M9 4v5H4m16 0h-5V4M4 15h5v5m6 0v-5h5"/>',
    refresh: '<path d="M19 7a8 8 0 1 0 1 8m-6-8h5V2"/>',
    undo: '<path d="m8 4-5 5 5 5m-5-5h8a7 7 0 0 1 7 7v3"/>',
    redo: '<path d="m16 4 5 5-5 5m5-5h-8a7 7 0 0 0-7 7v3"/>',
  });
  const icon = (name) =>
    '<svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">' +
    (icons[name] || '') +
    '</svg>';
  document.querySelectorAll('[data-icon]').forEach((node) => {
    node.innerHTML = icon(node.dataset.icon);
  });
  const $ = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? '').replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  const option = (value, chosen, label = value) =>
    '<option value="' +
    esc(value) +
    '"' +
    (value === chosen ? ' selected' : '') +
    '>' +
    esc(label) +
    '</option>';
  const labels = {
    did: 'Difference-in-differences',
    paired: 'Paired change',
    cross: 'Cross-sectional',
    observe: 'Single-visit mean',
  };
  const molecularModes = ['genes', 'pathways', 'modules'];
  const figureModes = [
    'cells',
    ...molecularModes,
    'comparison',
    'measurements',
    'trajectory',
    'changes',
  ];
  const titles = {
    overview: 'Analysis',
    participants: 'Cohort explorer',
    results: 'Results',
    bench: 'Pinned contrasts',
  };
  const sample = JSON.parse($('dataset').textContent);
  let study = sample,
    arms = [],
    obsByPerson,
    byPopulation,
    personById,
    state = {},
    peopleContext = {},
    previousTrigger = null,
    toastTimer,
    privateCounter = 0;
  let past = [],
    future = [];
  let playbackTimer = 0;
  let playbackVisit = null;
  let lastRecords = null;
  let currentFit = null,
    fitTask = null,
    fitStatus = 'idle',
    fitMessage = '',
    fitCache = new Map(),
    rowCache = new Map();
  const synthetic = () => study === sample,
    native = () => study.source === 'h5ad',
    isLong = () => ['did', 'paired'].includes(state.design);
  const fitted = () => synthetic() || (native() && state.design !== 'observe');
  const name = () => (synthetic() ? sample.study.name : study.title);
  const source = () => (synthetic() ? 'Synthetic study' : native() ? 'Local H5AD' : 'Local CSV');
  const feature = () => study.features.find((f) => f.id === state.feature) || study.features[0];
  const unitOf = (f) => f?.unit || study.unit || (synthetic() ? 'Mean module score' : 'Recorded score');
  const unit = () => unitOf(feature());
  const fdrFamily = (f) =>
    synthetic()
      ? ['gene', 'pathway'].includes(f?.kind)
        ? 'Eight illustrative gene and pathway outcomes'
        : 'Four original module scores'
      : fitted()
        ? 'Tested features in this comparison'
        : null;
  function fitProvenance(f) {
    if (!synthetic()) return native() && fitted() ? { package: 'sctrial ' + study.version } : {};
    const original = study.study,
      supplement = original.molecular;
    const molecularOutcome = ['gene', 'pathway'].includes(f?.kind);
    if (
      supplement &&
      (molecularOutcome || supplement.additionalPopulations?.includes(state.population))
    ) {
      return {
        package: supplement.package,
        seed: molecularOutcome ? supplement.seed : original.seed,
        referenceChecksum: supplement.baseSha256,
      };
    }
    return { package: original.package, seed: original.seed, commit: original.commit };
  }
  const scopeRows = (rows) =>
    rows.map((r) => {
      const f = study.features.find((f) => f.id === (r.id || r.feature));
      return {
        ...r,
        unit: unitOf(f),
        fdrFamily: fdrFamily(f),
        fitPackage: fitProvenance(f).package || null,
      };
    });
  const nfmt = (n) => (Number.isFinite(n) ? n.toLocaleString('en-US') : '—');
  const mean = (xs) => (xs.length ? xs.reduce((x, y) => x + y, 0) / xs.length : null);
  const second = () => arms.find((a) => a !== state.arm);
  const armsIn = () =>
    ['paired', 'observe'].includes(state.design) ? [state.arm] : [state.arm, second()].filter(Boolean);
  const datesIn = () => (isLong() ? [state.baseline, state.visit] : [state.visit]);
  const designs = () => [
    ...(arms.length > 1 && study.visits.length > 1 ? ['did'] : []),
    ...(study.visits.length > 1 ? ['paired'] : []),
    ...(arms.length > 1 ? ['cross'] : []),
    ...(arms.length === 1 && study.visits.length === 1 ? ['observe'] : []),
  ];
  const contrast = () => armsIn().join(' − ') + ' · ' + datesIn().join(' → ');
  const key = () =>
    JSON.stringify([state.design, state.arm, state.baseline, state.visit, state.population]);
  const color = (a) => (arms.indexOf(a) === 0 ? '#2969d5' : '#b86120');
  const rowAt = (id, v) => obsByPerson.get(id)?.get(v)?.get(state.population);
  const rowObserved = (r) => r && r.cells > 0;
  const method = () =>
    synthetic()
      ? study.study.methods[state.design]
      : fitted()
        ? currentFit?.method || 'Participant-level model'
        : state.design === 'observe'
          ? 'Mean across observed participants'
          : 'Unadjusted descriptive participant differences';
  const fitWarnings = () =>
    synthetic()
      ? study.resultWarnings?.[[state.design, state.arm, state.visit, state.population].join('|')] || []
      : currentFit?.warnings || [];
  const ready = () => !native() || !fitted() || fitStatus === 'ready';
  function clearFit() {
    if (fitTask) fitTask.abort();
    fitTask = null;
    currentFit = null;
    fitStatus = 'idle';
    fitMessage = '';
  }
  function init(next) {
    if ($('cellStage')) window.SctrialCells?.dispose($('cellStage'));
    if ($('molecularStage')) window.SctrialMolecular.dispose($('molecularStage'));
    playbackVisit = null;
    lastRecords = null;
    clearFit();
    past = [];
    future = [];
    fitCache.clear();
    rowCache.clear();
    study = next;
    arms = [...new Set(study.participants.map((p) => p.arm))];
    personById = new Map(study.participants.map((p) => [p.id, p]));
    obsByPerson = new Map();
    byPopulation = new Map();
    for (const row of study.observations) {
      if (!byPopulation.has(row.population)) byPopulation.set(row.population, []);
      byPopulation.get(row.population).push(row);
      if (!obsByPerson.has(row.id)) obsByPerson.set(row.id, new Map());
      const at = obsByPerson.get(row.id);
      if (!at.has(row.visit)) at.set(row.visit, new Map());
      at.get(row.visit).set(row.population, row);
    }
    const baseline =
      study.visits.find((v) =>
        /^(baseline|pre|pretreatment|pre-treatment|day 0|week 0)$/i.test(v.trim()),
      ) || study.visits[0];
    state = {
      view: 'overview',
      design: designs()[0],
      arm: arms[0],
      baseline,
      browseVisit: baseline,
      visit: study.visits.find((v) => v !== baseline) || baseline,
      population: study.populations[0],
      feature: study.features[0].id,
      mode: 'cells',
      selectedId: null,
    };
    peopleContext = {};
    sheets.reset('Primary analysis');
    bench.setStudy(synthetic() ? 'immune-study-62834-v1' : 'local-' + ++privateCounter, {
      persist: synthetic(),
      sharedFeatureUnit: new Set(study.features.map(unitOf)).size === 1 && synthetic(),
    });
    render();
  }
  function currentRows() {
    if (native() && fitted()) return currentFit?.rows || [];
    const k = key();
    if (rowCache.has(k)) return rowCache.get(k);
    let rows;
    if (synthetic()) {
      const raw =
        study.results[[state.design, state.arm, state.visit, state.population].join('|')] || [];
      rows = raw.map((r) => ({
        ...r,
        id: r.feature,
        label: study.features.find((f) => f.id === r.feature).label,
      }));
    } else {
      const groups = armsIn(),
        visits = datesIn();
      rows = study.features.map((f) => {
        const values = groups.map((arm) =>
          study.participants
            .filter((p) => p.arm === arm)
            .flatMap((p) => {
              const vs = visits.map((v) => rowAt(p.id, v)?.scores[f.id]);
              return vs.every(Number.isFinite) ? [vs.length === 2 ? vs[1] - vs[0] : vs[0]] : [];
            }),
        );
        const av = values.map(mean);
        return {
          id: f.id,
          label: f.label,
          estimate: av.every(Number.isFinite) ? (av.length === 1 ? av[0] : av[0] - av[1]) : null,
          lo: null,
          hi: null,
          p: null,
          q: null,
          n: values.reduce((a, v) => a + v.length, 0),
          nFirst: values[0].length,
          nOther: values[1]?.length ?? null,
        };
      });
    }
    if (rowCache.size >= 32) rowCache.delete(rowCache.keys().next().value);
    rows = scopeRows(rows);
    rowCache.set(k, rows);
    return rows;
  }
  async function requestFit() {
    if (!native() || !fitted()) return;
    const k = key(),
      file = study;
    clearFit();
    if (fitCache.has(k)) {
      currentFit = fitCache.get(k);
      fitStatus = 'ready';
      refreshFit();
      return;
    }
    const own = new AbortController();
    fitTask = own;
    fitStatus = 'loading';
    refreshFit();
    try {
      const fit = await localStudyClient.fit(
        file,
        {
          design: state.design,
          arm: state.arm,
          reference: second(),
          visit: state.visit,
          baseline: state.baseline,
          population: state.population,
        },
        own.signal,
      );
      if (own.signal.aborted || file !== study || key() !== k) return;
      currentFit = { ...fit, rows: scopeRows(fit.rows) };
      fitStatus = 'ready';
      if (fitCache.size >= 24) fitCache.delete(fitCache.keys().next().value);
      fitCache.set(k, currentFit);
    } catch (error) {
      if (own.signal.aborted || study !== file || key() !== k) return;
      fitStatus = 'error';
      fitMessage = error.message;
    } finally {
      if (fitTask === own) {
        fitTask = null;
        refreshFit();
      }
    }
  }
  function fitWhenVisible() {
    if (
      native() &&
      fitted() &&
      fitStatus === 'idle' &&
      (state.view === 'results' ||
        (state.view === 'overview' && !['cells', ...molecularModes].includes(state.mode)))
    ) {
      void requestFit();
    }
  }
  function refreshFit() {
    updateTop();
    if ($('resFile') && $('dialog').open) $('resFile').disabled = !ready();
    if (state.view === 'overview' && state.mode === 'comparison') paintEvidence();
    else if (state.view === 'results') renderResultsView();
  }
  function setParam(field, value) {
    if (state[field] === value) return;
    remember();
    const oldReference = state.baseline;
    state[field] = value;
    state.selectedId = null;
    if (isLong() && state.baseline === state.visit)
      state.visit =
        oldReference !== state.baseline ? oldReference : study.visits.find((v) => v !== state.baseline);
    clearFit();
    render();
  }
  const bench = createContrastBench({
    getSnapshot: () =>
      !ready()
        ? null
        : {
            title: contrast(),
            spec: {
              design: state.design,
              arm: state.arm,
              reference: armsIn()[1] || null,
              baseline: isLong() ? state.baseline : null,
              visit: state.visit,
              population: state.population,
              feature: state.feature,
              browseVisit: state.browseVisit,
              mode: state.mode,
            },
            rows: currentRows(),
            unit: unit(),
            method: method(),
          },
    onRestore: (spec) => {
      const { design, arm, baseline, visit, population, feature: id, mode } = spec;
      if (
        !designs().includes(design) ||
        !arms.includes(arm) ||
        !study.visits.includes(visit) ||
        !study.populations.includes(population)
      )
        return;
      Object.assign(state, {
        design,
        arm,
        baseline: baseline || state.baseline,
        browseVisit: study.visits.includes(spec.browseVisit) ? spec.browseVisit : visit,
        visit,
        population,
        feature: study.features.some((f) => f.id === id) ? id : study.features[0].id,
        mode: mode || 'trajectory',
        selectedId: null,
      });
      clearFit();
      navigate('overview');
    },
    onNotice: (message) => toast(message),
    onChange: (n) => setBadge(n),
    onExplore: () => changeViewMode('comparison'),
  });
  const sheets = window.createAnalysisSheets({
    getStudy: () => ({
      name: name(),
      source: study.source || 'example',
      arms,
      visits: study.visits,
      populations: study.populations,
      features: study.features,
      mapping: study.mapping || null,
      designs: designs(),
      fixedBaseline: synthetic() ? sample.visits[0] : null,
    }),
    getSpec: () => ({
      ...snapshotSpec(),
      mode: state.mode,
      visitOrder: [...study.visits],
    }),
    onOpen: ({ visitOrder, ...spec }) => {
      past = [];
      future = [];
      if (visitOrder) {
        study.visits = [...visitOrder];
      }
      Object.assign(state, spec, { selectedId: null });
      clearFit();
      navigate('overview');
    },
    onRename: () => updateTop(),
  });

  function setBadge(count) {
    $('pinCount').textContent = String(count);
    $('pinCount').hidden = !count;
  }
  function updateTop() {
    if (!figureModes.includes(state.mode)) state.mode = 'cells';
    document.body.dataset.mode = state.view === 'overview' ? state.mode : state.view;
    window.SctrialNavigation?.sync(document.body.dataset.mode);
    const modeSelect = $('chartMode');
    if (modeSelect) {
      let secondary = modeSelect.querySelector('[data-secondary]');
      if (state.view !== 'overview' && !secondary) {
        secondary = new Option('', '');
        secondary.dataset.secondary = '';
        secondary.hidden = true;
        secondary.disabled = true;
        modeSelect.add(secondary);
      }
      if (secondary) secondary.textContent = titles[state.view];
      modeSelect.value = state.view === 'overview' ? state.mode : '';
      const changes = modeSelect.querySelector('[value="changes"]');
      if (changes) changes.textContent = isLong() ? 'Changes' : 'Distribution';
    }
    document.querySelectorAll('button[data-mode]').forEach((b) => {
      const key = b.getAttribute('role') === 'menuitemradio' ? 'aria-checked' : 'aria-pressed';
      b.setAttribute(key, String(state.view === 'overview' && b.dataset.mode === state.mode));
    });
    $('studyName').textContent = name();
    $('sourceBadge').textContent = source();
    $('studyCount').textContent = nfmt(study.participants.length) + ' participants';
    $('pageTitle').textContent = state.view === 'overview' ? sheets.title() : titles[state.view];
    $('sectionCrumb').textContent = state.view === 'overview' ? 'Analysis workbook' : titles[state.view];
    sheets.setVisible(state.view === 'overview');
    window.SctrialAppMenu?.setState({
      view: state.view,
      mode: state.mode,
      longitudinal: isLong(),
      canPin: ready(),
    });
    $('pinButton').hidden = state.view === 'bench';
    $('pinButton').disabled = !ready();
    $('pinButton').title = ready() ? 'Save this comparison' : 'Waiting for a fitted result';
    document.querySelectorAll('[data-view]').forEach((n) => {
      if (n.dataset.view === state.view) n.setAttribute('aria-current', 'page');
      else n.removeAttribute('aria-current');
    });
    $('undo').disabled = !past.length;
    $('redo').disabled = !future.length;
    setBadge(bench.count());
  }
  const button = (action, glyph, text, css = 'text-btn') =>
    '<button type="button" class="' +
    css +
    '" data-action="' +
    action +
    '">' +
    icon(glyph) +
    esc(text) +
    '</button>';
  function comparisonRecords() {
    const specKey = JSON.stringify([
      state.design,
      state.arm,
      state.baseline,
      state.visit,
      state.population,
      state.feature,
    ]);
    if (lastRecords?.study === study && lastRecords.key === specKey) return lastRecords.rows;
    const groups = armsIn();
    const visitNames = datesIn();
    const rows = study.participants
      .filter((p) => groups.includes(p.arm))
      .map((p) => {
        const values = visitNames.map((v) => {
          const r = rowAt(p.id, v);
          const score = r?.scores[state.feature];
          return rowObserved(r) && Number.isFinite(score) ? score : null;
        });
        const missing = visitNames.filter((v, i) => values[i] === null);
        return {
          id: p.id,
          arm: p.arm,
          baselineValue: isLong() ? values[0] : null,
          followupValue: values[values.length - 1],
          contributes: missing.length === 0,
          reason: missing.length ? 'Missing score at ' + missing.join(' and ') : '',
        };
      });
    lastRecords = { study, key: specKey, rows };
    return rows;
  }

  function paintEvidence(records = comparisonRecords()) {
    const host = $('effectPlot');
    if (!host) return;
    const r = currentRows().find((row) => row.id === state.feature);
    const armData = armsIn().map((name) => {
      const people = records.filter((p) => p.arm === name && p.contributes);
      const values = people.map((p) => (isLong() ? p.followupValue - p.baselineValue : p.followupValue));
      const ids = people.map((p) => p.id);
      return {
        name,
        colorIndex: arms.indexOf(name),
        n: values.length,
        mean: mean(values),
        values,
        ids,
      };
    });
    const formula = armsIn().join(' − ');
    window.SctrialEvidence.render(host, {
      arms: armData,
      records,
      onSelect: selectPerson,
      onInspectTable: () => changeViewMode('measurements'),
      onOpenMatrix: () => changeViewMode('measurements'),
      onConfigure: configure,
      estimate: r?.estimate ?? null,
      interval: [r?.lo, r?.hi].every(Number.isFinite) ? [r.lo, r.hi] : null,
      contrast: formula,
      unit: unit(),
      feature: feature().label,
      longitudinal: isLong(),
      baseline: state.baseline,
      visit: state.visit,
      model: method(),
      descriptive: !fitted(),
      fdrFamily: fdrFamily(feature()),
      status: fitStatus === 'loading' ? 'Fitting comparison…' : fitMessage || r?.reason || '',
    });
    const retryArea = $('fitRetry');
    retryArea.replaceChildren();
    if (fitStatus === 'error') retryArea.innerHTML = button('retry', 'refresh', 'Retry fit', 'btn');
    if (fitWarnings().length) {
      const notes = document.createElement('details');
      notes.className = 'fit-warnings';
      const title = document.createElement('summary');
      title.textContent = 'Model notes (' + fitWarnings().length + ')';
      notes.append(title);
      for (const message of fitWarnings()) {
        const p = document.createElement('p');
        p.textContent = message;
        notes.append(p);
      }
      retryArea.append(notes);
    }
  }

  function paintMatrix(records) {
    window.SctrialMatrix.render($('participantMatrix'), {
      records,
      baseline: state.baseline,
      visit: state.visit,
      feature: feature().label,
      unit: unit(),
      armLabels: armsIn(),
      studyArms: arms,
      longitudinal: isLong(),
      selectedId: state.selectedId,
      onSelect: selectPerson,
    });
  }

  function paintCanvas() {
    const active = armsIn();
    const byId = new Map();
    const measured = new Set();
    const id = state.feature;
    for (const row of byPopulation.get(state.population) || []) {
      if (!active.includes(row.arm) || !rowObserved(row) || !Number.isFinite(row.scores[id])) continue;
      if (!byId.has(row.id)) byId.set(row.id, { id: row.id, arm: row.arm, points: [] });
      measured.add(row.visit);
      byId.get(row.id).points.push({ visit: row.visit, value: row.scores[id] });
    }
    renderContrastCanvas(
      $('plot'),
      {
        visits: study.visits,
        baseline: state.baseline,
        visit: state.visit,
        design: state.design,
        feature: feature(),
        unit: unit(),
        groups: active.map((a) => ({ name: a, color: color(a) })),
        series: [...byId.values()],
        mode: state.mode,
        selectedId: null,
        canSelectVisit: study.visits.length > 1,
        editableVisits: study.visits.filter((v) => !isLong() || v !== state.baseline),
      },
      { onSelectParticipant: selectPerson, onVisit: (v) => setParam('visit', v) },
    );
    return study.visits.filter((v) => measured.has(v));
  }

  const scoreFormat = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 6 });
  function measurement(value, signed = false) {
    if (!Number.isFinite(value)) return '—';
    return (signed && value > 0 ? '+' : '') + scoreFormat.format(value).replace('-', '−');
  }

  function participantDetail(person) {
    const rows = study.visits.map((v) => {
      const r = rowAt(person.id, v);
      const score = r?.scores[state.feature];
      const status = !obsByPerson.get(person.id)?.has(v)
        ? 'No specimen'
        : !rowObserved(r)
          ? 'Population absent'
          : !Number.isFinite(score)
            ? 'Missing score'
            : 'Measured';
      return (
        '<tr><th scope="row">' +
        esc(v) +
        '</th><td>' +
        measurement(score) +
        '</td><td>' +
        nfmt(r?.cells) +
        '</td><td>' +
        esc(status) +
        '</td></tr>'
      );
    });
    const record = comparisonRecords().find((p) => p.id === person.id);
    const summary = record?.contributes
      ? isLong()
        ? 'Paired change ' + measurement(record.followupValue - record.baselineValue, true)
        : 'Selected score ' + measurement(record.followupValue)
      : record?.reason || 'Outside this comparison';
    return (
      '<div class="ed-record"><div class="ed-person-context"><span>' +
      icon('people') +
      esc(person.arm) +
      '</span><span>' +
      icon('cells') +
      esc(state.population) +
      '</span></div><h3>' +
      esc(feature().label) +
      '</h3><p>' +
      esc(unit()) +
      '</p><div class="ed-record-scroll"><table><thead><tr><th>Visit</th><th>Score</th>' +
      '<th>Cells</th><th>Measurement</th></tr></thead><tbody>' +
      rows.join('') +
      '</tbody></table></div><div class="ed-record-footer"><strong>' +
      esc(summary) +
      '</strong>' +
      button('personRecord', 'arrow', 'Open cohort record') +
      '</div></div>'
    );
  }

  const wideRecord = window.matchMedia('(min-width: 930px)');
  let recordTrigger = null;

  function closeRecord(restore = false) {
    const drawer = $('recordDrawer');
    if (drawer) {
      drawer.remove();
      $('page').classList.remove('ed-with-record');
      if (restore) {
        const previous = recordTrigger?.element;
        const replacement =
          recordTrigger?.label &&
          Array.from($('page').querySelectorAll('[aria-label]')).find(
            (el) => el.getAttribute('aria-label') === recordTrigger.label,
          );
        const target = previous?.isConnected
          ? previous
          : replacement || $('page').querySelector('[data-param="feature"]');
        target?.focus({ preventScroll: true });
      }
      recordTrigger = null;
    }
    if ($('dialog').open && $('dialog').dataset.kind === 'participant') finish();
  }

  function selectPerson(id) {
    if (id == null || !personById.has(id)) return;
    state.selectedId = id;
    const title = 'Participant ' + id;
    const content = participantDetail(personById.get(id));
    if (!wideRecord.matches) {
      modal(title, content, 'participant');
      return;
    }
    let drawer = $('recordDrawer');
    const trigger = document.activeElement;
    if (!drawer?.contains(trigger)) {
      recordTrigger = { element: trigger, label: trigger?.getAttribute('aria-label') };
    }
    if (!drawer) {
      drawer = document.createElement('aside');
      drawer.id = 'recordDrawer';
      drawer.className = 'ed-inspector';
      drawer.setAttribute('role', 'dialog');
      drawer.setAttribute('aria-modal', 'false');
      drawer.setAttribute('aria-labelledby', 'recordHeading');
      drawer.tabIndex = -1;
      $('page').classList.add('ed-with-record');
      $('page').append(drawer);
    }
    drawer.innerHTML =
      '<div class="ed-inspector-head"><h2 id="recordHeading">' +
      esc(title) +
      '</h2><button type="button" class="icon-btn" data-action="closeRecord" aria-label="Close record">' +
      icon('close') +
      '</button></div>' +
      content;
    drawer.focus({ preventScroll: true });
  }

  wideRecord.addEventListener('change', () => {
    if (!$('recordDrawer') && !($('dialog').open && $('dialog').dataset.kind === 'participant')) return;
    const id = state.selectedId;
    closeRecord();
    if (id != null) selectPerson(id);
  });

  function setFeature(id) {
    if (!study.features.some((f) => f.id === id) || id === state.feature) return;
    remember();
    state.feature = id;
    state.selectedId = null;
    if (state.view !== 'overview') navigate('overview');
    else render();
  }

  function changeViewMode(mode) {
    if (!figureModes.includes(mode)) return;
    const viewChanged = state.view !== 'overview';
    if (!viewChanged && mode === state.mode) return;
    state.mode = mode;
    if (viewChanged) navigate('overview');
    else render();
  }

  function editLink(field, value, label, className, glyph) {
    return (
      '<button type="button" class="ed-edit ' +
      className +
      '" data-configure="' +
      field +
      '" aria-label="' +
      esc(label) +
      '" title="' +
      esc(label) +
      '">' +
      esc(value) +
      (glyph ? icon(glyph) : '') +
      '</button>'
    );
  }

  function inlineChoice(field, label, choices) {
    return (
      '<label class="ed-choice" title="' +
      esc(label) +
      '"><span aria-hidden="true">' +
      esc(state[field]) +
      '</span>' +
      icon('down') +
      '<select data-param="' +
      field +
      '" aria-label="' +
      esc(label) +
      '">' +
      choices.map((v) => option(v, state[field])).join('') +
      '</select></label>'
    );
  }

  function visitContext() {
    if (!isLong()) return inlineChoice('visit', 'Selected visit', study.visits);
    const reference = synthetic()
      ? '<span class="ed-reference">' + esc(state.baseline) + '</span>'
      : inlineChoice('baseline', 'Reference visit', study.visits);
    return (
      reference +
      '<span class="ed-direction" aria-hidden="true">→</span>' +
      inlineChoice(
        'visit',
        synthetic() ? 'Follow-up visit' : 'Comparison visit',
        study.visits.filter((v) => v !== state.baseline),
      )
    );
  }

  function figureHeading() {
    const groupName = armsIn().join(' − ');
    const outcome = feature().label;
    return (
      '<header class="ed-heading"><div class="ed-method-line">' +
      editLink('design', labels[state.design], 'Configure', 'ed-method', 'sliders') +
      '<span aria-hidden="true" class="ed-separator">/</span>' +
      editLink('arm', groupName, 'Configure group order', 'ed-contrast') +
      '</div><h1 class="ed-title" aria-label="' +
      esc(outcome) +
      '">' +
      '<label class="ed-outcome"><span class="ed-name" aria-hidden="true">' +
      esc(outcome) +
      '</span>' +
      icon('down') +
      '<select aria-label="Outcome" data-param="feature">' +
      study.features.map((f) => option(f.id, state.feature, f.label)).join('') +
      '</select></label></h1><div class="ed-context">' +
      visitContext() +
      '<span aria-hidden="true" class="ed-separator">·</span>' +
      inlineChoice('population', 'Cell population', study.populations) +
      '</div></header>'
    );
  }

  function cellArtboard(page) {
    let host = $('cellStage');
    if (!host) {
      if ($('plot')) disposeContrastCanvas($('plot'));
      host = document.createElement('div');
      host.id = 'cellStage';
      page.replaceChildren(host);
    }
    const pooled = native() || synthetic() ? study.populations[0] : null;
    const cellTypes = synthetic() ? study.cellTypes || [] : native() ? study.populations.slice(1) : [];
    window.SctrialCells.render(host, {
      studyName: name(),
      source: source(),
      sourceKind: synthetic() ? 'synthetic' : native() ? 'h5ad' : 'csv',
      populations: study.populations,
      allLabel: pooled,
      cellTypes,
      observations: study.observations,
      selected: state.population,
      onSelect: (population) => {
        if (study.populations.includes(population)) setParam('population', population);
      },
      onAnalyze: (population) => {
        if (!study.populations.includes(population)) return;
        if (population !== state.population) {
          remember();
          state.population = population;
          state.selectedId = null;
          clearFit();
        }
        changeViewMode('comparison');
      },
      onOpenStudy: openStudy,
    });
  }

  function stopPlayback() {
    clearInterval(playbackTimer);
    playbackTimer = 0;
    const button = $('trajectoryPlayback');
    if (button) {
      button.textContent = '▶';
      button.setAttribute('aria-label', 'Play visits');
      button.title = 'Play observed visits';
      button.setAttribute('aria-pressed', 'false');
    }
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopPlayback();
  });
  window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', stopPlayback);

  function visitPlayer() {
    return (
      '<div class="ed-playback"><button type="button" id="trajectoryPlayback" aria-label="Play visits" title="Play observed visits">▶</button>' +
      '<label for="trajectoryBrowse">Browse visits</label><input id="trajectoryBrowse" type="range" aria-label="Browse observed visits" min="0" step="1">' +
      '<output for="trajectoryBrowse" id="trajectoryFrame"></output></div>'
    );
  }

  function setupPlayback(visits) {
    const slider = $('trajectoryBrowse'),
      button = $('trajectoryPlayback'),
      output = $('trajectoryFrame');
    slider.max = Math.max(0, visits.length - 1);
    slider.disabled = button.disabled = visits.length < 2;
    const show = (index) => {
      const visit = visits[index] ?? null;
      playbackVisit = visit;
      slider.value = index;
      slider.setAttribute('aria-valuetext', visit || 'No measured visits');
      output.textContent = visit
        ? visit + ' · ' + (index + 1) + '/' + visits.length
        : 'No measured visits';
      window.setContrastVisitCursor($('plot'), visit);
    };
    show(Math.max(0, visits.indexOf(playbackVisit)));
    slider.oninput = () => {
      stopPlayback();
      show(Number(slider.value));
    };
    button.onclick = () => {
      if (playbackTimer) {
        stopPlayback();
        return;
      }
      if (Number(slider.value) === visits.length - 1) show(0);
      button.textContent = 'Ⅱ';
      button.setAttribute('aria-label', 'Pause visits');
      button.title = 'Pause visit playback';
      button.setAttribute('aria-pressed', 'true');
      playbackTimer = setInterval(() => {
        if (!slider.isConnected) {
          stopPlayback();
          return;
        }
        const index = Number(slider.value) + 1;
        if (index < visits.length) show(index);
        if (index >= visits.length - 1) stopPlayback();
      }, 1050);
    };
  }

  function molecularArtboard(page) {
    let host = $('molecularStage');
    if (!host) {
      if ($('plot')) disposeContrastCanvas($('plot'));
      host = document.createElement('div');
      host.id = 'molecularStage';
      page.replaceChildren(host);
    }
    const referenceIsBrowsed = isLong() && state.browseVisit === state.baseline;
    const comparisonVisit = referenceIsBrowsed ? state.visit : state.browseVisit;
    const visits = isLong() ? [state.baseline, comparisonVisit] : [comparisonVisit];
    window.SctrialMolecular.render(host, {
      mode: state.mode,
      study,
      selectedFeature: state.feature,
      selectedPopulation: state.population,
      selectedVisit: state.browseVisit,
      selectedArm: state.arm,
      analysisContext: armsIn().join(' − ') + ' · ' + visits.join(' → '),
      analysisRequiresExistingVisit: referenceIsBrowsed,
      source: source(),
      onFeature: setFeature,
      onDisplayedFeature: (id) => {
        state.feature = id;
      },
      onLinkGeneSet: synthetic()
        ? null
        : (featureId) =>
            window.SctrialGeneSets.open({
              study,
              featureId,
              show: modal,
              close: finish,
              onChange: () => molecularArtboard(page),
            }),
      onVisit: (visit) => {
        if (!study.visits.includes(visit) || visit === state.browseVisit) return;
        remember();
        state.browseVisit = visit;
        updateTop();
        molecularArtboard(page);
      },
      onPopulation: (population) => {
        if (study.populations.includes(population)) setParam('population', population);
      },
      onAnalyze: (id) => {
        if (!study.features.some((f) => f.id === id)) return;
        if (id !== state.feature || comparisonVisit !== state.visit) {
          remember();
          if (comparisonVisit !== state.visit) clearFit();
          state.feature = id;
          state.visit = comparisonVisit;
          state.selectedId = null;
        }
        changeViewMode('comparison');
      },
      onOpenStudy: openStudy,
    });
  }

  function overview() {
    const focusKey = document.activeElement?.getAttribute('data-param');
    const page = $('page');
    if (state.mode === 'cells') {
      cellArtboard(page);
      return;
    }
    if (molecularModes.includes(state.mode)) {
      molecularArtboard(page);
      return;
    }
    const active = page.firstElementChild;
    if (active?.classList.contains('ed-document') && active.dataset.display === state.mode) {
      active.querySelector('.ed-heading').outerHTML = figureHeading();
    } else {
      if ($('plot')) disposeContrastCanvas($('plot'));
      const surface =
        state.mode === 'comparison'
          ? '<div id="effectPlot"></div><div id="fitRetry"></div>'
          : state.mode === 'measurements'
            ? '<div id="participantMatrix"></div>'
            : (state.mode === 'trajectory' ? visitPlayer() : '') +
              '<div id="plot" aria-label="Participant plot"></div>';
      page.innerHTML =
        '<article class="ed-document" data-display="' +
        esc(state.mode) +
        '">' +
        figureHeading() +
        surface +
        '</article>';
    }
    if (state.mode === 'comparison') paintEvidence();
    else if (state.mode === 'measurements') paintMatrix(comparisonRecords());
    else {
      const observed = paintCanvas();
      if (state.mode === 'trajectory') setupPlayback(observed);
    }
    if (focusKey)
      document.querySelector('#page [data-param="' + focusKey + '"]')?.focus({ preventScroll: true });
  }

  function configure(focusField = 'design') {
    const draft = snapshotSpec();
    function selectField(field, label, options, disabled = false) {
      return (
        '<label class="ed-setting"><span>' +
        esc(label) +
        '</span><select data-setting="' +
        field +
        '" aria-label="' +
        esc(label) +
        '"' +
        (disabled ? ' disabled' : '') +
        '>' +
        options.map(([value, title]) => option(value, draft[field], title)).join('') +
        '</select></label>'
      );
    }
    const pairs = (items) => items.map((v) => [v, v]);
    function draw() {
      const longitudinal = ['did', 'paired'].includes(draft.design);
      const twoGroups = ['did', 'cross'].includes(draft.design);
      const visits = longitudinal ? study.visits.filter((v) => v !== draft.baseline) : study.visits;
      if (!visits.includes(draft.visit)) draft.visit = visits[0];
      modal(
        'Configure comparison',
        '<form id="comparisonSettings" class="ed-settings"><div class="ed-settings-grid">' +
          selectField(
            'design',
            'Study design',
            designs().map((d) => [d, labels[d]]),
          ) +
          selectField(
            'arm',
            twoGroups ? 'Group order' : 'Group',
            arms.map((a) => [a, twoGroups ? a + ' − ' + arms.find((other) => other !== a) : a]),
          ) +
          (longitudinal
            ? selectField(
                'baseline',
                synthetic() ? 'Baseline visit' : 'Reference visit',
                pairs(study.visits),
                synthetic(),
              )
            : '') +
          selectField(
            'visit',
            longitudinal ? (synthetic() ? 'Follow-up visit' : 'Comparison visit') : 'Selected visit',
            pairs(visits),
          ) +
          selectField('population', 'Cell population', pairs(study.populations)) +
          selectField(
            'feature',
            'Outcome',
            study.features.map((f) => [f.id, f.label]),
          ) +
          '</div>' +
          (!synthetic() && study.visits.length > 1
            ? '<button type="button" class="text-btn ed-order" id="visitOrder">' +
              icon('calendar') +
              'Set visit order…</button>'
            : '') +
          '<div class="ed-setting-actions"><button type="button" class="btn" id="cancelSettings">Cancel</button>' +
          '<button type="submit" class="btn accent">Apply changes</button></div></form>',
      );
      const form = $('comparisonSettings');
      form.onchange = (e) => {
        const field = e.target.dataset.setting;
        if (!field) return;
        draft[field] = e.target.value;
        if (['baseline', 'design'].includes(field)) {
          draw();
          formFocus(field);
        }
      };
      $('cancelSettings').onclick = finish;
      if ($('visitOrder'))
        $('visitOrder').onclick = async () => {
          const ordered = await localStudyClient.reviewVisitOrder({
            ...study,
            visits: draft.visitOrder || study.visits,
          });
          if (ordered) {
            draft.visitOrder = ordered;
            $('visitOrder').title = ordered.join(' → ');
          }
        };
      form.onsubmit = (e) => {
        e.preventDefault();
        const { visitOrder, ...next } = draft;
        if (
          JSON.stringify(next) !== JSON.stringify(snapshotSpec()) ||
          (visitOrder && JSON.stringify(visitOrder) !== JSON.stringify(study.visits))
        ) {
          remember();
          Object.assign(state, next, { selectedId: null });
          if (visitOrder) {
            study.visits = visitOrder;
          }
          clearFit();
          render();
        }
        finish();
        document.querySelector('#page [data-configure="design"]')?.focus();
      };
    }
    function formFocus(field) {
      document
        .querySelector('#comparisonSettings [data-setting="' + field + '"]:not([disabled])')
        ?.focus();
    }
    draw();
    formFocus(focusField);
  }

  function renderResultsView() {
    const page = $('page');
    if (!ready()) {
      page.innerHTML =
        '<div class="card pending"><h2>Model results</h2><p role="status">' +
        esc(
          fitStatus === 'loading'
            ? 'Fitting selected comparison…'
            : fitMessage || 'Results unavailable.',
        ) +
        '</p>' +
        (!fitTask ? button('retry', 'refresh', 'Run fit', 'btn') : '') +
        '</div>';
      return;
    }
    renderResults(
      page,
      {
        rows: currentRows(),
        selected: state.feature,
        unit: unit(),
        reference: fitted(),
        fdrDescription: synthetic()
          ? 'FDR is calculated separately for the four original module scores and the eight additional gene and pathway outcomes.'
          : null,
        isMean: state.design === 'observe',
        contrast: contrast(),
      },
      {
        onSelect: (id) => {
          state.feature = id;
          renderResultsView();
        },
        onShowOverview: (id) => {
          state.feature = id;
          state.mode = 'trajectory';
          state.selectedId = null;
          navigate('overview');
        },
        onExport: () => exportFile('results'),
      },
    );
    if (fitWarnings().length) {
      const d = document.createElement('details');
      d.className = 'fit-warnings';
      const summary = document.createElement('summary');
      summary.textContent = 'Model notes (' + fitWarnings().length + ')';
      d.append(summary);
      for (const w of fitWarnings()) {
        const p = document.createElement('p');
        p.textContent = w;
        d.append(p);
      }
      page.append(d);
    }
  }
  function render() {
    stopPlayback();
    closeRecord();
    if ($('cellStage') && (state.view !== 'overview' || state.mode !== 'cells')) {
      window.SctrialCells.dispose($('cellStage'));
    }
    if ($('molecularStage') && (state.view !== 'overview' || !molecularModes.includes(state.mode))) {
      window.SctrialMolecular.dispose($('molecularStage'));
    }
    updateTop();
    if (state.view !== 'overview' && $('plot')) disposeContrastCanvas($('plot'));
    if (state.view === 'overview') overview();
    else if (state.view === 'participants') {
      renderPeople(
        $('page'),
        {
          ...peopleContext,
          participants: study.participants,
          arms,
          visits: study.visits,
          observations: study.observations,
          population: state.population,
          selectedVisits: datesIn(),
          analysisArms: armsIn(),
          feature: feature(),
        },
        {},
      );
      peopleContext = {};
    } else if (state.view === 'results') renderResultsView();
    else bench.render($('page'));
    fitWhenVisible();
  }
  function navigate(view, context = {}) {
    peopleContext = context;
    state.view = view;
    render();
    window.scrollTo(0, 0);
  }
  function modal(title, html, kind = '') {
    const d = $('dialog');
    d.dataset.kind = kind;
    if (!d.open) previousTrigger = document.activeElement;
    $('dialogTitle').textContent = title;
    $('dialogBody').innerHTML = html;
    if (!d.open) d.showModal();
  }
  function finish() {
    if ($('dialog').open) $('dialog').close();
  }
  function toast(text) {
    $('toast').textContent = text;
    $('toast').hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      $('toast').hidden = true;
    }, 2600);
  }
  $('dialog').addEventListener('close', () => {
    if (previousTrigger?.isConnected) previousTrigger.focus();
    previousTrigger = null;
  });
  $('dismiss').onclick = finish;
  $('dialog').addEventListener('click', (e) => {
    if (e.target === $('dialog')) {
      const r = $('dialog').getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
        finish();
    }
  });
  function search() {
    modal(
      'Find in study',
      '<label class="field"><span>Participant ID or score</span><input class="field-input" id="query" autocomplete="off" type="search" placeholder="Search this study"></label><div class="modal-list" id="matches"></div>',
    );
    const data = [
      ...study.features.map((f) => ({ key: f.id, title: f.label, type: 'Outcome score' })),
      ...study.participants.map((p) => ({ key: p.id, title: p.id, type: p.arm })),
    ];
    function update() {
      const q = $('query').value.trim().toLowerCase(),
        hits = data.filter((i) => (i.title + ' ' + i.type).toLowerCase().includes(q)).slice(0, 30);
      $('matches').innerHTML =
        hits
          .map(
            (i, n) =>
              '<button data-match="' +
              n +
              '"><b>' +
              esc(i.title) +
              '</b><small>' +
              esc(i.type) +
              '</small></button>',
          )
          .join('') || '<p class="empty">No matches.</p>';
      document.querySelectorAll('[data-match]').forEach(
        (b) =>
          (b.onclick = () => {
            const i = hits[Number(b.dataset.match)];
            finish();
            if (data.indexOf(i) < study.features.length) {
              state.feature = i.key;
              state.mode = 'comparison';
              state.selectedId = null;
              navigate('overview');
            } else navigate('participants', { initialId: i.key });
          }),
      );
    }
    $('query').oninput = update;
    update();
    $('query').focus();
  }
  function download(file, data, type) {
    const url = URL.createObjectURL(new Blob([data], { type }));
    const a = document.createElement('a');
    a.download = file;
    a.href = url;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
  const choice = (id, glyph, title, note) =>
    '<button class="choice" id="' +
    id +
    '"><i>' +
    icon(glyph) +
    '</i><span><strong>' +
    esc(title) +
    '</strong><small>' +
    esc(note) +
    '</small></span></button>';
  function openStudy() {
    modal(
      'Open study',
      '<label class="drop-zone" id="drop" tabindex="0" role="button">' +
        icon('upload') +
        '<strong>Choose a participant CSV</strong><small>or drop a file here · up to 8 MB</small><input id="csvFile" type="file" accept=".csv,text/csv" hidden></label><p class="hint">One row per participant, group, visit, and cell population. Include cell counts and at least one numeric score. Imported studies support descriptive comparisons.</p><div id="importError" class="error" role="alert" hidden></div>' +
        (localStudyClient.available()
          ? choice(
              'h5map',
              'layers',
              'Map H5AD columns',
              'Participant, visits, groups and outcome scores',
            )
          : '<details class="launch"><summary>Use an H5AD file</summary><p>From a checkout of this branch, install the package and launch it with a local study file.</p><code>python -m pip install -e .<br>python -m sctrial.interactive study.h5ad</code></details>') +
        choice(
          'openWorkbook',
          'file',
          'Open analysis workbook',
          'Add saved analysis sheets to the current study',
        ) +
        choice('template', 'file', 'Download example CSV', 'Column names and a small test study') +
        choice(
          'restore',
          'layers',
          'Open example study',
          'Synthetic cohort with fitted reference results',
        ),
    );
    $('openWorkbook').onclick = () => {
      const picker = document.createElement('input');
      picker.type = 'file';
      picker.id = 'workbookFile';
      picker.accept = '.json,application/json';
      picker.hidden = true;
      const openDialog = $('dialog');
      $('dialogBody').append(picker);
      picker.oncancel = () => picker.remove();
      picker.onchange = async () => {
        const file = picker.files[0];
        if (!file) return;
        try {
          if (file.size > 2 * 1024 * 1024) throw Error('Choose a workbook smaller than 2 MB.');
          const text = await file.text();
          if (!openDialog.open || !picker.isConnected) return;
          const count = sheets.importWorkbook(text);
          finish();
          toast(count === 1 ? 'One analysis sheet opened' : count + ' analysis sheets opened');
        } catch (e) {
          if (!openDialog.open || !picker.isConnected) return;
          $('importError').textContent = e.message;
          $('importError').hidden = false;
        } finally {
          picker.remove();
        }
      };
      picker.click();
    };
    async function read(file) {
      if (!file) return;
      try {
        if (file.size > 8 * 1024 * 1024) throw Error('Choose a CSV smaller than 8 MB.');
        const next = readStudyCSV(await file.text(), file.name);
        finish();
        await localStudyClient.openCSV(next, init);
      } catch (e) {
        $('importError').textContent = e.message;
        $('importError').hidden = false;
      }
    }
    const offlineApp = document.createElement('a');
    offlineApp.className = 'offline-app';
    offlineApp.textContent = 'Download offline app';
    offlineApp.href = localStudyClient.available()
      ? '/sctrial-workbook.html'
      : location.href.split('#')[0];
    offlineApp.download = 'sctrial-workbook.html';
    $('dialogBody').append(offlineApp);
    if ($('h5map'))
      $('h5map').onclick = () => {
        finish();
        void localStudyClient.openMapping(init).catch((e) => toast(e.message));
      };
    $('csvFile').onchange = (e) => read(e.target.files[0]);
    $('drop').onkeydown = (e) => {
      if ([' ', 'Enter'].includes(e.key)) {
        e.preventDefault();
        $('csvFile').click();
      }
    };
    for (const event of ['dragenter', 'dragover'])
      $('drop').addEventListener(event, (e) => {
        e.preventDefault();
        $('drop').classList.add('over');
      });
    $('drop').ondragleave = () => $('drop').classList.remove('over');
    $('drop').ondrop = (e) => {
      e.preventDefault();
      $('drop').classList.remove('over');
      read(e.dataTransfer.files[0]);
    };
    $('template').onclick = () => download('participant-summaries.csv', studyCSVTemplate(), 'text/csv');
    $('restore').onclick = () => {
      finish();
      init(sample);
      toast('Example study opened');
    };
  }
  function csv(rows) {
    return (
      rows
        .map((row) =>
          row
            .map((v) => {
              if (v == null) return '';
              if (typeof v === 'number') return String(v);
              let s = String(v);
              if (/^[\s]*[=+@-]/.test(s)) s = "'" + s;
              return '"' + s.replace(/"/g, '""') + '"';
            })
            .join(','),
        )
        .join('\r\n') + '\r\n'
    );
  }
  function exportFile(kind) {
    if (kind === 'workbook') {
      download('sctrial-workbook.json', sheets.exportWorkbook(), 'application/json');
    } else if (kind === 'settings') {
      const spec = {
        study: name(),
        source: source(),
        design: state.design,
        orderedGroups: armsIn(),
        visits: datesIn(),
        population: state.population,
        featurePanel: study.features.map((f) => ({
          ...f,
          unit: unitOf(f),
          fdrFamily: fdrFamily(f),
          fitPackage: fitProvenance(f).package || null,
        })),
        fdrFamily: fdrFamily(feature()),
        selectedFeature: state.feature,
        unit: unit(),
        method: method(),
        ...fitProvenance(feature()),
        ...(native()
          ? {
              mapping: study.mapping || null,
              version: study.version || null,
              warnings: currentFit?.warnings || [],
            }
          : {}),
      };
      download('sctrial-comparison.json', JSON.stringify(spec, null, 2), 'application/json');
    } else if (kind === 'results') {
      const headers = [
        'feature',
        'measure',
        'design',
        'first_group',
        'second_group',
        'visits',
        'population',
        'estimate',
        'n',
        ...(fitted() ? ['ci_lower', 'ci_upper', 'p', 'fdr', 'fdr_family', 'fit_package'] : []),
        'unit',
      ];
      const rows = currentRows().map((r) => [
        r.label,
        state.design === 'observe' ? 'mean' : fitted() ? 'model estimate' : 'descriptive difference',
        state.design,
        state.arm,
        armsIn()[1] ?? '',
        datesIn().join(' to '),
        state.population,
        r.estimate,
        r.n,
        ...(fitted() ? [r.lo, r.hi, r.p, r.q, r.fdrFamily, r.fitPackage] : []),
        r.unit,
      ]);
      download('sctrial-results.csv', csv([headers, ...rows]), 'text/csv');
    } else {
      const header = [
        'participant',
        'arm',
        'visit',
        'population',
        'cells',
        ...study.features.map((f) => {
          if (f.inputSource === 'csv' && f.inputName) return f.inputName;
          return ['gene', 'pathway', 'module'].includes(f.kind) ? f.kind + ':' + f.label : f.label;
        }),
      ];
      const rows = (byPopulation.get(state.population) || [])
        .filter((r) => armsIn().includes(r.arm) && datesIn().includes(r.visit))
        .map((r) => [
          r.id,
          r.arm,
          r.visit,
          r.population,
          r.cells,
          ...study.features.map((f) => r.scores[f.id]),
        ]);
      download('sctrial-participant-summaries.csv', csv([header, ...rows]), 'text/csv');
    }
    toast('Export downloaded');
  }
  function exportMenu() {
    modal(
      'Export study data',
      choice(
        'saveWorkbook',
        'file',
        'Save analysis workbook',
        'Reopen all analysis sheets with this study',
      ) +
        choice(
          'resFile',
          'chart',
          'Feature results · CSV',
          fitted()
            ? 'Estimates, intervals, p-values and fitted counts'
            : 'Descriptive estimates and participant counts',
        ) +
        choice(
          'setFile',
          'code',
          'Comparison settings · JSON',
          'Group order, visits, population and feature panel',
        ) +
        choice(
          'obsFile',
          'people',
          'Participant summaries · CSV',
          'Includes participant IDs for the selected groups and visits',
        ),
    );
    $('resFile').disabled = !ready();
    if (native() && fitted() && fitStatus === 'idle') void requestFit();
    for (const [id, kind] of [
      ['saveWorkbook', 'workbook'],
      ['resFile', 'results'],
      ['setFile', 'settings'],
      ['obsFile', 'observations'],
    ])
      $(id).onclick = () => {
        finish();
        exportFile(kind);
      };
  }
  function info() {
    const entries = [
      ['Study', name()],
      ['Source', source()],
      ['Measurement', unit()],
      ['Method', method()],
      ...(synthetic()
        ? [
            ['Fit package', fitProvenance(feature()).package],
            ['FDR family', fdrFamily(feature())],
            ['Interpretation', sample.study.smallSample],
          ]
        : []),
      ...(fitWarnings().length ? [['Model notes', fitWarnings().join(' ')]] : []),
    ];
    modal(
      'Study information',
      '<dl class="meta-list">' +
        entries.map(([k, v]) => '<div><dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd></div>').join('') +
        '</dl>',
    );
  }
  function snapshotSpec() {
    return Object.fromEntries(
      ['design', 'arm', 'baseline', 'visit', 'browseVisit', 'population', 'feature'].map((k) => [
        k,
        state[k],
      ]),
    );
  }
  function travel(back) {
    const from = back ? past : future,
      to = back ? future : past;
    if (!from.length) return;
    to.push(historyEntry());
    const entry = from.pop();
    Object.assign(state, entry.spec);
    study.visits = entry.visitOrder;
    state.selectedId = null;
    clearFit();
    render();
  }
  function historyEntry() {
    return { spec: snapshotSpec(), visitOrder: [...study.visits] };
  }
  function remember() {
    past.push(historyEntry());
    if (past.length > 40) past.shift();
    future.length = 0;
  }
  const actions = {
    open: openStudy,
    configure: () => configure(),
    closeRecord: () => closeRecord(true),
    saveWorkbook: () => exportFile('workbook'),
    export: exportMenu,
    search,
    info,
    results: () => navigate('results'),
    cohort: () => navigate('participants'),
    personRecord: () => {
      if (state.selectedId) {
        finish();
        navigate('participants', { initialId: state.selectedId });
      }
    },
    pin: () => {
      if (ready()) bench.pin();
    },
    retry: () => void requestFit(),
    undo: () => travel(true),
    redo: () => travel(false),
  };
  document.addEventListener('sctrial:menu-command', ({ detail }) => {
    if (!detail || document.querySelector('dialog[open]')) return;
    if (detail.action && Object.hasOwn(actions, detail.action)) actions[detail.action]();
    else if (detail.view === 'overview' && detail.mode) changeViewMode(detail.mode);
    else if (Object.hasOwn(titles, detail.view)) navigate(detail.view);
  });
  document.addEventListener('click', (e) => {
    const view = e.target.closest('[data-view]');
    if (view) {
      navigate(view.dataset.view);
      return;
    }
    const f = e.target.closest('[data-feature-id]');
    if (f) {
      setFeature(f.dataset.featureId);
      return;
    }
    const configureButton = e.target.closest('[data-configure]');
    if (configureButton) {
      configure(configureButton.dataset.configure);
      return;
    }
    const m = e.target.closest('button[data-mode]');
    if (m) {
      changeViewMode(m.dataset.mode);
      return;
    }
    const action = e.target.closest('[data-action]');
    if (action) actions[action.dataset.action]?.();
  });
  document.addEventListener('change', (e) => {
    if (e.target.id === 'chartMode') {
      changeViewMode(e.target.value);
      return;
    }
    if (e.target.dataset.param) {
      const field = e.target.dataset.param;
      if (field === 'feature') setFeature(e.target.value);
      else setParam(field, e.target.value);
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && $('recordDrawer') && !document.querySelector('dialog[open]')) {
      e.preventDefault();
      closeRecord(true);
      return;
    }
    if (
      document.querySelector('dialog[open]') ||
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) ||
      e.target.isContentEditable
    )
      return;
    if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      travel(!e.shiftKey);
      return;
    }
    if (
      (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey) ||
      ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'k')
    ) {
      e.preventDefault();
      search();
    }
  });
  init(sample);
  if (localStudyClient.available())
    void localStudyClient.openMapping(init).catch((e) => toast(e.message));
})();
