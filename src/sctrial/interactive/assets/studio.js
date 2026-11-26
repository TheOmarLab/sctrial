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
  const titles = {
    overview: 'Contrast studio',
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
  const unit = () =>
    synthetic() ? 'Mean module score' : native() ? study.unit || 'Mean score' : 'Recorded score';
  const nfmt = (n) => (Number.isFinite(n) ? n.toLocaleString('en-US') : '—');
  const mean = (xs) => (xs.length ? xs.reduce((x, y) => x + y, 0) / xs.length : null);
  const number = (n, sign = false, prec = 2) =>
    Number.isFinite(n) ? (n < 0 ? '−' : sign && n > 0 ? '+' : '') + Math.abs(n).toFixed(prec) : '—';
  const probability = (n) =>
    Number.isFinite(n) ? (n > 0 && n < 0.001 ? n.toExponential(1) : n.toFixed(3)) : '—';
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
  const color = (a) => (arms.indexOf(a) === 0 ? '#70ddbf' : '#ffb879');
  const lightColor = (a) => (arms.indexOf(a) === 0 ? '#4b9680' : '#c78345');
  const rowAt = (id, v) => obsByPerson.get(id)?.get(v)?.get(state.population);
  const rowObserved = (r) => r && r.cells > 0;
  const eligible = (p) =>
    armsIn().includes(p.arm) && datesIn().every((v) => rowObserved(rowAt(p.id, v)));
  const method = () =>
    synthetic()
      ? study.study.methods[state.design]
      : fitted()
        ? currentFit?.method || 'Participant-level model'
        : state.design === 'observe'
          ? 'Mean across observed participants'
          : 'Unadjusted descriptive participant differences';
  const ready = () => !native() || !fitted() || fitStatus === 'ready';
  const benchmarkMethods = {
    did: 'Two-group change',
    paired: 'Within-group change',
    cross: 'At one visit',
    observe: 'Single-group mean',
  };
  function clearFit() {
    if (fitTask) fitTask.abort();
    fitTask = null;
    currentFit = null;
    fitStatus = 'idle';
    fitMessage = '';
  }
  function init(next) {
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
      visit: study.visits.find((v) => v !== baseline) || baseline,
      population: study.populations[0],
      feature: study.features[0].id,
      mode: 'trajectory',
      selectedId: null,
      focus: false,
    };
    peopleContext = {};
    bench.setStudy(synthetic() ? 'immune-study-62834-v1' : 'local-' + ++privateCounter, {
      persist: synthetic(),
    });
    render();
    if (native() && fitted()) void requestFit();
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
      currentFit = fit;
      fitStatus = 'ready';
      if (fitCache.size >= 24) fitCache.delete(fitCache.keys().next().value);
      fitCache.set(k, fit);
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
  function refreshFit() {
    updateTop();
    if (state.view === 'overview') paintInspector();
    else if (state.view === 'results') renderResultsView();
  }
  function setParam(field, value) {
    if (state[field] === value) return;
    remember();
    state[field] = value;
    state.selectedId = null;
    if (isLong() && state.baseline === state.visit)
      state.visit = study.visits.find((v) => v !== state.baseline);
    clearFit();
    render();
    if (native() && fitted()) void requestFit();
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
        visit,
        population,
        feature: study.features.some((f) => f.id === id) ? id : study.features[0].id,
        mode: mode || 'trajectory',
        selectedId: null,
      });
      clearFit();
      navigate('overview');
      if (native() && fitted()) void requestFit();
    },
    onNotice: (message) => toast(message),
    onChange: (n) => setBadge(n),
    onExplore: () => navigate('overview'),
  });
  function setBadge(count) {
    $('pinCount').textContent = String(count);
    $('pinCount').hidden = !count;
  }
  function updateTop() {
    $('studyName').textContent = name();
    $('sourceBadge').textContent = source();
    $('studyCount').textContent = nfmt(study.participants.length) + ' participants';
    $('pageTitle').textContent = titles[state.view];
    $('pinButton').hidden = state.view === 'bench';
    $('pinButton').disabled = !ready();
    $('pinButton').title = ready() ? 'Save this comparison' : 'Waiting for a fitted result';
    document.body.classList.toggle('workspace-focus', state.focus && state.view === 'overview');
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
  function toolbar() {
    function control(field, label, glyph, options, value, caption) {
      return (
        '<label class="control"><i class="label-icon">' +
        icon(glyph) +
        '</i><span><small>' +
        esc(label) +
        '</small><b>' +
        esc(caption ?? value) +
        '</b></span><select id="ctrl-' +
        field +
        '" data-param="' +
        field +
        '" aria-label="' +
        esc(label) +
        '">' +
        options +
        '</select><i>' +
        icon('down') +
        '</i></label>'
      );
    }
    const single = ['paired', 'observe'].includes(state.design),
      groupPrint = (a) => (single ? a : a + ' − ' + arms.find((v) => v !== a));
    return (
      '<div class="controlbar" aria-label="Active contrast">' +
      control(
        'design',
        'Study design',
        'paired',
        designs()
          .map((d) => option(d, state.design, labels[d]))
          .join(''),
        state.design,
        labels[state.design],
      ) +
      control(
        'arm',
        single ? 'Group' : 'Group order',
        'people',
        arms.map((a) => option(a, state.arm, groupPrint(a))).join(''),
        state.arm,
        groupPrint(state.arm),
      ) +
      control(
        'population',
        'Cell population',
        'cells',
        study.populations.map((p) => option(p, state.population)).join(''),
        state.population,
      ) +
      '</div>'
    );
  }
  function visitSelector() {
    const dates = isLong() ? study.visits.filter((v) => v !== state.baseline) : study.visits;
    const choose = (param, label, values, locked) =>
      '<label class="time-pick"><small>' +
      esc(label) +
      '</small>' +
      (locked
        ? '<b>' + esc(state[param]) + '</b>'
        : '<select data-param="' +
          param +
          '" aria-label="' +
          esc(label) +
          '">' +
          values.map((v) => option(v, state[param])).join('') +
          '</select>') +
      '</label>';
    return (
      '<div class="timebar">' +
      (isLong()
        ? choose('baseline', 'Baseline visit', study.visits, synthetic()) +
          '<span aria-hidden="true">→</span>'
        : '') +
      choose('visit', isLong() ? 'Follow-up visit' : 'Selected visit', dates, false) +
      '</div>'
    );
  }
  function paintCanvas() {
    const active = armsIn();
    let visits = study.visits;
    if (!synthetic() && isLong())
      visits = [state.baseline, ...study.visits.filter((v) => v !== state.baseline)];
    const series = byPopulation.get(state.population) || [],
      byId = new Map(),
      id = state.feature;
    for (const r of series) {
      if (!active.includes(r.arm) || !Number.isFinite(r.scores[id])) continue;
      if (!byId.has(r.id)) byId.set(r.id, { id: r.id, arm: r.arm, points: [] });
      byId.get(r.id).points.push({ visit: r.visit, value: r.scores[id] });
    }
    renderContrastCanvas(
      $('plot'),
      {
        visits,
        baseline: state.baseline,
        visit: state.visit,
        design: state.design,
        feature: feature(),
        unit: unit(),
        groups: active.map((a) => ({ name: a, color: color(a) })),
        series: [...byId.values()],
        mode: state.mode,
        selectedId: state.selectedId,
        canSelectVisit: visits.length > 1,
        editableVisits: visits.filter((v) => !isLong() || v !== state.baseline),
      },
      { onSelectParticipant: selectPerson, onVisit: (v) => setParam('visit', v) },
    );
  }
  function selectPerson(id) {
    state.selectedId = id ?? null;
    paintCanvas();
    paintInspector();
    syncRoster();
    chartInstruction();
  }
  function setFeature(id) {
    if (!study.features.some((f) => f.id === id)) return;
    state.feature = id;
    state.selectedId = null;
    if (state.view !== 'overview') {
      navigate('overview');
      return;
    }
    $('stageTitle').textContent = feature().label;
    paintCanvas();
    paintInspector();
    syncRoster();
    chartInstruction();
  }
  function syncRoster() {
    document
      .querySelectorAll('[data-person-index]')
      .forEach((b) =>
        b.setAttribute(
          'aria-pressed',
          String(study.participants[Number(b.dataset.personIndex)].id === state.selectedId),
        ),
      );
  }
  function roster() {
    const active = armsIn(),
      selected = study.participants.filter((p) => active.includes(p.arm)),
      coverage = selected.filter(eligible).length;
    const groupHtml = active
      .map((a) => {
        const targets = study.participants.map((p, i) => ({ p, i })).filter((it) => it.p.arm === a);
        return (
          '<div class="roster-group"><span class="roster-label">' +
          esc(a) +
          '</span><div class="roster-keys">' +
          targets
            .slice(0, 14)
            .map(({ p, i }) => {
              const mark = datesIn()
                .map((v) => '<em class="' + (rowObserved(rowAt(p.id, v)) ? '' : 'none') + '"></em>')
                .join('');
              return (
                '<button type="button" data-person-index="' +
                i +
                '" class="slot" style="--mark:' +
                lightColor(a) +
                '" aria-label="Inspect ' +
                esc(p.id) +
                '" aria-pressed="' +
                (state.selectedId === p.id) +
                '">' +
                esc(p.id) +
                '<span aria-hidden="true">' +
                mark +
                '</span></button>'
              );
            })
            .join('') +
          (targets.length > 14 ? button('cohort', 'arrow', '+' + (targets.length - 14)) : '') +
          '</div></div>'
        );
      })
      .join('');
    return (
      '<section class="roster"><div class="roster-head"><div><h2>Participant roster</h2><small>' +
      coverage +
      ' of ' +
      selected.length +
      ' with ' +
      (isLong() ? 'both visits' : 'the selected visit') +
      '</small></div>' +
      button('cohort', 'arrow', 'Open cohort') +
      '</div><div class="roster-groups">' +
      groupHtml +
      '</div><div class="roster-legend"><i></i>Observed <i></i>Missing <span>· ' +
      esc(datesIn().join(' → ')) +
      '</span></div></section>'
    );
  }
  function overview() {
    const focusKey = document.activeElement?.getAttribute('data-param');
    if ($('plot')) disposeContrastCanvas($('plot'));
    $('page').innerHTML =
      toolbar() +
      '<div class="studio-grid"><section class="stage" aria-labelledby="stageTitle"><div class="stage-head"><div><div class="stage-context">' +
      esc(state.population) +
      '</div><h2 id="stageTitle">' +
      esc(feature().label) +
      '</h2></div><div class="stage-buttons"><div class="stage-toggle" role="group" aria-label="Plot view">' +
      ['trajectory', 'changes']
        .map(
          (mode, i) =>
            '<button data-mode="' +
            mode +
            '" aria-pressed="' +
            (mode === state.mode) +
            '">' +
            icon(i ? 'points' : 'trend') +
            (i ? (isLong() ? 'Changes' : 'Distribution') : 'Trajectories') +
            '</button>',
        )
        .join('') +
      '</div><button data-action="focus" class="stage-expand" title="' +
      (state.focus ? 'Exit focus view' : 'Focus view') +
      '" aria-label="' +
      (state.focus ? 'Exit focus view' : 'Focus view') +
      '">' +
      icon(state.focus ? 'shrink' : 'expand') +
      '</button></div></div><div id="plot" class="stage-plot"></div><div class="stage-foot"><div class="chart-instruction" id="chartInstruction"></div>' +
      visitSelector() +
      '</div></section><aside class="inspector" id="inspector" aria-label="Inspector" tabindex="-1"></aside></div>' +
      roster();
    paintCanvas();
    paintInspector();
    chartInstruction();
    if (focusKey) document.querySelector('[data-param="' + focusKey + '"]')?.focus();
  }
  function selectedEstimate() {
    return currentRows().find((r) => r.id === state.feature);
  }
  function intervalGraphic(row) {
    if (![row.lo, row.estimate, row.hi].every(Number.isFinite) || row.lo > row.hi) return '';
    const extent = Math.max(Math.abs(row.lo), Math.abs(row.hi), Math.abs(row.estimate));
    if (!extent) return '';
    const position = (value) => 50 + 45 * (value / extent);
    const start = position(row.lo),
      end = position(row.hi);
    return (
      '<div class="interval-ruler" aria-hidden="true"><div class="ruler-line"></div><div class="ruler-zero"><small>0</small></div>' +
      '<div class="ruler-whisker" style="left:' +
      start +
      '%;width:' +
      (end - start) +
      '%"></div>' +
      '<i style="left:' +
      position(row.estimate) +
      '%"></i></div>'
    );
  }
  function chartInstruction() {
    const el = $('chartInstruction');
    if (!el) return;
    if (state.selectedId != null) {
      el.innerHTML = button('inspectSelection', 'person', 'Inspect ' + state.selectedId);
    } else {
      el.innerHTML =
        icon('crosshair') +
        '<span>' +
        (state.mode === 'changes' ? 'Select a point' : 'Select a trace') +
        ' to inspect</span>';
    }
  }
  function paintInspector() {
    const box = $('inspector');
    if (!box) return;
    const active = box.contains(document.activeElement)
      ? document.activeElement.dataset.featureId
      : null;
    const person = state.selectedId == null ? null : personById.get(state.selectedId);
    if (person) {
      const visits = study.visits.map((v) => ({ v, r: rowAt(person.id, v) })),
        first = rowAt(person.id, state.baseline)?.scores[state.feature],
        last = rowAt(person.id, state.visit)?.scores[state.feature];
      const summary = isLong()
        ? Number.isFinite(last) && Number.isFinite(first)
          ? last - first
          : null
        : last;
      box.innerHTML =
        '<div class="ins-head"><span>' +
        icon('person') +
        'Participant inspector</span><button class="icon-btn" id="clearSelection" aria-label="Clear participant selection">' +
        icon('close') +
        '</button></div><div class="subject-title"><span class="subject-swatch" style="background:' +
        lightColor(person.arm) +
        '"></span><h3>' +
        esc(person.id) +
        '</h3><span>' +
        esc(person.arm) +
        '</span></div><div class="est-box"><div class="est-num">' +
        number(summary, isLong()) +
        '</div><div class="est-caption">' +
        esc(isLong() ? 'Within-participant change' : unit()) +
        '</div></div><h3>Visit measurements</h3><div class="ins-scroll"><table class="ins-tbl"><thead><tr><th>Visit</th><th>Score</th><th>Cells</th></tr></thead><tbody>' +
        visits
          .map(
            ({ v, r }) =>
              '<tr class="' +
              (datesIn().includes(v) ? 'in' : '') +
              '"><td>' +
              esc(v) +
              '</td><td>' +
              number(r?.scores[state.feature]) +
              '</td><td>' +
              nfmt(r?.cells) +
              '</td></tr>',
          )
          .join('') +
        '</tbody></table></div><div class="ins-bottom">' +
        button('personRecord', 'arrow', 'Participant record') +
        '</div>';
      $('clearSelection').onclick = () => {
        selectPerson(null);
        document.querySelector('[data-feature-id][aria-pressed=true]')?.focus();
      };
      return;
    }
    const rows = currentRows(),
      r = selectedEstimate(),
      loading = fitStatus === 'loading',
      have = ready(),
      finite = Number.isFinite(r?.estimate);
    const ci = r && r.lo != null && r.hi != null,
      ciText = ci ? number(r.lo, true) + ' to ' + number(r.hi, true) : 'Unavailable';
    const subtitle = synthetic() ? 'Reference fit' : native() && fitted() ? 'Local fit' : 'Descriptive';
    const fitLabel =
      state.design === 'did'
        ? 'Difference in change'
        : state.design === 'paired'
          ? 'Paired change'
          : state.design === 'cross'
            ? 'Group difference'
            : 'Participant mean';
    let result =
      '<div class="ins-head"><span>' +
      icon('crosshair') +
      'Contrast inspector</span><small class="ins-method">' +
      subtitle +
      '</small></div><div class="est-box"><h3>' +
      fitLabel +
      '</h3><div class="est-num">' +
      (have && finite ? number(r.estimate, true) : '—') +
      '</div><div class="est-caption">' +
      esc(contrast()) +
      '</div>';
    if (have && r) {
      if (ci) result += intervalGraphic(r);
      result +=
        '<div class="estimate-ticks">' +
        (fitted() ? '<span>95% interval<b>' + esc(ciText) + '</b></span>' : '') +
        '<span>Participants<b>' +
        nfmt(r.n) +
        '</b></span></div>';
    } else {
      result +=
        '<p class="small" role="status">' +
        (loading ? 'Fitting selected comparison…' : esc(fitMessage || 'Fit unavailable.')) +
        '</p>' +
        (!loading ? button('retry', 'refresh', 'Retry fit') : '');
    }
    if (have && r && !finite && r.reason) result += '<p class="small">' + esc(r.reason) + '</p>';
    result +=
      '</div><div class="outcomes-head"><h3>Outcome panel</h3><span>' +
      study.features.length +
      ' scores</span></div><div class="outcomes-shell"><div class="outcomes">' +
      study.features
        .map((f) => {
          const row = rows.find((x) => x.id === f.id);
          return (
            '<button data-feature-id="' +
            esc(f.id) +
            '" class="outcome" aria-pressed="' +
            (f.id === state.feature) +
            '"><i class="radio"></i><span class="flabel">' +
            esc(f.label) +
            '</span><strong>' +
            number(row?.estimate, true) +
            '</strong></button>'
          );
        })
        .join('') +
      '</div></div><div class="ins-bottom">' +
      button('results', 'arrow', 'Open results table') +
      '</div>';
    box.innerHTML = result;
    if (active)
      Array.from(box.querySelectorAll('[data-feature-id]'))
        .find((b) => b.dataset.featureId === active)
        ?.focus();
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
          state.selectedId = null;
          navigate('overview');
        },
        onExport: () => exportFile('results'),
      },
    );
    if (currentFit?.warnings?.length) {
      const d = document.createElement('details');
      d.className = 'fit-warnings';
      const summary = document.createElement('summary');
      summary.textContent = 'Model notes (' + currentFit.warnings.length + ')';
      d.append(summary);
      for (const w of currentFit.warnings) {
        const p = document.createElement('p');
        p.textContent = w;
        d.append(p);
      }
      page.append(d);
    }
  }
  function render() {
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
  }
  function navigate(view, context = {}) {
    peopleContext = context;
    state.view = view;
    state.focus = false;
    render();
    window.scrollTo(0, 0);
  }
  function modal(title, html) {
    const d = $('dialog');
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
        choice('template', 'file', 'Download example CSV', 'Column names and a small test study') +
        choice(
          'restore',
          'layers',
          'Open example study',
          'Synthetic cohort with fitted reference results',
        ),
    );
    async function read(file) {
      if (!file) return;
      try {
        if (file.size > 8 * 1024 * 1024) throw Error('Choose a CSV smaller than 8 MB.');
        const next = readStudyCSV(await file.text(), file.name);
        finish();
        init(next);
        toast('Study opened');
      } catch (e) {
        $('importError').textContent = e.message;
        $('importError').hidden = false;
      }
    }
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
    if (kind === 'settings') {
      const spec = {
        study: name(),
        source: source(),
        design: state.design,
        orderedGroups: armsIn(),
        visits: datesIn(),
        population: state.population,
        featurePanel: study.features,
        selectedFeature: state.feature,
        unit: unit(),
        method: method(),
        ...(synthetic()
          ? { package: study.study.package, commit: study.study.commit, seed: study.study.seed }
          : {}),
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
        ...(fitted() ? ['ci_lower', 'ci_upper', 'p', 'fdr'] : []),
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
        ...(fitted() ? [r.lo, r.hi, r.p, r.q] : []),
      ]);
      download('sctrial-results.csv', csv([headers, ...rows]), 'text/csv');
    } else {
      const header = [
        'participant',
        'arm',
        'visit',
        'population',
        'cells',
        ...study.features.map((f) => f.label),
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
    for (const [id, kind] of [
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
            ['Package', sample.study.package],
            ['FDR', sample.study.multiplicity],
            ['Interpretation', sample.study.smallSample],
          ]
        : []),
      ...(native() && currentFit?.warnings?.length
        ? [['Model notes', currentFit.warnings.join(' ')]]
        : []),
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
      ['design', 'arm', 'baseline', 'visit', 'population'].map((k) => [k, state[k]]),
    );
  }
  function travel(back) {
    const from = back ? past : future,
      to = back ? future : past;
    if (!from.length) return;
    to.push(snapshotSpec());
    Object.assign(state, from.pop());
    state.selectedId = null;
    clearFit();
    render();
    if (native() && fitted()) void requestFit();
  }
  function remember() {
    past.push(snapshotSpec());
    if (past.length > 40) past.shift();
    future.length = 0;
  }
  const actions = {
    inspectSelection: () => {
      const panel = $('inspector');
      panel?.focus({ preventScroll: true });
      panel?.scrollIntoView({ block: 'nearest' });
    },
    open: openStudy,
    export: exportMenu,
    search,
    info,
    results: () => navigate('results'),
    cohort: () => navigate('participants'),
    personRecord: () => {
      if (state.selectedId) navigate('participants', { initialId: state.selectedId });
    },
    pin: () => {
      if (ready()) bench.pin();
    },
    retry: () => void requestFit(),
    focus: () => {
      state.focus = !state.focus;
      updateTop();
      overview();
      document.querySelector('[data-action="focus"]').focus();
    },
    undo: () => travel(true),
    redo: () => travel(false),
  };
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
    const p = e.target.closest('[data-person-index]');
    if (p) {
      const id = study.participants[Number(p.dataset.personIndex)].id;
      selectPerson(state.selectedId === id ? null : id);
      return;
    }
    const m = e.target.closest('[data-mode]');
    if (m) {
      if (state.mode !== m.dataset.mode) {
        state.mode = m.dataset.mode;
        overview();
        document.querySelector('[data-mode="' + state.mode + '"]').focus();
      }
      return;
    }
    const action = e.target.closest('[data-action]');
    if (action) actions[action.dataset.action]?.();
  });
  document.addEventListener('change', (e) => {
    if (e.target.dataset.param) {
      const field = e.target.dataset.param;
      setParam(field, e.target.value);
    }
  });
  document.addEventListener('keydown', (e) => {
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
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      search();
    }
    if (e.key === 'Escape' && state.focus) {
      state.focus = false;
      render();
    }
  });
  init(sample);
  if (localStudyClient.available())
    void localStudyClient.openMapping(init).catch((e) => toast(e.message));
})();
