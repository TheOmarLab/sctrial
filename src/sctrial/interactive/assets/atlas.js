(() => {
  'use strict';

  const instances = new WeakMap();
  const modes = {
    cells: ['Cells', 'cells'],
    genes: ['Expression', 'gene'],
    pathways: ['Pathways', 'path'],
    modules: ['Scores', 'score'],
  };
  const paths = {
    cells:
      '<circle cx="8" cy="7" r="4"/><circle cx="15" cy="13" r="3.5"/><circle cx="6" cy="15" r="2.5"/>',
    gene: '<path d="M5 2c0 7 10 9 10 16M15 2c0 7-10 9-10 16M6 5h8M8 8h4M6 15h8"/>',
    path: '<path d="m5 6 8 6m-8-6 1 9m7-3 3-7"/><circle cx="4" cy="5" r="2"/><circle cx="14" cy="13" r="2.5"/><circle cx="6" cy="16" r="1.5"/><circle cx="16" cy="4" r="1.5"/>',
    score: '<path d="M3 15h14M5 12V8m5 4V3m5 9V6"/>',
    plot: '<path d="M3 3v13h14M6 12l4-4 3 1 4-5"/><circle cx="6" cy="12" r=".85"/><circle cx="10" cy="8" r=".85"/>',
    down: '<path d="m6 8 4 4 4-4"/>',
    close: '<path d="m5 5 10 10m0-10L5 15"/>',
    minus: '<path d="M4 10h12"/>',
    plus: '<path d="M4 10h12m-6-6v12"/>',
    reset: '<path d="M3 8V3h5m4 0h5v5m0 4v5h-5m-4 0H3v-5"/>',
    play: '<path d="m6 4 10 6-10 6z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M6 4v12m7-12v12" stroke-width="3"/>',
    previous: '<path d="m12 5-5 5 5 5"/>',
    next: '<path d="m8 5 5 5-5 5"/>',
    info: '<circle cx="10" cy="10" r="7.5"/><path d="M10 9v5m0-8v.2"/>',
    layers: '<path d="m2 7 8-4 8 4-8 4zM3 11l7 3.5 7-3.5M3 15l7 3 7-3"/>',
    controls:
      '<path d="M3 6h4m4 0h6M3 14h8m4 0h2"/><circle cx="9" cy="6" r="2"/><circle cx="13" cy="14" r="2"/>',
    atlas:
      '<circle cx="4.5" cy="6" r="1.6"/><circle cx="9" cy="3.5" r="1.5"/><circle cx="9" cy="9" r="2.3"/><circle cx="3.8" cy="11.2" r="1.1"/><circle cx="6" cy="15.4" r="1.6"/><circle cx="12.7" cy="14.9" r="2.2"/><circle cx="15.4" cy="7.4" r="1.7"/>',
  };
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function icon(name) {
    const n = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    n.setAttribute('viewBox', '0 0 20 20');
    n.setAttribute('aria-hidden', 'true');
    n.innerHTML = paths[name];
    return n;
  }
  function button(text, image, cls, click, onlyIcon = false) {
    const b = el('button', cls);
    b.type = 'button';
    b.setAttribute('aria-label', text);
    if (image) b.append(icon(image));
    if (!onlyIcon) b.append(el('span', '', text));
    else b.title = text;
    b.addEventListener('click', click);
    return b;
  }
  function picker(label, change) {
    const box = el('label', 'at-field');
    const name = el('span', 'at-field-name', label);
    const select = el('select');
    select.setAttribute('aria-label', label);
    select.addEventListener('change', () => change(select.value));
    box.append(name, select, icon('down'));
    return { box, select, name };
  }
  function fillSelect(select, entries, value) {
    const options = Array.from(select.options);
    if (
      entries.length !== options.length ||
      entries.some(([k, v], i) => options[i].value !== k || options[i].text !== v)
    )
      select.replaceChildren(...entries.map(([k, v]) => new Option(v, k)));
    select.value = value ?? '';
  }
  function featureList(study, mode) {
    const all = study.features || [];
    if (mode === 'genes') return all.filter((f) => f.kind === 'gene');
    if (mode === 'pathways') return all.filter((f) => f.kind === 'pathway');
    if (mode === 'modules')
      return all.filter((f) => !f.kind || f.kind === 'module' || f.kind === 'score');
    return all;
  }
  const fmt = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 });
  const counts = new Intl.NumberFormat('en');
  const share = new Intl.NumberFormat('en', { style: 'percent', maximumSignificantDigits: 3 });
  function create(host, options) {
    let cfg = options;
    let data = window.SctrialAtlasData.prepare(cfg.study);
    const selectablePopulations = cfg.study.populations;
    const populationIndex = new Map(data.populations.map((name, i) => [name, i]));
    const state = {
      mode: 'cells',
      featureId: null,
      population: null,
      visit: null,
    };
    const memory = new Map();
    let open = false,
      timer = null,
      scene = null,
      destroyed = false;
    let atlasArm = null;
    let paints = 0;
    const abort = new AbortController();

    const root = el('section', 'at-workspace');
    root.setAttribute('data-atlas', '');
    root.setAttribute('aria-label', 'Cell atlas');
    const plane = el('div', 'at-plane');
    const catalog = el('aside', 'at-catalog');
    catalog.setAttribute('aria-label', 'Atlas explorer');
    const brand = el('header', 'at-brand');
    brand.append(icon('atlas'), el('h1', '', 'Cell atlas'));
    const controls = el('div', 'at-catalog-controls');
    controls.id = 'atStudyControls';
    const mobileToggle = button(
      'Atlas controls',
      'controls',
      'at-mobile-toggle',
      () => {
        showControls(!root.classList.contains('at-catalog-open'));
      },
      true,
    );
    mobileToggle.setAttribute('aria-controls', controls.id);
    const dismiss = button(
      'Close atlas controls',
      'close',
      'at-sheet-close',
      () => showControls(false),
      true,
    );
    const sheetHead = el('div', 'at-sheet-head');
    sheetHead.append(el('h2', '', 'Atlas controls'), dismiss);
    const scrim = button(
      'Dismiss atlas controls',
      null,
      'at-scrim',
      () => showControls(false),
      true,
    );
    scrim.tabIndex = -1;
    const head = el('header', 'at-map-head');
    const nameBlock = el('div', 'at-map-name');
    const scopeKicker = el('span', 'at-map-scope');
    const mapTitle = el('h2', 'at-map-title', 'Cell populations');
    const scope = el('section', 'at-scope');
    const arm = picker('Atlas arm', (value) => {
      stop();
      atlasArm = value === '' ? null : value;
      const before = paints;
      cfg.onArmChange?.(atlasArm);
      if (before === paints) paint();
    });
    arm.box.classList.add('at-arm');
    arm.box.title =
      'Filter the atlas by study arm. The plot uses the configured analysis contrast.';
    arm.name.textContent = 'Study arm';
    scope.append(arm.box);
    const plotButton = button('Plot results', 'plot', 'at-plot', () => openPlot());
    const plotBlock = el('div', 'at-plot-block');
    const context = el('span', 'at-context');
    const plotOutcome = el('span', 'at-plot-outcome');
    const plotContrast = el('span');
    context.append(plotOutcome, plotContrast);
    plotBlock.append(plotButton, context);
    const tools = el('section', 'at-layer-section');
    const sectionTitle = el('h2', 'at-section-title', 'Layers');
    const layers = el('div', 'at-layers');
    layers.setAttribute('role', 'group');
    layers.setAttribute('aria-label', 'Measurement overlay');
    const layerButtons = new Map();
    const layerCounts = new Map();
    const names = {
      cells: 'Cell types',
      genes: 'Gene expression',
      pathways: 'Pathways',
      modules: 'Module scores',
    };
    for (const [key, [label, glyph]] of Object.entries(modes)) {
      const btn = button(label, glyph, 'at-layer', () => {
        if (state.mode === key) return;
        const candidates = featureList(cfg.study, key);
        const prior = memory.get(key) ?? state.featureId;
        const f = candidates.find((item) => item.id === prior) || candidates[0];
        state.mode = key;
        state.featureId = f?.id ?? null;
        changed();
      });
      const fullLabel = btn.querySelector('span');
      fullLabel.textContent = names[key];
      fullLabel.className = 'at-layer-full';
      btn.append(el('span', 'at-layer-short', label));
      const total = el('small', 'at-layer-total');
      btn.append(total);
      layerCounts.set(key, total);
      layerButtons.set(key, btn);
      layers.append(btn);
    }
    const population = picker('Cell population', (name) => selectGroup(name));
    population.box.classList.add('at-pop-select');
    population.box.title = 'Choose a population, including pooled measurements';
    const popSection = el('section', 'at-pop-section');
    const popHead = el('div', 'at-pop-head');
    popHead.append(el('h2', 'at-section-title', 'Populations'), population.box);
    const popList = el('div', 'at-pop-list');
    popList.setAttribute('role', 'group');
    popList.setAttribute('aria-label', 'Study populations');
    popSection.append(popHead, popList);
    const popRows = new Map();
    const cohortTotal = el('div', 'at-total');
    const totals = [
      [el('strong'), 'Observed cells'],
      [el('strong'), 'Participants'],
    ];
    for (const [num, title] of totals) {
      const block = el('div');
      block.append(num, el('span', '', title));
      cohortTotal.append(block);
    }
    controls.append(sheetHead, scope, popSection, cohortTotal);
    tools.append(sectionTitle, layers, mobileToggle);
    catalog.append(brand, tools, controls);
    const feature = picker('Gene', (id) => {
      state.featureId = id;
      changed();
    });
    feature.box.classList.add('at-feature');
    const geneSet = el('details', 'at-geneset');
    const summary = el('summary', '', 'Gene set');
    summary.append(icon('down'));
    const setPanel = el('div', 'at-set-panel');
    geneSet.append(summary, setPanel);
    const titleLine = el('div', 'at-title-line');
    titleLine.append(mapTitle, feature.box, geneSet);
    nameBlock.append(scopeKicker, titleLine);
    head.append(nameBlock, plotBlock);
    const stage = el('div', 'at-stage');
    const sceneHost = el('div', 'at-scene');
    const positionKey = el('span', 'at-position-key', 'Schematic positions');
    positionKey.title =
      'Cluster area shows the fraction of observed cells. Positions are illustrative.';
    stage.append(sceneHost);
    const signalHost = el('div', 'at-signal');
    const signal = window.SctrialAtlasSignal.mount(signalHost);
    const legendRow = el('div', 'at-keyline');
    const legend = el('div', 'at-legend');
    const legendTitle = el('span', 'at-legend-label');
    const min = el('span', 'at-limit');
    const ramp = el('span', 'at-ramp');
    const max = el('span', 'at-limit');
    const missing = el('span', 'at-missing', 'Missing');
    const note = el('span', 'at-schematic');
    const proportions = el('div', 'at-proportions');
    proportions.setAttribute('role', 'group');
    proportions.setAttribute('aria-label', 'Share of observed cells by population');
    const segments = new Map();
    const scaleKey = el('span', 'at-scale-key');
    scaleKey.append(min, ramp, max);
    legend.append(legendTitle, proportions, scaleKey, missing, note);
    const zoom = el('div', 'at-zoom');
    zoom.append(
      button('Zoom out', 'minus', '', () => scene?.zoomBy(1 / 1.2), true),
      button('Zoom in', 'plus', '', () => scene?.zoomBy(1.2), true),
      button('Fit clusters', 'reset', '', () => scene?.reset(), true),
    );
    const mapNavigation = el('div', 'at-map-navigation');
    mapNavigation.append(positionKey, zoom);
    legendRow.append(legend, mapNavigation);
    const empty = el('div', 'at-empty');
    const emptyText = el('span');
    const load = button('Open study', null, 'at-empty-open', () => cfg.onOpenStudy?.());
    empty.append(emptyText, load);
    plane.append(head, empty, stage, legendRow, signalHost);

    const time = el('div', 'at-time');
    const play = button(
      'Play visits',
      'play',
      'at-play',
      () => {
        if (timer != null) {
          stop();
          return;
        }
        if (data.visits.length < 2) return;
        // Start at the beginning if the last observed visit is already selected.
        if (state.visit === data.visits.at(-1)) setVisit(data.visits[0], true);
        timer = setInterval(() => {
          const next = data.visits.indexOf(state.visit) + 1;
          if (next < data.visits.length) setVisit(data.visits[next], true);
          if (next >= data.visits.length - 1) stop();
        }, 1400);
        showPlay();
      },
      true,
    );
    const visit = picker('Visit', (v) => setVisit(v));
    visit.box.classList.add('at-visit');
    const track = el('div', 'at-track');
    track.setAttribute('role', 'group');
    track.setAttribute('aria-label', 'Observed visits');
    const tickButtons = new Map();
    function setTicks() {
      tickButtons.clear();
      track.replaceChildren();
      for (const v of data.visits) {
        const tick = button(v, null, 'at-tick', () => setVisit(v));
        tick.querySelector('span').append(el('small', 'at-tick-count'));
        tickButtons.set(v, tick);
        track.append(tick);
      }
    }
    setTicks();
    const back = button('Previous visit', 'previous', 'at-step', () => step(-1), true);
    const next = button('Next visit', 'next', 'at-step', () => step(1), true);
    time.append(play, visit.box, track, back, next);
    plane.append(time);

    const drawer = el('aside', 'at-drawer');
    drawer.setAttribute('aria-label', 'Results');
    const drawerHead = el('header', 'at-drawer-head');
    const drawerTitle = el('h2', '', 'Results');
    const drawerLabels = el('div', 'at-drawer-labels');
    const drawerContext = el('span', 'at-drawer-context');
    drawerLabels.append(drawerTitle, drawerContext);
    const close = button('Close results', 'close', 'at-close', () => closePlot(true), true);
    drawerHead.append(drawerLabels, close);
    const mount = el('div', 'at-result');
    mount.id = 'atlasResultMount';
    drawer.append(drawerHead, mount);
    root.append(catalog, plane, drawer, scrim);
    host.replaceChildren(root);

    function showControls(value, moveFocus = true) {
      root.classList.toggle('at-catalog-open', value);
      mobileToggle.setAttribute('aria-expanded', String(value));
      if (value) {
        controls.setAttribute('role', 'dialog');
        controls.setAttribute('aria-modal', 'true');
        controls.setAttribute('aria-label', 'Atlas controls');
        if (moveFocus) dismiss.focus();
      } else {
        controls.removeAttribute('role');
        controls.removeAttribute('aria-modal');
        controls.removeAttribute('aria-label');
        if (moveFocus) mobileToggle.focus();
      }
    }
    mobileToggle.setAttribute('aria-expanded', 'false');
    window.matchMedia('(max-width:980px)').addEventListener(
      'change',
      (event) => {
        if (!event.matches && root.classList.contains('at-catalog-open'))
          showControls(false, false);
      },
      { signal: abort.signal },
    );

    function showPlay() {
      const label = timer == null ? 'Play visits' : 'Pause visits';
      play.replaceChildren(icon(timer == null ? 'play' : 'pause'));
      play.title = label;
      play.setAttribute('aria-label', label);
      play.setAttribute('aria-pressed', String(timer != null));
    }
    function stop() {
      clearInterval(timer);
      timer = null;
      showPlay();
    }
    function step(by) {
      const v = data.visits[data.visits.indexOf(state.visit) + by];
      if (v != null) setVisit(v);
    }
    function setVisit(v, fromPlay = false) {
      if (!fromPlay) stop();
      if (!data.visits.includes(v) || (fromPlay && v === state.visit)) return;
      state.visit = v;
      changed(fromPlay, !fromPlay);
    }
    function selectGroup(name) {
      if (!selectablePopulations.includes(name) || state.population === name) return;
      state.population = name;
      changed();
    }
    function requestPlot() {
      if (!state.featureId || !state.population || !state.visit) return;
      cfg.onPlot?.({ ...state }, mount);
    }
    function setOpen(value) {
      open = !!value;
      root.classList.toggle('at-has-results', open);
      drawer.hidden = !open;
      plotButton.setAttribute('aria-expanded', String(open));
      plotButton.setAttribute('aria-controls', mount.id);
    }
    function openPlot() {
      if (plotButton.disabled) return;
      stop();
      setOpen(true);
      requestPlot();
      close.focus({ preventScroll: true });
    }
    function closePlot(user = true) {
      if (!open) return;
      setOpen(false);
      if (user) {
        cfg.onClosePlot?.();
        plotButton.focus({ preventScroll: true });
      }
    }
    function changed(fromPlay = false, visitSelected = false) {
      if (state.featureId) memory.set(state.mode, state.featureId);
      if (!fromPlay) stop();
      if (!state.featureId && open) closePlot();
      const before = paints;
      cfg.onSelect?.({
        ...state,
        fromPlayback: fromPlay,
        playing: fromPlay,
        ...(visitSelected && { visitSelected: true }),
      });
      if (before === paints) paint();
    }
    function ensureState() {
      if (cfg.selectedAtlasArm !== undefined) {
        atlasArm = data.arms.includes(cfg.selectedAtlasArm) ? cfg.selectedAtlasArm : null;
      }
      state.mode = modes[cfg.selectedMode] ? cfg.selectedMode : state.mode;
      state.population = selectablePopulations.includes(cfg.selectedPopulation)
        ? cfg.selectedPopulation
        : (state.population ?? data.populations[0]);
      state.visit = data.visits.includes(cfg.selectedVisit)
        ? cfg.selectedVisit
        : (state.visit ?? data.visits[0]);
      const fs = featureList(cfg.study, state.mode);
      state.featureId =
        fs.find((f) => f.id === cfg.selectedFeature)?.id ??
        fs.find((f) => f.id === state.featureId)?.id ??
        fs[0]?.id ??
        null;
      if (state.featureId) memory.set(state.mode, state.featureId);
    }
    let lastSetKey = null;
    function setInfo(f) {
      geneSet.hidden = state.mode !== 'pathways' || !f;
      if (geneSet.hidden) {
        geneSet.open = false;
        lastSetKey = null;
        return;
      }
      const members = Array.isArray(f.members) ? f.members : [];
      const key = JSON.stringify([f.id, members, f.membershipSource?.name]);
      if (key === lastSetKey) return;
      lastSetKey = key;
      geneSet.open = false;
      const title = el('strong', '', f.label || f.id);
      const info = el(
        'p',
        '',
        members.length ? members.length + ' gene-set members' : 'No gene-set definition linked.',
      );
      setPanel.replaceChildren(title, info);
      if (members.length) {
        const list = el('div', 'at-members');
        for (const gene of members) list.append(el('span', '', gene));
        setPanel.append(list);
        if (f.membershipSource?.name) setPanel.append(el('small', '', f.membershipSource.name));
      }
      if (cfg.onLinkGeneSet)
        setPanel.append(
          button(
            members.length ? 'Replace gene set…' : 'Link gene set…',
            null,
            'at-set-action',
            () => {
              geneSet.open = false;
              cfg.onLinkGeneSet(f.id);
            },
          ),
        );
    }
    let traceArm, traceData, traceStudy;
    function fractionHistory() {
      if (traceStudy === data && traceArm === atlasArm) return traceData;
      traceStudy = data;
      traceArm = atlasArm;
      traceData = new Map();
      if (data.visits.length < 2 || data.visits.length > 24 || data.populations.length > 60)
        return traceData;
      for (const visitName of data.visits) {
        for (const g of data.groups({ visit: visitName, arm: atlasArm })) {
          if (!traceData.has(g.name)) traceData.set(g.name, []);
          traceData.get(g.name).push(g.totalCells ? g.fraction : null);
        }
      }
      return traceData;
    }
    function updatePopulations(groups) {
      const traces = fractionHistory();
      const visible = groups.slice(0, groups.length <= 60 ? groups.length : 12);
      const chosen = groups.find((g) => g.name === state.population);
      if (chosen && !visible.includes(chosen)) visible.push(chosen);
      if (!chosen) {
        const pooled = data.pooled({ visit: state.visit, arm: atlasArm });
        if (pooled?.name === state.population) visible.unshift(pooled);
      }
      const shown = new Set(visible.map((g) => g.name));
      for (const [name, row] of popRows) {
        if (!shown.has(name)) {
          row.btn.remove();
          popRows.delete(name);
        }
      }
      for (const g of visible) {
        let row = popRows.get(g.name);
        if (!row) {
          const btn = button(g.name, null, 'at-pop-row', () => selectGroup(g.name));
          const dot = el('i');
          btn.prepend(dot);
          const value = el('span', 'at-pop-share');
          const trace = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          trace.setAttribute('viewBox', '0 0 54 21');
          trace.setAttribute('aria-hidden', 'true');
          const path = document.createElementNS(trace.namespaceURI, 'path');
          const selectedMark = document.createElementNS(trace.namespaceURI, 'circle');
          selectedMark.setAttribute('r', '1.8');
          trace.append(path, selectedMark);
          btn.append(trace, value);
          popList.append(btn);
          row = { btn, value, trace, path, selectedMark };
          popRows.set(g.name, row);
        }
        row.btn.style.setProperty(
          '--pop-color',
          populationIndex.has(g.name)
            ? 'hsl(' +
                window.SctrialAtlasScene.pigment(g.name, populationIndex.get(g.name)) +
                ' 49% 66%)'
            : '#c4c7c5',
        );
        row.btn.setAttribute('aria-pressed', String(g.name === state.population));
        row.value.textContent =
          g.fraction == null ? 'Pooled' : g.totalCells ? share.format(g.fraction) : '—';
        const values = traces.get(g.name);
        row.btn.title =
          g.name +
          ' · ' +
          counts.format(g.count) +
          ' of ' +
          counts.format(g.totalCells) +
          ' cells' +
          (values ? '. Line shows cell fraction across visits on a shared 0–100% scale.' : '');
        row.btn.setAttribute('aria-description', row.value.textContent + '. ' + row.btn.title);
        row.trace.style.display = values ? '' : 'none';
        if (values) {
          const coords = values.map((v, i) =>
            v == null
              ? null
              : [(1 + (i * 51) / (values.length - 1)).toFixed(2), (19 - 17 * v).toFixed(2)],
          );
          const path = coords
            .map((p, i) => (p ? (i && coords[i - 1] ? 'L' : 'M') + p.join(' ') : ''))
            .join('');
          row.path.setAttribute('d', path);
          const point = coords[data.visits.indexOf(state.visit)];
          row.selectedMark.style.display = point ? '' : 'none';
          if (point) {
            row.selectedMark.setAttribute('cx', point[0]);
            row.selectedMark.setAttribute('cy', point[1]);
          }
        }
      }
      population.box.title = 'Choose from all ' + selectablePopulations.length + ' populations';
    }
    function showComposition(groups) {
      for (const [i, group] of groups.entries()) {
        let segment = segments.get(group.name);
        if (!segment) {
          segment = el('button', 'at-proportion');
          segment.type = 'button';
          segment.addEventListener('click', () => selectGroup(group.name));
          segment.style.setProperty(
            '--segment',
            'hsl(' + window.SctrialAtlasScene.pigment(group.name, i) + ' 47% 62%)',
          );
          proportions.append(segment);
          segments.set(group.name, segment);
        }
        const label =
          group.name +
          ' · ' +
          share.format(group.fraction) +
          ' (' +
          counts.format(group.count) +
          ' of ' +
          counts.format(group.totalCells) +
          ' cells)';
        segment.title = label;
        segment.setAttribute('aria-label', label);
        segment.setAttribute('aria-pressed', String(group.name === state.population));
        segment.style.flexGrow = String(group.fraction);
        segment.hidden = !(group.count > 0);
      }
    }
    function paint() {
      paints++;
      const fs = featureList(cfg.study, state.mode);
      const selected = fs.find((f) => f.id === state.featureId);
      const molecular = state.mode !== 'cells';
      const measure = molecular && !!selected;
      const args = {
        featureId: measure ? selected.id : null,
        visit: state.visit,
        arm: atlasArm,
      };
      const groups = data.groups(args),
        scale = measure ? data.domain({ featureId: selected.id, includePooled: true }) : null;
      const cellDomain = measure ? data.cellDomain({ featureId: selected.id }) : null;
      const colorDomain = cellDomain || scale;
      const so = {
        groups,
        domain: scale,
        cellDomain,
        signalColor: window.SctrialAtlasSignal.color,
        selected: state.population,
        measurement: measure,
        featureLabel: selected?.label || '',
        visit: state.visit,
        onSelect: selectGroup,
      };
      if (scene) scene.update(so);
      else scene = window.SctrialAtlasScene.mount(sceneHost, so);
      signalHost.hidden = !measure;
      if (measure) {
        const pooled = groups.some((g) => g.name === state.population) ? null : data.pooled(args);
        signal.update({
          groups: pooled ? [...groups, pooled] : groups,
          selected: state.population,
          feature: selected,
          visit: state.visit,
          arm: atlasArm,
          mode: state.mode,
          scale,
          cellDomain,
          participantMeans:
            cfg.study.source === 'h5ad' || cfg.study.study?.source === 'Synthetic study',
        });
      }
      const cohort = data.cohort(args);
      totals[0][0].textContent = counts.format(cohort.count);
      totals[1][0].textContent = counts.format(cohort.participants);
      updatePopulations(groups);
      scopeKicker.textContent = [
        state.visit,
        atlasArm || (data.arms.length === 1 ? data.arms[0] : 'All arms'),
      ]
        .filter(Boolean)
        .join('  /  ');
      mapTitle.hidden = measure;
      mapTitle.textContent = molecular ? names[state.mode] : 'Cell populations';
      fillSelect(arm.select, [['', 'All arms'], ...data.arms.map((a) => [a, a])], atlasArm ?? '');
      arm.box.hidden = data.arms.length < 2;
      scope.hidden = arm.box.hidden;
      for (const [key, b] of layerButtons) {
        b.setAttribute('aria-pressed', String(key === state.mode));
        layerCounts.get(key).textContent =
          key === 'cells' ? groups.length : featureList(cfg.study, key).length;
      }
      fillSelect(
        population.select,
        selectablePopulations.map((x) => [x, x]),
        state.population,
      );
      feature.box.hidden = !molecular || !fs.length;
      const label =
        { genes: 'Gene', pathways: 'Pathway', modules: 'Module score' }[state.mode] || 'Outcome';
      feature.select.setAttribute('aria-label', label);
      feature.name.textContent = label;
      fillSelect(
        feature.select,
        fs.map((f) => [f.id, f.label || f.id]),
        state.featureId,
      );
      setInfo(selected);
      empty.hidden = !molecular || fs.length > 0;
      emptyText.textContent =
        {
          genes: 'No gene expression measurements in this study.',
          pathways: 'No pathway scores in this study.',
          modules: 'No module or recorded scores in this study.',
        }[state.mode] || '';
      load.hidden = !cfg.onOpenStudy;
      plotButton.disabled = !state.featureId || !state.population || !state.visit;
      legend.classList.toggle('at-quant', measure);
      legendTitle.textContent = cellDomain
        ? { genes: 'Cell expression', pathways: 'Cell pathway score', modules: 'Cell score' }[
            state.mode
          ]
        : measure
          ? 'Label color · mean'
          : 'Share of observed cells';
      proportions.hidden = measure || !cohort.count;
      if (!proportions.hidden) showComposition(groups);
      note.textContent = cellDomain
        ? (selected.unit || '').replace(/^mean /i, '')
        : measure
          ? 'Cells colored by type'
          : '';
      note.hidden = !note.textContent;
      note.classList.toggle('at-type-key', measure && !cellDomain);
      scaleKey.hidden = !colorDomain;
      scaleKey.title = measure ? 'Fixed scale across cell types, visits and study arms' : '';
      min.textContent = colorDomain ? fmt.format(colorDomain[0]) : '';
      min.classList.toggle(
        'at-zero-swatch',
        !!colorDomain && colorDomain[0] === 0 && colorDomain[1] > 0,
      );
      max.textContent =
        colorDomain && colorDomain[0] !== colorDomain[1] ? fmt.format(colorDomain[1]) : '';
      ramp.classList.toggle('at-constant', !!colorDomain && colorDomain[0] === colorDomain[1]);
      ramp.style.background = colorDomain ? window.SctrialAtlasSignal.gradient(colorDomain) : '';
      missing.hidden =
        !measure ||
        !groups.some(
          (g) =>
            g.count &&
            (cellDomain
              ? (g.cellBins?.counts.reduce((a, b) => a + b, 0) ?? 0) < g.count
              : !Number.isFinite(g.value)),
        );
      fillSelect(
        visit.select,
        data.visits.map((x) => [x, x]),
        state.visit,
      );
      const i = data.visits.indexOf(state.visit);
      back.disabled = i <= 0;
      next.disabled = i >= data.visits.length - 1;
      play.disabled = data.visits.length < 2;
      for (const [v, b] of tickButtons) {
        b.setAttribute('aria-pressed', String(v === state.visit));
        const visitCount = data.cohort({ visit: v, arm: atlasArm }).participants;
        b.querySelector('.at-tick-count').textContent =
          counts.format(visitCount) + (visitCount === 1 ? ' participant' : ' participants');
        b.title = v + ' · ' + counts.format(visitCount) + ' participants with captured cells';
      }
      const current = tickButtons.get(state.visit);
      if (current && track.clientWidth) {
        const b = current.getBoundingClientRect(),
          t = track.getBoundingClientRect();
        if (b.left < t.left || b.right > t.right)
          track.scrollLeft += b.left - t.left - (t.width - b.width) / 2;
      }
      plotOutcome.textContent = !molecular && selected ? selected.label || selected.id : '';
      plotOutcome.hidden = !plotOutcome.textContent;
      plotContrast.textContent = cfg.analysisContext || '';
      context.hidden = !context.textContent || !state.featureId;
      plotButton.title = ['Plot results', selected?.label, cfg.analysisContext]
        .filter(Boolean)
        .join(' · ');
      drawerContext.textContent = cfg.resultContext || cfg.analysisContext || '';
      drawerContext.hidden = !drawerContext.textContent || !cfg.plotOpen;
      drawerTitle.textContent = cfg.resultTitle || 'Results';
    }
    function update(opts) {
      if (destroyed) return;
      cfg = opts;
      if (
        cfg.study.visits.some((v, i) => data.visits[i] !== v) ||
        cfg.study.visits.length !== data.visits.length
      ) {
        stop();
        data = window.SctrialAtlasData.prepare(cfg.study);
        setTicks();
      }
      ensureState();
      if (typeof cfg.plotOpen === 'boolean') setOpen(cfg.plotOpen);
      paint();
    }
    function destroy() {
      if (destroyed) return;
      destroyed = true;
      stop();
      abort.abort();
      scene?.destroy();
      signal.destroy();
      instances.delete(host);
      root.remove();
    }
    root.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && root.classList.contains('at-catalog-open')) {
        e.stopPropagation();
        showControls(false);
      } else if (
        e.key === 'Tab' &&
        root.classList.contains('at-catalog-open') &&
        controls.contains(e.target)
      ) {
        const focusables = Array.from(controls.querySelectorAll('button, select')).filter(
          (n) => n.offsetWidth > 0 && !n.disabled,
        );
        if (e.shiftKey && e.target === focusables[0]) {
          e.preventDefault();
          focusables.at(-1)?.focus();
        } else if (!e.shiftKey && e.target === focusables.at(-1)) {
          e.preventDefault();
          focusables[0]?.focus();
        }
      } else if (e.key === 'Escape' && geneSet.open) {
        e.stopPropagation();
        geneSet.open = false;
        summary.focus();
      } else if (e.key === 'Escape' && open && drawer.contains(e.target)) {
        e.stopPropagation();
        closePlot();
      }
    });
    document.addEventListener(
      'visibilitychange',
      () => {
        if (document.hidden) stop();
      },
      { signal: abort.signal },
    );
    document.addEventListener(
      'pointerdown',
      (e) => {
        if (geneSet.open && !geneSet.contains(e.target)) geneSet.open = false;
      },
      { signal: abort.signal },
    );
    setOpen(!!cfg.plotOpen);
    update(cfg);
    return { study: cfg.study, update, closePlot, openPlot, destroy };
  }
  function render(host, cfg) {
    let c = instances.get(host);
    if (c && c.study !== cfg.study) {
      c.destroy();
      c = null;
    }
    if (c) c.update(cfg);
    else {
      c = create(host, cfg);
      instances.set(host, c);
    }
    return c;
  }
  window.SctrialAtlas = {
    render,
    destroy: (host) => instances.get(host)?.destroy(),
  };
})();
