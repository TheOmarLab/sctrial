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
      entries.some(
        ([k, v], i) => options[i].value !== k || options[i].text !== v,
      )
    )
      select.replaceChildren(...entries.map(([k, v]) => new Option(v, k)));
    select.value = value ?? '';
  }
  function featureList(study, mode) {
    const all = study.features || [];
    if (mode === 'genes') return all.filter((f) => f.kind === 'gene');
    if (mode === 'pathways') return all.filter((f) => f.kind === 'pathway');
    if (mode === 'modules')
      return all.filter(
        (f) => !f.kind || f.kind === 'module' || f.kind === 'score',
      );
    return all;
  }
  const fmt = new Intl.NumberFormat('en', { maximumSignificantDigits: 4 });
  function create(host, options) {
    let cfg = options;
    let data = window.SctrialAtlasData.prepare(cfg.study);
    const selectablePopulations = cfg.study.populations;
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
    const abort = new AbortController();

    const root = el('section', 'at-workspace');
    root.setAttribute('data-atlas', '');
    root.setAttribute('aria-label', 'Cell atlas');
    const plane = el('div', 'at-plane');
    const head = el('header', 'at-top');
    const nameBlock = el('div', 'at-title');
    nameBlock.append(el('h1', '', 'Cell atlas'));
    const studyLine = el('p', 'at-subtitle');
    nameBlock.append(studyLine);
    const plotButton = button('Plot results', 'plot', 'at-plot', () =>
      openPlot(),
    );
    const plotBlock = el('div', 'at-plot-block');
    const context = el('span', 'at-context');
    plotBlock.append(plotButton, context);
    head.append(nameBlock, plotBlock);
    const tools = el('div', 'at-tools');
    const layers = el('div', 'at-layers');
    layers.setAttribute('role', 'group');
    layers.setAttribute('aria-label', 'Measurement overlay');
    const layerButtons = new Map();
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
      layerButtons.set(key, btn);
      layers.append(btn);
    }
    const population = picker('Cell population', (name) => selectGroup(name));
    const feature = picker('Gene', (id) => {
      state.featureId = id;
      changed();
    });
    const geneSet = el('details', 'at-geneset');
    const summary = el('summary', '', 'Gene set');
    summary.append(icon('down'));
    const setPanel = el('div', 'at-set-panel');
    geneSet.append(summary, setPanel);
    tools.append(layers, population.box, feature.box, geneSet);
    const stage = el('div', 'at-stage');
    const sceneHost = el('div', 'at-scene');
    stage.append(sceneHost);
    const legendRow = el('div', 'at-keyline');
    const legend = el('div', 'at-legend');
    const legendTitle = el('span', 'at-legend-label');
    const min = el('span', 'at-limit');
    const ramp = el('span', 'at-ramp');
    const max = el('span', 'at-limit');
    const missing = el('span', 'at-missing', 'Missing');
    const note = el('span', 'at-schematic', 'Illustrative clusters');
    const scaleKey = el('span', 'at-scale-key');
    scaleKey.append(min, ramp, max);
    legend.append(legendTitle, scaleKey, missing, note);
    const zoom = el('div', 'at-zoom');
    zoom.append(
      button('Zoom out', 'minus', '', () => scene?.zoomBy(1 / 1.2), true),
      button('Zoom in', 'plus', '', () => scene?.zoomBy(1.2), true),
      button('Fit clusters', 'reset', '', () => scene?.reset(), true),
    );
    legendRow.append(legend, zoom);
    const empty = el('div', 'at-empty');
    const emptyText = el('span');
    const load = button('Open study', null, 'at-empty-open', () =>
      cfg.onOpenStudy?.(),
    );
    empty.append(emptyText, load);
    plane.append(head, tools, empty, stage, legendRow);

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
        tickButtons.set(v, tick);
        track.append(tick);
      }
    }
    setTicks();
    const back = button(
      'Previous visit',
      'previous',
      'at-step',
      () => step(-1),
      true,
    );
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
    const close = button(
      'Close results',
      'close',
      'at-close',
      () => closePlot(true),
      true,
    );
    drawerHead.append(drawerLabels, close);
    const mount = el('div', 'at-result');
    mount.id = 'atlasResultMount';
    drawer.append(drawerHead, mount);
    root.append(plane, drawer);
    host.replaceChildren(root);

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
      if (!data.visits.includes(v) || v === state.visit) return;
      state.visit = v;
      changed(fromPlay);
    }
    function selectGroup(name) {
      if (!selectablePopulations.includes(name) || state.population === name)
        return;
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
    function changed(fromPlay = false) {
      if (state.featureId) memory.set(state.mode, state.featureId);
      if (!fromPlay) stop();
      if (!state.featureId && open) closePlot();
      paint();
      cfg.onSelect?.({ ...state, fromPlayback: fromPlay, playing: fromPlay });
      // Selecting or playing visits is an exploration action. The Plot button applies it.
    }
    function ensureState() {
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
        members.length
          ? members.length + ' gene-set members'
          : 'No gene-set definition linked.',
      );
      setPanel.replaceChildren(title, info);
      if (members.length) {
        const list = el('div', 'at-members');
        for (const gene of members) list.append(el('span', '', gene));
        setPanel.append(list);
        if (f.membershipSource?.name)
          setPanel.append(el('small', '', f.membershipSource.name));
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
    function paint() {
      const fs = featureList(cfg.study, state.mode);
      const selected = fs.find((f) => f.id === state.featureId);
      const molecular = state.mode !== 'cells';
      const measure = molecular && !!selected;
      const args = {
        featureId: measure ? selected.id : null,
        visit: state.visit,
        arm: null,
      };
      const groups = data.groups(args),
        scale = measure ? data.domain(args) : null;
      const displayGroups = groups.map((g) => ({
        ...g,
        level:
          scale && Number.isFinite(g.value)
            ? scale[0] === scale[1]
              ? 0.5
              : Math.max(
                  0,
                  Math.min(1, (g.value - scale[0]) / (scale[1] - scale[0])),
                )
            : null,
      }));
      const so = {
        groups: displayGroups,
        selected: state.population,
        measurement: measure,
        featureLabel: selected?.label || '',
        visit: state.visit,
        onSelect: selectGroup,
      };
      if (scene) scene.update(so);
      else scene = window.SctrialAtlasScene.mount(sceneHost, so);
      const nCells = groups.reduce((sum, g) => sum + g.count, 0);
      studyLine.textContent =
        groups.length +
        (groups.length === 1 ? ' population' : ' populations') +
        '  ·  ' +
        nCells.toLocaleString('en') +
        ' observed cells';
      for (const [key, b] of layerButtons)
        b.setAttribute('aria-pressed', String(key === state.mode));
      fillSelect(
        population.select,
        selectablePopulations.map((x) => [x, x]),
        state.population,
      );
      feature.box.hidden = !molecular || !fs.length;
      const label =
        { genes: 'Gene', pathways: 'Pathway', modules: 'Module score' }[
          state.mode
        ] || 'Outcome';
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
      plotButton.disabled =
        !state.featureId || !state.population || !state.visit;
      const mixedArms = data.arms.length > 1;
      legend.classList.toggle('at-quant', measure);
      legendTitle.textContent = measure
        ? 'Participant mean' +
          (mixedArms ? ' · all arms' : '') +
          (selected.unit ? ' · ' + selected.unit : '')
        : 'Illustrative clusters';
      note.hidden = !measure;
      scaleKey.hidden = !scale;
      min.textContent = scale ? fmt.format(scale[0]) : '';
      max.textContent =
        scale && scale[0] !== scale[1] ? fmt.format(scale[1]) : '';
      ramp.classList.toggle('at-constant', !!scale && scale[0] === scale[1]);
      missing.hidden =
        !measure || !groups.some((g) => !Number.isFinite(g.value));
      fillSelect(
        visit.select,
        data.visits.map((x) => [x, x]),
        state.visit,
      );
      const i = data.visits.indexOf(state.visit);
      back.disabled = i <= 0;
      next.disabled = i >= data.visits.length - 1;
      play.disabled = data.visits.length < 2;
      for (const [v, b] of tickButtons)
        b.setAttribute('aria-pressed', String(v === state.visit));
      const current = tickButtons.get(state.visit);
      if (current && track.clientWidth) {
        const b = current.getBoundingClientRect(),
          t = track.getBoundingClientRect();
        if (b.left < t.left || b.right > t.right)
          track.scrollLeft += b.left - t.left - (t.width - b.width) / 2;
      }
      context.textContent = cfg.analysisContext || '';
      context.hidden = !context.textContent || !state.featureId;
      plotButton.title = cfg.analysisContext
        ? 'Plot results · ' + cfg.analysisContext
        : 'Plot results';
      drawerContext.textContent = cfg.analysisContext || '';
      drawerContext.hidden =
        !drawerContext.textContent ||
        !cfg.plotOpen ||
        (cfg.resultTitle && cfg.resultTitle !== 'Participant results');
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
      instances.delete(host);
      root.remove();
    }
    root.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && geneSet.open) {
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
