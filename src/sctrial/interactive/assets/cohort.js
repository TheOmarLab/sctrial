(() => {
  'use strict';

  const recordsPerPage = 12;
  const previousViews = new WeakMap();
  const counts = new Intl.NumberFormat('en-US');
  const scores = new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 });
  let nextId = 0;

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function makeButton(label, className, handler) {
    const button = node('button', className, label);
    button.type = 'button';
    button.addEventListener('click', handler);
    return button;
  }

  function makeSelect(name, items, initial, change) {
    const field = node('label', 'pe-control');
    field.append(node('span', 'pe-label', name));
    const select = node('select', 'select');
    select.setAttribute('aria-label', name);
    for (const [value, label] of items) select.add(new Option(label, value));
    select.value = initial;
    select.addEventListener('change', () => change(select.value));
    field.append(select);
    return field;
  }

  function indexVisits(observations, population) {
    const index = new Map();
    for (const obs of observations) {
      let visits = index.get(obs.id);
      if (!visits) {
        visits = new Map();
        index.set(obs.id, visits);
      }
      // A specimen may contain other populations but no observations in the selected one.
      if (!visits.has(obs.visit)) visits.set(obs.visit, null);
      if (obs.population === population && obs.cells > 0) visits.set(obs.visit, obs);
    }
    return index;
  }

  function status(entry, featureId) {
    if (entry === undefined) return ['no', 'No specimen recorded'];
    if (entry === null) return ['partial', 'Population unobserved'];
    if (!Number.isFinite(entry.scores[featureId])) return ['unmeasured', 'Score unmeasured'];
    return ['yes', 'Score measured'];
  }

  function visitMark(kind, label) {
    const mark = node('span', 'pe-mark pe-' + kind);
    mark.setAttribute('role', 'img');
    mark.setAttribute('aria-label', label);
    mark.title = label;
    return mark;
  }

  function pageControls(page, pageCount, onPage) {
    const pager = node('div', 'pe-pages');
    const prev = makeButton('Previous', 'pe-step', () => onPage(page - 1));
    const next = makeButton('Next', 'pe-step', () => onPage(page + 1));
    prev.disabled = page === 0;
    next.disabled = page >= pageCount - 1;
    pager.append(prev, node('span', 'pe-page-number', page + 1 + ' / ' + pageCount), next);
    return pager;
  }

  function renderPeople(container, model, actions = {}) {
    const old = previousViews.get(container);
    const state =
      old && old.study === model.participants
        ? old
        : {
            study: model.participants,
            arm: '',
            coverage: '',
            query: '',
            page: 0,
          };
    const prefix = 'pe-' + ++nextId;
    const { participants, arms, visits, population, feature } = model;
    const selected = [...new Set(model.selectedVisits)];
    const analysisArms = new Set(model.analysisArms);
    const index = indexVisits(model.observations, population);
    const shownVisits = visits.length <= 5 ? visits : selected;
    const unused = new Map();
    if (model.filterArm !== undefined && model.filterArm !== state.requestedArm) {
      state.arm = model.filterArm;
      state.requestedArm = model.filterArm;
    }
    if (model.query !== undefined && model.query !== state.requestedQuery) {
      state.query = model.query;
      state.requestedQuery = model.query;
    }
    if (!arms.includes(state.arm)) state.arm = '';
    const rows = participants.map((person) => {
      const visitsByPerson = index.get(person.id) || unused;
      const observed = selected.map((v) => visitsByPerson.get(v));
      const inArm = analysisArms.has(person.arm);
      const covered = inArm && selected.length > 0 && observed.every((entry) => Number.isFinite(entry?.scores[feature.id]));
      const cells = observed.reduce((sum, obs) => sum + (obs?.cells || 0), 0);
      return { person, visitsByPerson, inArm, covered, cells };
    });
    previousViews.set(container, state);

    const root = node('section', 'pe card');
    root.setAttribute('aria-labelledby', prefix + '-title');
    const head = node('div', 'pe-head');
    const title = node('h2', '', 'Visit coverage');
    title.id = prefix + '-title';
    head.append(
      title,
      node('p', 'pe-sub', [feature.label, population, selected.join(' and ')].filter(Boolean).join(' · ')),
    );
    const filters = node('div', 'pe-filters');
    const searchField = node('label', 'pe-control pe-search');
    searchField.append(node('span', 'pe-label', 'Participant ID'));
    const input = node('input', 'pe-input');
    input.type = 'search';
    input.placeholder = 'Search participants';
    input.value = state.query;
    searchField.append(input);
    const groupSelect = makeSelect(
      'Group',
      [['', 'All groups'], ...arms.map((a) => [a, a])],
      state.arm,
      (value) => {
        state.arm = value;
        state.page = 0;
        update();
      },
    );
    const coverageSelect = makeSelect(
      'Coverage',
      [
        ['', 'All participants'],
        ['included', 'Contributing'],
        ['missing', 'Missing required score'],
        ['outside', 'Outside selected groups'],
      ],
      state.coverage,
      (value) => {
        state.coverage = value;
        state.page = 0;
        update();
      },
    );
    filters.append(searchField, groupSelect, coverageSelect);
    const scroller = node('div', 'pe-scroll');
    const table = node('table', 'pe-table');
    table.setAttribute('aria-label', 'Participant visit coverage');
    const tableHead = table.createTHead().insertRow();
    ['Participant', 'Group', ...shownVisits, 'Cells¹', 'Contribution'].forEach((header) => {
      const th = node('th', '', header);
      th.scope = 'col';
      tableHead.append(th);
    });
    const body = table.createTBody();
    scroller.append(table);
    const footer = node('div', 'pe-footer'),
      range = node('p', 'pe-range');
    range.setAttribute('aria-live', 'polite');
    const pager = node('div');
    footer.append(range, pager);
    const legend = node('div', 'pe-legend');
    for (const [kind, label] of [
      ['yes', 'Score measured'],
      ['unmeasured', 'Score unmeasured'],
      ['partial', 'Population unobserved'],
      ['no', 'Specimen unrecorded'],
    ]) {
      const k = node('span', 'pe-key');
      const mark = visitMark(kind, label);
      mark.setAttribute('aria-hidden', 'true');
      k.append(mark, node('span', '', label));
      legend.append(k);
    }
    const note = node(
      'p',
      'pe-note',
      '¹ Cells in the selected population and visits. Each contributing participant has a measured score at every visit in the comparison.',
    );
    const extras = node('div', 'pe-notes');
    extras.append(legend, note);
    root.append(head, filters, scroller, footer, extras);
    container.replaceChildren(root);

    function update() {
      const needle = state.query.trim().toLocaleLowerCase();
      const visible = rows.filter(
        (r) =>
          (!needle || String(r.person.id).toLocaleLowerCase().includes(needle)) &&
          (!state.arm || r.person.arm === state.arm) &&
          (!state.coverage ||
            (state.coverage === 'included' && r.covered) ||
            (state.coverage === 'missing' && r.inArm && !r.covered) ||
            (state.coverage === 'outside' && !r.inArm)),
      );
      const pages = Math.max(1, Math.ceil(visible.length / recordsPerPage));
      state.page = Math.max(0, Math.min(state.page, pages - 1));
      const first = state.page * recordsPerPage;
      body.replaceChildren();
      visible.slice(first, first + recordsPerPage).forEach((record) => {
        const tr = body.insertRow();
        const id = node('th');
        id.scope = 'row';
        const open = makeButton(String(record.person.id), 'pe-person', () => openParticipant(record));
        open.setAttribute('aria-label', 'Inspect participant ' + record.person.id);
        id.append(open);
        tr.append(id);
        const group = tr.insertCell();
        const badge = node('span', 'pe-group', record.person.arm);
        badge.classList.toggle('pe-second', arms.indexOf(record.person.arm) === 1);
        group.append(badge);
        shownVisits.forEach((v) => tr.insertCell().append(visitMark(...status(record.visitsByPerson.get(v), feature.id))));
        tr.insertCell().append(node('span', 'pe-num', counts.format(record.cells)));
        const label = record.covered ? 'Contributing' : record.inArm ? 'Missing score' : 'Outside groups';
        tr.insertCell().append(
          node(
            'span',
            'pe-status pe-' + (record.covered ? 'covered' : record.inArm ? 'incomplete' : 'outside'),
            label,
          ),
        );
      });
      if (!visible.length) {
        const td = body.insertRow().insertCell();
        td.colSpan = shownVisits.length + 4;
        td.className = 'pe-empty';
        td.textContent = 'No participants match these filters.';
      }
      const last = Math.min(first + recordsPerPage, visible.length);
      range.textContent = visible.length
        ? first + 1 + '–' + last + ' of ' + counts.format(visible.length) + ' participants'
        : '0 participants';
      pager.replaceChildren();
      if (pages > 1) {
        const move = (page) => {
          state.page = page;
          update();
          const preferred = page > 0 ? 'button:first-child' : 'button:last-child';
          pager.querySelector(preferred)?.focus();
        };
        pager.append(pageControls(state.page, pages, move));
      }
    }

    function openParticipant(record) {
      const dialog = node('dialog', 'pe-detail');
      const title = node('h2', '', String(record.person.id));
      title.id = prefix + '-detail-title';
      dialog.setAttribute('aria-labelledby', title.id);
      const heading = node('div', 'pe-dialog-head');
      const titleBlock = node('div');
      titleBlock.append(node('p', 'pe-dialog-label', 'Participant'), title);
      const close = makeButton('Close', 'pe-close', () => dialog.close());
      heading.append(titleBlock, close);
      const context = node('p', 'pe-dialog-sub', record.person.arm + ' · ' + population);
      const panel = node('div', 'pe-scroll');
      const detail = node('table', 'pe-table pe-detail-table');
      const thead = detail.createTHead().insertRow();
      for (const h of ['Visit', 'Status', 'Cells', feature.label]) {
        const th = node('th', '', h);
        th.scope = 'col';
        thead.append(th);
      }
      const detailBody = detail.createTBody();
      panel.append(detail);
      const nav = node('div', 'pe-detail-nav');
      const totalPages = Math.max(1, Math.ceil(visits.length / recordsPerPage));
      function detailPage(page) {
        detailBody.replaceChildren();
        visits.slice(page * recordsPerPage, (page + 1) * recordsPerPage).forEach((v) => {
          const obs = record.visitsByPerson.get(v);
          const tr = detailBody.insertRow();
          const th = node('th', '', v);
          th.scope = 'row';
          tr.append(th);
          tr.insertCell().textContent = status(obs, feature.id)[1];
          tr.insertCell().textContent = obs ? counts.format(obs.cells) : '—';
          const val = obs?.scores?.[feature.id];
          tr.insertCell().textContent = Number.isFinite(val) ? scores.format(val) : '—';
        });
        nav.replaceChildren();
        if (totalPages > 1) nav.append(pageControls(page, totalPages, detailPage));
      }
      detailPage(0);
      dialog.append(heading, context, panel, nav);
      root.append(dialog);
      dialog.addEventListener('close', () => dialog.remove(), { once: true });
      dialog.showModal();
      close.focus();
      if (typeof actions.onParticipant === 'function') actions.onParticipant(record.person.id);
    }

    input.addEventListener('input', () => {
      state.query = input.value;
      state.page = 0;
      update();
    });
    update();
    if (model.initialId !== undefined) {
      const initial = rows.find((record) => record.person.id === model.initialId);
      if (initial) openParticipant(initial);
    }
  }

  globalThis.renderPeople = renderPeople;
})();
