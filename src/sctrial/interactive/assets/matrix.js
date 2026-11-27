(function () {
  'use strict';

  const pageSize = 40;
  const states = new WeakMap();
  const compareIds = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
  const compactNumber = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 });
  const symbols = {
    search: 'M19.5 19.5l-4.4-4.4m2.2-5a7.2 7.2 0 1 1-14.4 0 7.2 7.2 0 0 1 14.4 0Z',
    left: 'm14 6-6 6 6 6',
    right: 'm9 6 6 6-6 6',
    asc: 'M12 19V5m-5 5 5-5 5 5',
    desc: 'M12 5v14m-5-5 5 5 5-5',
    sort: 'm6 10 3-3 3 3m0 4 3 3 3-3',
    check: 'm5 12 4 4 10-10',
  };

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.7');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS(svg.namespaceURI, 'path');
    path.setAttribute('d', symbols[name]);
    svg.append(path);
    return svg;
  }

  function button(className, label) {
    const element = node('button', className, label);
    element.type = 'button';
    return element;
  }

  function valueLabel(value, signed = false) {
    if (!Number.isFinite(value)) return '—';
    if (value === 0) return '0';
    const magnitude = Math.abs(value);
    const display =
      magnitude < 0.0001 || magnitude >= 1000000
        ? magnitude.toExponential(2)
        : compactNumber.format(magnitude);
    return (value < 0 ? '−' : signed ? '+' : '') + display;
  }

  function normalize(record, config) {
    const baseline = Number.isFinite(record.baselineValue) ? record.baselineValue : null;
    const followup = Number.isFinite(record.followupValue) ? record.followupValue : null;
    const missing = [];
    if (config.longitudinal && baseline === null) missing.push(config.baseline);
    if (followup === null) missing.push(config.visit);
    const included = record.contributes === true && !missing.length;
    const change =
      config.longitudinal && baseline !== null && followup !== null ? followup - baseline : null;
    const reason = missing.length
      ? 'Missing ' + missing.join(' and ')
      : String(record.reason || 'Outside the analysis set');
    return {
      id: String(record.id),
      arm: String(record.arm ?? ''),
      baseline,
      followup,
      delta: Number.isFinite(change) ? change : null,
      included,
      reason,
    };
  }

  function numberCell(value) {
    const cell = node('td', 'mx-number', valueLabel(value));
    if (value === null) {
      cell.classList.add('mx-missing');
      cell.setAttribute('aria-label', 'Missing measurement');
      cell.title = 'Missing measurement';
    } else {
      cell.title = String(value);
    }
    return cell;
  }

  function changeCell(value, maxDelta) {
    const cell = numberCell(value);
    cell.classList.add('mx-delta');
    if (value === null) {
      cell.title = 'A change requires both measurements';
      return cell;
    }
    const label = node('span', 'mx-delta-value', valueLabel(value, true));
    const track = node('span', 'mx-delta-track');
    track.setAttribute('aria-hidden', 'true');
    track.title = 'Change on a common scale, centered at zero';
    if (value !== 0 && maxDelta > 0) {
      const fill = node('i', value > 0 ? 'mx-positive' : 'mx-negative');
      fill.style.width = (50 * (Math.abs(value) / maxDelta)).toFixed(4) + '%';
      track.append(fill);
    }
    cell.replaceChildren(label, track);
    return cell;
  }

  function render(container, config) {
    if (!container || !config) return;
    const scope = JSON.stringify([
      config.baseline,
      config.visit,
      config.feature?.id ?? config.feature,
      config.population ?? '',
      config.armLabels,
      Boolean(config.longitudinal),
    ]);
    const previous = states.get(container);
    const state =
      previous?.scope === scope
        ? previous
        : {
            scope,
            query: '',
            filter: 'all',
            sort: 'id',
            descending: false,
            page: 0,
          };
    states.set(container, state);
    const records = (config.records || []).map((record) => normalize(record, config));
    const counts = new Map();
    let includedCount = 0;
    let maxDelta = 0;
    records.forEach((row) => {
      if (!counts.has(row.arm)) counts.set(row.arm, { total: 0, included: 0 });
      const count = counts.get(row.arm);
      count.total++;
      if (row.included) {
        count.included++;
        includedCount++;
      }
      if (row.delta !== null) maxDelta = Math.max(maxDelta, Math.abs(row.delta));
    });
    const arms = [...new Set([...(config.armLabels || []).map(String), ...counts.keys()])];
    const armOrder = new Map(arms.map((arm, index) => [arm, index]));
    const root = node('section', 'scl-matrix');
    root.setAttribute('aria-label', 'Participant measurements');
    const titleBar = node('div', 'mx-titlebar');
    const heading = node('div', 'mx-heading');
    heading.append(node('h3', '', 'Participant measurements'));
    const feature = typeof config.feature === 'string' ? config.feature : config.feature?.label;
    const subtitle = [feature, config.unit && config.unit !== feature ? config.unit : '']
      .filter(Boolean)
      .join(' · ');
    if (subtitle) heading.append(node('p', '', subtitle));
    const finder = node('label', 'mx-search');
    finder.append(icon('search'));
    const input = node('input', 'mx-input');
    input.type = 'search';
    input.placeholder = 'Find participant';
    input.setAttribute('aria-label', 'Find participant by ID or arm');
    input.value = state.query;
    finder.append(input);
    titleBar.append(heading, finder);
    const filters = node('div', 'mx-filterbar');
    const segments = node('div', 'mx-segments');
    segments.setAttribute('role', 'group');
    segments.setAttribute('aria-label', 'Filter analysis set');
    const tabs = new Map();
    const filterChoices = [
      ['all', 'All', records.length],
      ['included', 'Included', includedCount],
      ['excluded', 'Excluded', records.length - includedCount],
    ];
    filterChoices.forEach(([key, label, total]) => {
      const tab = button('mx-segment');
      tab.append(node('span', '', label), node('span', 'mx-counter', String(total)));
      tab.addEventListener('click', () => {
        state.filter = key;
        state.page = 0;
        refresh();
      });
      tabs.set(key, tab);
      segments.append(tab);
    });
    filters.append(segments);
    const viewport = node('div', 'mx-viewport');
    viewport.tabIndex = 0;
    viewport.setAttribute('role', 'region');
    viewport.setAttribute('aria-label', 'Measurement table, scroll for more rows');
    const table = node('table', 'mx-table');
    const caption = node(
      'caption',
      'mx-sr-only',
      config.longitudinal
        ? 'Measurements by participant. Change is ' +
            config.visit +
            ' minus ' +
            config.baseline +
            '.'
        : 'Measurements by participant at ' + config.visit + '.',
    );
    const head = node('thead');
    const headings = node('tr');
    const columnHeaders = new Map();
    function addColumn(label, sublabel, key) {
      const cell = node('th');
      cell.scope = 'col';
      if (key === 'baseline' || key === 'followup') cell.className = 'mx-numeric-heading';
      if (key) {
        const sort = button('mx-sort');
        const headingText = node('span', 'mx-col-label');
        headingText.append(node('strong', '', label));
        if (sublabel) headingText.append(node('small', '', sublabel));
        const arrow = node('span', 'mx-sort-icon');
        sort.append(headingText, arrow);
        sort.addEventListener('click', () => {
          state.descending = state.sort === key ? !state.descending : key !== 'id';
          state.sort = key;
          state.page = 0;
          refresh();
        });
        sort.setAttribute('aria-label', 'Sort within arm by ' + label);
        cell.append(sort);
        columnHeaders.set(key, { cell, arrow });
      } else {
        cell.append(node('strong', '', label));
      }
      headings.append(cell);
    }
    addColumn('Participant', null, 'id');
    if (config.longitudinal) addColumn(config.baseline, 'Reference visit', 'baseline');
    addColumn(
      config.visit,
      config.longitudinal ? 'Comparison visit' : 'Selected visit',
      'followup',
    );
    if (config.longitudinal) addColumn('Change', 'Comparison − reference', 'delta');
    addColumn('Analysis set');
    head.append(headings);
    const body = node('tbody');
    table.append(caption, head, body);
    viewport.append(table);
    const footer = node('div', 'mx-footer');
    const live = node('span', 'mx-range');
    live.setAttribute('aria-live', 'polite');
    const navigation = node('div', 'mx-pages');
    const lastPage = button('mx-page-btn');
    lastPage.setAttribute('aria-label', 'Previous page of participants');
    lastPage.append(icon('left'));
    const nextPage = button('mx-page-btn');
    nextPage.setAttribute('aria-label', 'Next page of participants');
    nextPage.append(icon('right'));
    lastPage.addEventListener('click', () => {
      state.page--;
      refresh();
    });
    nextPage.addEventListener('click', () => {
      state.page++;
      refresh();
    });
    navigation.append(lastPage, nextPage);
    footer.append(live, navigation);
    root.append(titleBar, filters, viewport, footer);
    container.replaceChildren(root);

    function select(id) {
      for (const row of body.querySelectorAll('tr[data-participant]')) {
        const active = row.dataset.participant === id;
        row.classList.toggle('mx-active', active);
        const control = row.querySelector('.mx-person-btn');
        if (active) control?.setAttribute('aria-current', 'true');
        else control?.removeAttribute('aria-current');
      }
      config.onSelect?.(id);
    }

    function refresh() {
      const query = state.query.trim().toLocaleLowerCase();
      const rows = records.filter(
        (row) =>
          (state.filter === 'all' || (state.filter === 'included') === row.included) &&
          (!query ||
            row.id.toLocaleLowerCase().includes(query) ||
            row.arm.toLocaleLowerCase().includes(query)),
      );
      rows.sort((a, b) => {
        if (a.arm !== b.arm) return armOrder.get(a.arm) - armOrder.get(b.arm);
        const key = state.sort;
        const order = state.descending ? -1 : 1;
        if (key === 'id') return compareIds.compare(a.id, b.id) * order;
        if (a[key] === null && b[key] !== null) return 1;
        if (b[key] === null && a[key] !== null) return -1;
        return (a[key] - b[key]) * order || compareIds.compare(a.id, b.id);
      });
      for (const [key, tab] of tabs) {
        tab.setAttribute('aria-pressed', String(state.filter === key));
      }
      for (const [key, { cell, arrow }] of columnHeaders) {
        const current = state.sort === key;
        if (current) cell.setAttribute('aria-sort', state.descending ? 'descending' : 'ascending');
        else cell.removeAttribute('aria-sort');
        arrow.replaceChildren(icon(current ? (state.descending ? 'desc' : 'asc') : 'sort'));
      }
      state.page = Math.min(
        Math.max(0, Math.ceil(rows.length / pageSize) - 1),
        Math.max(0, state.page),
      );
      const start = state.page * pageSize;
      const page = rows.slice(start, start + pageSize);
      const fragment = document.createDocumentFragment();
      let lastArm;
      page.forEach((record) => {
        if (record.arm !== lastArm) {
          lastArm = record.arm;
          const group = node('tr', 'mx-group');
          const cell = node('td');
          cell.colSpan = headings.children.length;
          const line = node('div', 'mx-group-line');
          const marker = node('i', 'mx-arm-marker');
          marker.setAttribute('aria-hidden', 'true');
          marker.dataset.arm = String((config.studyArms || arms).indexOf(record.arm) % 2);
          const count = counts.get(record.arm);
          const n = node(
            'span',
            'mx-group-count',
            count.included + ' / ' + count.total + ' included',
          );
          n.title = 'Analysis set in this arm before table filters';
          line.append(marker, node('strong', '', record.arm || 'Unassigned arm'), n);
          cell.append(line);
          group.append(cell);
          fragment.append(group);
        }
        const row = node('tr', 'mx-row');
        row.dataset.participant = record.id;
        row.setAttribute('aria-label', record.id + (record.arm ? ', ' + record.arm : ''));
        const active = config.selectedId != null && String(config.selectedId) === record.id;
        row.classList.toggle('mx-active', active);
        const idCell = node('th', 'mx-participant');
        idCell.scope = 'row';
        if (typeof config.onSelect === 'function') {
          const person = button('mx-person-btn', record.id);
          person.setAttribute('aria-label', 'Inspect participant ' + record.id);
          if (active) person.setAttribute('aria-current', 'true');
          idCell.append(person);
          row.addEventListener('click', () => select(record.id));
          row.classList.add('mx-selectable');
        } else {
          idCell.textContent = record.id;
        }
        row.append(idCell);
        if (config.longitudinal) row.append(numberCell(record.baseline));
        row.append(numberCell(record.followup));
        if (config.longitudinal) row.append(changeCell(record.delta, maxDelta));
        const status = node('td', 'mx-status');
        const badge = node('span', record.included ? 'mx-included' : 'mx-excluded');
        if (record.included) badge.append(icon('check'), node('span', '', 'Included'));
        else {
          badge.append(node('i'), node('span', '', record.reason));
          badge.setAttribute('aria-label', 'Excluded: ' + record.reason);
        }
        status.append(badge);
        row.append(status);
        fragment.append(row);
      });
      if (!page.length) {
        const empty = node('tr');
        const cell = node(
          'td',
          'mx-empty',
          records.length
            ? 'No participants match these filters.'
            : 'No participant records are available.',
        );
        cell.colSpan = headings.children.length;
        empty.append(cell);
        fragment.append(empty);
      }
      body.replaceChildren(fragment);
      viewport.scrollTop = 0;
      live.textContent = page.length
        ? start +
          1 +
          '–' +
          (start + page.length) +
          ' of ' +
          rows.length +
          (rows.length === 1 ? ' participant' : ' participants')
        : '0 participants';
      navigation.hidden = rows.length <= pageSize;
      lastPage.disabled = state.page === 0;
      nextPage.disabled = start + pageSize >= rows.length;
    }
    input.addEventListener('input', () => {
      state.query = input.value;
      state.page = 0;
      refresh();
    });
    refresh();
  }

  window.SctrialMatrix = { render };
})();
