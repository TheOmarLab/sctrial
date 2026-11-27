(() => {
  'use strict';

  const sorting = new WeakMap();
  const formatDecimal = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  const signed = (n) =>
    Number.isFinite(n) ? (n < 0 ? '−' : n > 0 ? '+' : '') + formatDecimal.format(Math.abs(n)) : '—';
  const bound = (n) => (n === Infinity ? '∞' : n === -Infinity ? '−∞' : signed(n));
  const interval = (row) =>
    row.lo === null ||
    row.lo === undefined ||
    Number.isNaN(row.lo) ||
    row.hi === null ||
    row.hi === undefined ||
    Number.isNaN(row.hi)
      ? '—'
      : bound(row.lo) + ' to ' + bound(row.hi);
  function probability(n) {
    if (!Number.isFinite(n)) return '—';
    if (n === 0) return '0';
    return n < 0.001 ? n.toExponential(1).replace('e-', 'e−') : n.toFixed(3);
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderResults(container, model, actions) {
    const isMean = !model.reference && Boolean(model.isMean);
    const currentFocus = container.contains(document.activeElement)
      ? document.activeElement.dataset.resultFocus
      : undefined;
    const sort = sorting.get(container) || { key: '', ascending: true };
    const source = Array.isArray(model.rows) ? model.rows : [];
    const selected = source.find((row) => row.id === model.selected) || source[0];
    const rows = [...source];
    const available = model.reference
      ? ['label', 'estimate', 'p', 'q', 'n']
      : ['label', 'estimate', 'n'];
    if (available.includes(sort.key)) {
      rows.sort((a, b) => {
        if (sort.key === 'label') return a.label.localeCompare(b.label) * (sort.ascending ? 1 : -1);
        const x = a[sort.key],
          y = b[sort.key];
        if (!Number.isFinite(x)) return Number.isFinite(y) ? 1 : 0;
        if (!Number.isFinite(y)) return -1;
        return (x - y) * (sort.ascending ? 1 : -1);
      });
    }
    const layout = el('div', 'results-layout');
    const card = el('section', 'card results-table-card');
    const heading = el('div', 'section-head result-head');
    const left = el('div');
    left.append(
      el(
        'h2',
        '',
        model.reference
          ? 'Feature estimates'
          : isMean
            ? 'Participant means'
            : 'Descriptive contrasts',
      ),
    );
    heading.append(left);
    const exportButton = el('button', 'btn small', 'Export CSV');
    exportButton.type = 'button';
    exportButton.dataset.resultFocus = 'export';
    exportButton.disabled = source.length === 0;
    exportButton.addEventListener('click', actions.onExport);
    heading.append(exportButton);
    card.append(heading);

    const scroller = el('div', 'results-scroll');
    const table = el('table', 'results-table');
    table.setAttribute(
      'aria-label',
      model.reference
        ? 'Feature estimates and uncertainty'
        : isMean
          ? 'Participant means by feature'
          : 'Descriptive participant-level contrasts',
    );
    const header = table.createTHead().insertRow();
    function createColumn(key, name) {
      const th = el('th');
      th.scope = 'col';
      if (!available.includes(key)) {
        th.textContent = name;
        header.append(th);
        return;
      }
      const sorted = sort.key === key;
      if (sorted) th.setAttribute('aria-sort', sort.ascending ? 'ascending' : 'descending');
      const button = el('button', 'sort-head');
      button.type = 'button';
      button.title = 'Sort by ' + name;
      button.dataset.resultFocus = 'sort-' + key;
      const glyph = el('span', 'sort-arrow', sorted ? (sort.ascending ? '↑' : '↓') : '↕');
      glyph.setAttribute('aria-hidden', 'true');
      button.append(document.createTextNode(name), glyph);
      button.addEventListener('click', () => {
        sorting.set(container, {
          key,
          ascending: sorted ? !sort.ascending : key !== 'n' && key !== 'estimate',
        });
        renderResults(container, model, actions);
      });
      th.append(button);
      header.append(th);
    }
    createColumn('label', 'Feature');
    createColumn('estimate', model.reference ? 'Estimate' : isMean ? 'Mean' : 'Difference');
    if (model.reference) {
      createColumn('ci', '95% CI');
      createColumn('p', 'p');
      createColumn('q', 'FDR');
    }
    createColumn('n', 'n');
    const body = table.createTBody();
    for (const row of rows) {
      const tr = body.insertRow();
      if (row.id === selected?.id) tr.classList.add('chosen-feature');
      const nameCell = el('th', 'feature-cell');
      nameCell.scope = 'row';
      const featureButton = el('button', 'feature-link', row.label);
      featureButton.type = 'button';
      featureButton.dataset.resultFocus = 'feature-' + row.id;
      featureButton.setAttribute('aria-pressed', String(row.id === selected?.id));
      function selectRow() {
        actions.onSelect(row.id);
      }
      featureButton.addEventListener('click', selectRow);
      tr.addEventListener('click', (event) => {
        if (!featureButton.contains(event.target)) selectRow();
      });
      nameCell.append(featureButton);
      tr.append(nameCell);
      tr.append(el('td', 'result-number', signed(row.estimate)));
      if (model.reference) {
        const adjusted = el('td', 'result-number', probability(row.q));
        if (row.fdrFamily) adjusted.title = row.fdrFamily;
        tr.append(
          el('td', 'result-interval', interval(row)),
          el('td', 'result-number', probability(row.p)),
          adjusted,
        );
      }
      tr.append(el('td', 'result-number', Number.isFinite(row.n) ? String(row.n) : '—'));
    }
    if (!source.length) {
      const td = el('td', 'result-empty', 'No features are available for this comparison.');
      td.colSpan = header.cells.length;
      body.insertRow().append(td);
    }
    scroller.append(table);
    card.append(scroller);
    const foot = el(
      'p',
      'result-foot muted',
      model.reference
        ? 'n = contributing participants. ' +
            (model.fdrDescription || 'FDR is reported across the tested features.')
        : isMean
          ? 'Each participant contributes one summary at the selected visit.'
          : 'Differences are calculated from the imported participant summaries.',
    );
    card.append(foot);
    layout.append(card);

    if (selected) {
      const detail = el('section', 'card result-detail');
      const group = el('div', 'result-detail-top');
      const name = el('div');
      name.append(el('span', 'detail-kicker', 'Selected feature'), el('h2', '', selected.label));
      group.append(name);
      const trace = el('button', 'btn small', 'View trajectory');
      trace.type = 'button';
      trace.dataset.resultFocus = 'trace';
      trace.addEventListener('click', () => actions.onShowOverview(selected.id));
      group.append(trace);
      detail.append(group);
      const facts = el('dl', 'detail-facts');
      function fact(label, value, special) {
        const div = el('div', special ? 'fact feature-estimate' : 'fact');
        div.append(el('dt', '', label), el('dd', '', value));
        facts.append(div);
      }
      fact(
        model.reference ? 'Estimate' : isMean ? 'Participant mean' : 'Descriptive difference',
        signed(selected.estimate),
        true,
      );
      if (model.reference) fact('95% confidence interval', interval(selected));
      fact('Participants', Number.isFinite(selected.n) ? String(selected.n) : '—');
      const selectedUnit = selected.unit || model.unit;
      if (selectedUnit) fact('Units', selectedUnit);
      if (model.reference && selected.fdrFamily) {
        fact('Adjusted p-value', probability(selected.q));
        fact('FDR family', selected.fdrFamily);
      }
      detail.append(facts);
      if (model.contrast) {
        const contrast = el('div', 'selected-contrast');
        contrast.append(
          el('span', '', isMean ? 'Summary' : 'Contrast'),
          el('p', '', model.contrast),
        );
        detail.append(contrast);
      }
      layout.append(detail);
    }
    container.replaceChildren(layout);
    if (currentFocus)
      Array.from(container.querySelectorAll('[data-result-focus]'))
        .find((node) => node.dataset.resultFocus === currentFocus)
        ?.focus({ preventScroll: true });
  }
  globalThis.renderResults = renderResults;
})();
