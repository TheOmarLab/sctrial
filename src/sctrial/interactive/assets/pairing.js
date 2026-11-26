(() => {
  'use strict';
  const node = (tag, cls, label) => {
    const element = document.createElement(tag);
    if (cls) element.className = cls;
    if (label != null) element.textContent = label;
    return element;
  };

  function drawGroup(group, visits, paired, onSelect) {
    const block = node('div', 'pm-group');
    block.dataset.arm = String(group.colorIndex);
    const heading = node('div', 'pm-group-head');
    const count = group.rows.filter((r) => r.contributes).length;
    heading.append(
      node('strong', '', group.arm),
      node('small', '', count + ' / ' + group.rows.length + (paired ? ' paired' : ' measured')),
    );
    block.append(heading);
    const grid = node('div', 'pm-grid');
    const labels = node('div', 'pm-visits');
    for (const visit of visits) {
      const item = node('span', '', visit);
      item.title = visit;
      labels.append(item);
    }
    grid.append(labels);
    const columns = node('div', 'pm-columns');
    const visible = group.rows.slice(0, 12);
    columns.style.gridTemplateColumns =
      'repeat(' + Math.max(visible.length, 1) + ', minmax(0, 1fr))';
    for (const row of visible) {
      const values = paired ? [row.baselineValue, row.followupValue] : [row.followupValue];
      const description =
        row.id +
        ' · ' +
        visits
          .map((v, i) => v + ' ' + (Number.isFinite(values[i]) ? 'measured' : 'missing'))
          .join(', ');
      const item = node('button', 'pm-sample');
      item.type = 'button';
      item.title = description;
      item.setAttribute('aria-label', 'Open sampling record for ' + row.id + '. ' + description);
      item.classList.toggle('pm-complete', paired && values.every(Number.isFinite));
      for (const value of values)
        item.append(node('i', Number.isFinite(value) ? 'pm-observed' : 'pm-absent'));
      item.addEventListener('click', () => onSelect(row.id));
      columns.append(item);
    }
    grid.append(columns);
    block.append(grid);
    if (visible.length < group.rows.length)
      block.append(
        node(
          'small',
          'pm-limit',
          'Showing ' + visible.length + ' of ' + group.rows.length + ' participants',
        ),
      );
    return block;
  }

  function render(host, { records, arms, studyArms, baseline, visit, longitudinal, onSelect }) {
    if (!host) return;
    const section = node('section', 'pair-map');
    section.setAttribute(
      'aria-label',
      longitudinal ? 'Participant pairing map' : 'Participant sampling map',
    );
    const groups = arms.map((arm) => ({
      arm,
      colorIndex: studyArms.indexOf(arm),
      rows: records.filter((row) => row.arm === arm),
    }));
    const visits = longitudinal ? [baseline, visit] : [visit];
    const head = node('div', 'pm-head');
    const title = node('h3', '', longitudinal ? 'Sample pairing' : 'Sample coverage');
    head.append(title);
    const graph = node('div', 'pm-graph');
    graph.id = 'samplingGraph';
    const expander = node('button', 'pm-expand');
    expander.type = 'button';
    expander.setAttribute('aria-controls', graph.id);
    expander.setAttribute('aria-label', 'Show sampling map');
    expander.setAttribute('aria-expanded', 'false');
    expander.addEventListener('click', () => {
      const open = !section.classList.contains('pm-expanded');
      section.classList.toggle('pm-expanded', open);
      expander.setAttribute('aria-expanded', String(open));
      expander.setAttribute('aria-label', (open ? 'Hide' : 'Show') + ' sampling map');
    });
    head.append(expander);
    graph.append(node('div', 'pm-caption', 'One column per participant'));
    for (const group of groups) graph.append(drawGroup(group, visits, longitudinal, onSelect));
    const key = node('div', 'pm-key');
    key.append(
      node('i', 'pm-observed'),
      node('span', '', 'Measured'),
      node('i', 'pm-absent'),
      node('span', '', 'Missing'),
    );
    graph.append(key);
    section.append(head, graph);
    host.replaceChildren(section);
  }

  window.SctrialPairMap = { render };
})();
