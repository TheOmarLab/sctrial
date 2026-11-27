(() => {
  'use strict';

  const storageVersion = 1;
  const prefix = 'sctrial.pinned-contrasts.v1.';
  const icons = {
    pin: 'M16 3l5 5-3 1-4 5v3l-2 2-3-5-5-3 2-2h3l5-4zM8 16l-5 5',
    pencil: 'M14 5l5 5M4 20l5-1L21 7l-5-5L4 14z',
    close: 'M7 7l10 10M17 7L7 17',
    open: 'M5 12h13m-5-5 5 5-5 5',
    swap: 'M3 8h16m-4-4 4 4-4 4M21 16H5m4-4-4 4 4 4',
  };

  function el(name, className, text) {
    const node = document.createElement(name);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function shape(name) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    Object.entries({
      viewBox: '0 0 24 24',
      width: '15',
      height: '15',
      fill: 'none',
      stroke: 'currentColor',
      'stroke-width': '1.7',
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
      'aria-hidden': 'true',
    }).forEach(([key, value]) => svg.setAttribute(key, value));
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('d', icons[name]);
    svg.append(path);
    return svg;
  }

  function button(text, style, click, glyph) {
    const node = el('button', 'cbench-button ' + style);
    node.type = 'button';
    if (glyph) node.append(shape(glyph));
    if (text) node.append(document.createTextNode(text));
    node.addEventListener('click', click);
    return node;
  }

  function iconButton(label, glyph, click) {
    const node = button('', 'cbench-icon', click, glyph);
    node.setAttribute('aria-label', label);
    node.title = label;
    return node;
  }

  function numberToken(v) {
    if (Number.isNaN(v)) return '__sctrial_nan__';
    if (v === Infinity) return '__sctrial_inf__';
    if (v === -Infinity) return '__sctrial_negative_inf__';
    return v;
  }
  const encode = (value) =>
    JSON.stringify(value, (_, v) =>
      typeof v === 'number' && !Number.isFinite(v) ? { $sctrialNumber: numberToken(v) } : v,
    );
  const decode = (value) =>
    JSON.parse(value, (_, v) => {
      if (!v || typeof v !== 'object' || Object.keys(v).length !== 1) return v;
      switch (v.$sctrialNumber) {
        case '__sctrial_nan__':
          return NaN;
        case '__sctrial_inf__':
          return Infinity;
        case '__sctrial_negative_inf__':
          return -Infinity;
        default:
          return v;
      }
    });
  const clone = (v) =>
    typeof structuredClone === 'function' ? structuredClone(v) : decode(encode(v));
  const dict = (v) => v && typeof v === 'object' && !Array.isArray(v);

  function stable(v) {
    if (Array.isArray(v)) return v.map(stable);
    if (!dict(v)) return v;
    return Object.fromEntries(
      Object.keys(v)
        .sort()
        .map((k) => [k, stable(v[k])]),
    );
  }

  const rowUnit = (row, snapshot) => row?.unit || snapshot?.unit || '';
  const signatures = new WeakMap();
  function fingerprint(s) {
    if (!signatures.has(s)) {
      // Plot mode and highlighted feature travel with a pin when it is reopened.
      // The fitted comparison is identified by its study design and feature panel.
      const spec = { ...s.spec };
      delete spec.mode;
      delete spec.feature;
      const rows = s.rows
        .map((row) => ({ ...row, unit: rowUnit(row, s) }))
        .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
      signatures.set(s, encode(stable([spec, s.method || '', rows])));
    }
    return signatures.get(s);
  }

  function isSnapshot(s) {
    if (!dict(s) || !dict(s.spec) || !Array.isArray(s.rows) || !s.rows.length) return false;
    const ids = new Set();
    return s.rows.every((row) => {
      if (!dict(row) || typeof row.id !== 'string' || !row.id || ids.has(row.id)) return false;
      ids.add(row.id);
      return true;
    });
  }

  function readSaved(key) {
    try {
      const raw = localStorage.getItem(prefix + key);
      if (!raw) return {};
      const saved = decode(raw);
      if (!dict(saved) || saved.version !== storageVersion || !Array.isArray(saved.pins)) return {};
      const seen = new Set();
      const valid = saved.pins.filter((p) => {
        if (
          !dict(p) ||
          typeof p.id !== 'string' ||
          !p.id ||
          seen.has(p.id) ||
          typeof p.name !== 'string' ||
          !isSnapshot(p.snapshot)
        )
          return false;
        seen.add(p.id);
        return true;
      });
      return { pins: valid, left: saved.left, right: saved.right };
    } catch {
      return {};
    }
  }

  const designNames = {
    did: 'Difference-in-differences',
    paired: 'Paired change',
    cross: 'Cross-sectional',
    observe: 'Participant mean',
  };
  function visits(s) {
    const x = s.spec;
    if ((x.design === 'did' || x.design === 'paired') && x.baseline && x.visit)
      return x.baseline + ' → ' + x.visit;
    return x.visit || (Array.isArray(x.visits) ? x.visits.join(' → ') : '');
  }
  function groups(s) {
    const x = s.spec;
    if (Array.isArray(x.orderedGroups)) return x.orderedGroups.join(' − ');
    if (x.arm && x.reference && ['did', 'cross'].includes(x.design))
      return x.arm + ' − ' + x.reference;
    return x.arm || '';
  }
  function defaultTitle(s) {
    return String(
      s.title ||
        [s.spec.population, visits(s)].filter(Boolean).join(' · ') ||
        designNames[s.spec.design] ||
        'Current contrast',
    ).slice(0, 110);
  }

  function estimate(x) {
    if (x === Infinity || x === 'Infinity') return '+∞';
    if (x === -Infinity || x === '-Infinity') return '−∞';
    if (!Number.isFinite(x)) return '—';
    const magnitude = Math.abs(x);
    const exact =
      magnitude > 0 && magnitude < 0.001
        ? magnitude.toExponential(2)
        : new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 3,
          }).format(magnitude);
    return (x < 0 ? '−' : x > 0 ? '+' : '') + exact;
  }
  function hasInterval(r) {
    const usable = (n) =>
      (typeof n === 'number' && !Number.isNaN(n)) || n === 'Infinity' || n === '-Infinity';
    return usable(r.lo) && usable(r.hi);
  }

  function plotNode(tag, attributes, parent) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
    if (parent) parent.append(node);
    return node;
  }

  function estimateDomain(rows) {
    let bound = 0,
      finite = false;
    for (const row of rows) {
      if (!row) continue;
      for (const value of [row.estimate, row.lo, row.hi]) {
        if (Number.isFinite(value)) {
          finite = true;
          bound = Math.max(bound, Math.abs(value));
        }
      }
    }
    if (!finite) return null;
    if (bound === 0) return 1;
    const base = 10 ** Math.floor(Math.log10(bound));
    if (!Number.isFinite(base) || base === 0) return bound;
    const step = [1, 2, 2.5, 5, 10].find((v) => v * base >= bound);
    return Number.isFinite(step * base) ? step * base : bound;
  }

  function axisNumber(x) {
    if (x === 0) return '0';
    const n = Math.abs(x);
    const label = n < 0.001 || n >= 10000 ? n.toExponential(0) : String(Number(n.toPrecision(3)));
    return (x < 0 ? '−' : '+') + label;
  }

  function axisLabels(domain) {
    const box = el('div', 'cbench-axis');
    box.setAttribute(
      'aria-label',
      'Shared estimate scale from ' + estimate(-domain) + ' to ' + estimate(domain),
    );
    for (const [className, value] of [
      ['lo', -domain],
      ['zero', 0],
      ['hi', domain],
    ])
      box.append(el('span', 'cbench-tick-' + className, axisNumber(value)));
    return box;
  }

  function pairPlot(ra, rb, domain, name) {
    const x = (v) => 50 + 45 * Math.max(-1, Math.min(1, Number(v) / domain)) + '%';
    const summary = [ra, rb]
      .map((r, i) => {
        const pin = i ? 'B' : 'A';
        if (!r) return pin + ' feature absent';
        return (
          pin +
          ' ' +
          estimate(r.estimate) +
          (hasInterval(r) ? ', 95% interval ' + estimate(r.lo) + ' to ' + estimate(r.hi) : '')
        );
      })
      .join('. ');
    const svg = plotNode('svg', {
      class: 'cbench-forest',
      width: '100%',
      height: 29,
      role: 'img',
      'aria-label': name + '. ' + summary,
    });
    const line = (x1, y1, x2, y2, attributes) =>
      plotNode('line', { x1, y1, x2, y2, ...attributes }, svg);
    line('50%', 0, '50%', 29, { stroke: '#93a7a7', 'stroke-dasharray': '2 2' });
    [ra, rb].forEach((r, i) => {
      const y = i ? 22 : 7,
        color = i ? 'var(--cb-b)' : 'var(--cb-a)';
      line('5%', y, '95%', y, { stroke: '#e6eced' });
      if (!r) return;
      if (hasInterval(r)) {
        const left = x(r.lo),
          right = x(r.hi);
        line(left, y, right, y, { stroke: color, 'stroke-width': 2.2 });
        for (const [end, at, sign] of [
          [r.lo, left, -1],
          [r.hi, right, 1],
        ]) {
          if (Number.isFinite(end))
            line(at, y - 3, at, y + 3, { stroke: color, 'stroke-width': 1.4 });
          else {
            const place = plotNode(
              'svg',
              { x: at, y, width: 1, height: 1, overflow: 'visible' },
              svg,
            );
            plotNode(
              'path',
              {
                d: 'M ' + -sign * 4 + ' -3 L 0 0 L ' + -sign * 4 + ' 3',
                stroke: color,
                'stroke-width': 1.5,
                fill: 'none',
              },
              place,
            );
          }
        }
      }
      if (Number.isFinite(r.estimate)) {
        const point = plotNode(
          'svg',
          { x: x(r.estimate), y, width: 1, height: 1, overflow: 'visible' },
          svg,
        );
        if (i)
          plotNode(
            'path',
            {
              d: 'M 0 -4.6 L 4.6 0 L 0 4.6 L -4.6 0 Z',
              fill: color,
              stroke: '#fff',
              'stroke-width': 1,
            },
            point,
          );
        else
          plotNode(
            'circle',
            { cx: 0, cy: 0, r: 4, fill: color, stroke: '#fff', 'stroke-width': 1 },
            point,
          );
      }
    });
    return svg;
  }

  function createContrastBench({
    getSnapshot,
    onRestore,
    onNotice = () => {},
    onChange = () => {},
    onExplore,
  } = {}) {
    let scope = '',
      persistent = false,
      sharedFeatureUnit = false,
      pins = [],
      leftId = null,
      rightId = null,
      container = null,
      undo = null;
    let renameId = null,
      localCounter = 0,
      pendingFocus = null;

    const find = (id) => pins.find((p) => p.id === id);
    const notice = (message) => {
      onNotice(message);
    };
    function persist() {
      if (!persistent || !scope) return;
      try {
        localStorage.setItem(
          prefix + scope,
          encode({ version: storageVersion, pins, left: leftId, right: rightId }),
        );
      } catch {
        notice('Pins are available for this session. Browser storage is unavailable.');
      }
    }
    function selection() {
      if (!find(leftId)) leftId = pins[0]?.id || null;
      if (!find(rightId) || rightId === leftId)
        rightId = pins.find((p) => p.id !== leftId)?.id || null;
    }
    function redraw() {
      if (container?.isConnected && container.firstElementChild?.classList.contains('cbench'))
        render(container);
    }
    function changed() {
      selection();
      persist();
      redraw();
      onChange(pins.length);
    }
    function setStudy(key, options = {}) {
      if (typeof key !== 'string' || !key) throw new Error('A study key is required.');
      const store = options.persist === true;
      const commonUnit = options.sharedFeatureUnit === true;
      if (scope === key && store === persistent && commonUnit === sharedFeatureUnit) return;
      scope = key;
      persistent = store;
      sharedFeatureUnit = commonUnit;
      const saved = store ? readSaved(key) : {};
      pins = saved.pins || [];
      leftId = saved.left || pins[0]?.id || null;
      rightId = saved.right || null;
      undo = null;
      renameId = null;
      pendingFocus = null;
      selection();
      redraw();
      onChange(pins.length);
    }
    function pin() {
      if (!scope) {
        notice('Open a study first.');
        return false;
      }
      const current = getSnapshot?.();
      if (!isSnapshot(current)) {
        notice('Select a comparison with results to pin.');
        return false;
      }
      const snapshot = clone(current);
      const signature = fingerprint(snapshot);
      const duplicate = pins.find((p) => fingerprint(p.snapshot) === signature);
      if (duplicate) {
        notice('This contrast is already pinned.');
        return false;
      }
      const id =
        globalThis.crypto?.randomUUID?.() ||
        'pin-' + Date.now().toString(36) + '-' + ++localCounter;
      pins.push({ id, name: defaultTitle(snapshot), snapshot });
      if (!leftId) leftId = id;
      else rightId = id;
      undo = null;
      renameId = null;
      changed();
      notice('Contrast pinned.');
      return true;
    }
    function remove(id) {
      const index = pins.findIndex((p) => p.id === id);
      if (index < 0) return;
      undo = { pin: pins[index], index, left: leftId, right: rightId };
      pins.splice(index, 1);
      renameId = null;
      pendingFocus = 'undo';
      changed();
      notice('Pin removed.');
    }
    function restoreDeleted() {
      if (!undo) return;
      const item = undo;
      pins.splice(Math.min(item.index, pins.length), 0, item.pin);
      leftId = item.left;
      rightId = item.right;
      undo = null;
      pendingFocus = 'open-' + item.pin.id;
      changed();
      notice('Pin restored.');
    }
    function setPair(side, id) {
      if (!find(id)) return;
      if (side === 'a') {
        if (id === rightId) rightId = leftId;
        leftId = id;
      } else {
        if (id === leftId) leftId = rightId;
        rightId = id;
      }
      pendingFocus = 'selector-' + side;
      selection();
      persist();
      redraw();
    }
    function openPin(id) {
      const item = find(id);
      if (item && onRestore) onRestore(clone(item.snapshot.spec), clone(item.snapshot));
    }

    function sideSelector(side, item) {
      const block = el('div', 'cbench-side cbench-' + side);
      const badge = el('span', 'cbench-letter', side.toUpperCase());
      badge.setAttribute('aria-hidden', 'true');
      const col = el('div', 'cbench-choice');
      const select = el('select', 'cbench-select');
      select.dataset.benchFocus = 'selector-' + side;
      select.setAttribute('aria-label', 'Pinned contrast ' + side.toUpperCase());
      pins.forEach((p) => select.add(new Option(p.name, p.id)));
      if (!item) {
        const placeholder = new Option(
          side === 'a' ? 'First contrast' : 'Second contrast',
          '',
          true,
          true,
        );
        select.add(placeholder, 0);
        select.value = '';
        select.disabled = true;
      } else {
        select.value = item.id;
        select.addEventListener('change', () => setPair(side, select.value));
      }
      col.append(select);
      if (item) {
        const x = item.snapshot;
        col.append(
          el(
            'small',
            '',
            [designNames[x.spec.design] || x.spec.design, groups(x)].filter(Boolean).join(' · '),
          ),
        );
        col.append(el('small', '', [visits(x), x.spec.population].filter(Boolean).join(' · ')));
      }
      block.append(badge, col);
      return block;
    }

    function table(a, b) {
      const pins = [a, b];
      const cols = pins.map((p) =>
        p ? new Map(p.snapshot.rows.map((r) => [r.id, r])) : new Map(),
      );
      const ids = new Set([...cols[0].keys(), ...cols[1].keys()]);
      const units = cols.map(
        (rows, i) => new Set([...rows.values()].map((r) => rowUnit(r, pins[i]?.snapshot))),
      );
      const sameUnits = new Set([...units[0], ...units[1]]).size === 1;
      const matched = new Set(
        [...ids].filter((id) => {
          const left = cols[0].get(id);
          const right = cols[1].get(id);
          const unit = rowUnit(left, a?.snapshot);
          return left && right && unit && unit === rowUnit(right, b.snapshot);
        }),
      );
      const compare = matched.size > 0;
      const sharedDomain =
        compare && sameUnits && sharedFeatureUnit
          ? estimateDomain([...cols[0].values(), ...cols[1].values()])
          : null;
      const scroll = el('div', 'cbench-scroll');
      scroll.tabIndex = 0;
      scroll.setAttribute('role', 'region');
      scroll.setAttribute('aria-label', 'Scrollable comparison table');
      const t = el('table', 'cbench-table');
      t.setAttribute('aria-label', 'Side-by-side feature estimates for pinned contrasts');
      const header = t.createTHead().insertRow();
      const columnLabels = pins.map((pin, index) => {
        const side = index ? 'B' : 'A';
        if (!pin) return side;
        const unit = units[index].size === 1 ? [...units[index]][0] : '';
        return side + ' · ' + (unit || 'Estimate');
      });
      ['Feature', ...columnLabels].forEach((text) => {
        const c = el('th', '', text);
        c.scope = 'col';
        header.append(c);
      });
      if (compare) {
        t.classList.add('cbench-with-forest');
        if (sharedDomain) header.cells[0].append(axisLabels(sharedDomain));
        else header.cells[0].textContent = 'Feature · separate scales';
        [1, 2].forEach((column) => {
          const glyph = el('span', 'cbench-key cbench-key-' + column);
          glyph.setAttribute('aria-hidden', 'true');
          header.cells[column].prepend(glyph);
        });
      }
      const body = t.createTBody();
      ids.forEach((id) => {
        const tr = body.insertRow();
        const exemplar = cols[0].get(id) || cols[1].get(id);
        const name = el('th', 'cbench-row-label', exemplar.label || id);
        name.scope = 'row';
        const pair = cols.map((records) => records.get(id));
        const rowUnits = [
          ...new Set(pair.flatMap((row, i) => (row ? [rowUnit(row, pins[i].snapshot)] : []))),
        ];
        const showRowUnit =
          units.some((values) => values.size > 1) && rowUnits.length === 1 && rowUnits[0];
        if (showRowUnit) name.append(el('small', '', rowUnits[0]));
        name.setAttribute(
          'aria-label',
          (exemplar.label || id) + (showRowUnit ? ', ' + rowUnits[0] : ''),
        );
        if (matched.has(id)) {
          const domain = sharedDomain || estimateDomain(pair);
          if (domain) {
            name.append(pairPlot(pair[0], pair[1], domain, exemplar.label || id));
            if (!sharedDomain) name.append(axisLabels(domain));
          }
        }
        tr.append(name);
        cols.forEach((records, i) => {
          const row = records.get(id),
            td = tr.insertCell();
          td.className = 'cbench-value cbench-' + (i ? 'b' : 'a');
          if (!row) {
            td.append(el('span', 'cbench-unavailable', '—'));
            if (i === 0 || b) td.append(el('small', '', 'Feature absent'));
            return;
          }
          td.append(el('strong', '', estimate(row.estimate)));
          if (!showRowUnit && units[i].size > 1 && rowUnit(row, pins[i].snapshot)) {
            td.append(el('small', '', rowUnit(row, pins[i].snapshot)));
          }
          if (hasInterval(row))
            td.append(el('small', '', '95% CI ' + estimate(row.lo) + ' to ' + estimate(row.hi)));
          if (Number.isInteger(row.n) && row.n >= 0)
            td.append(el('small', '', 'n = ' + row.n.toLocaleString()));
        });
      });
      scroll.append(t);
      return scroll;
    }
    function renameForm(item) {
      const form = el('form', 'cbench-name-form');
      const input = el('input', 'cbench-name-input');
      input.maxLength = 110;
      input.value = item.name;
      input.setAttribute('aria-label', 'Pin name');
      input.dataset.benchFocus = 'edit-' + item.id;
      const apply = el('button', 'cbench-button cbench-strong', 'Save');
      apply.type = 'submit';
      const cancel = button('Cancel', '', () => {
        renameId = null;
        pendingFocus = 'rename-' + item.id;
        redraw();
      });
      form.append(input, apply, cancel);
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        const title = input.value.trim();
        if (!title) {
          input.setCustomValidity('Enter a name.');
          input.reportValidity();
          return;
        }
        item.name = title;
        renameId = null;
        pendingFocus = 'rename-' + item.id;
        changed();
      });
      input.addEventListener('input', () => input.setCustomValidity(''));
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          e.preventDefault();
          cancel.click();
        }
      });
      return form;
    }
    function shelfRow(item) {
      const li = el('li', 'cbench-pin');
      if (renameId === item.id) {
        li.append(renameForm(item));
        return li;
      }
      const badge = el(
        'span',
        'cbench-dot ' + (item.id === leftId ? 'cbench-a' : item.id === rightId ? 'cbench-b' : ''),
        item.id === leftId ? 'A' : item.id === rightId ? 'B' : '•',
      );
      badge.setAttribute('aria-hidden', 'true');
      const block = el('div', 'cbench-row-label');
      block.append(el('strong', '', item.name));
      block.append(
        el(
          'small',
          '',
          [groups(item.snapshot), item.snapshot.spec.population, visits(item.snapshot)]
            .filter(Boolean)
            .join(' · '),
        ),
      );
      const open = iconButton('Reopen ' + item.name, 'open', () => openPin(item.id));
      open.dataset.benchFocus = 'open-' + item.id;
      const rename = iconButton('Rename ' + item.name, 'pencil', () => {
        renameId = item.id;
        pendingFocus = 'edit-' + item.id;
        redraw();
      });
      rename.dataset.benchFocus = 'rename-' + item.id;
      const drop = iconButton('Remove ' + item.name, 'close', () => remove(item.id));
      const controls = el('div', 'cbench-pin-actions');
      controls.append(open, rename, drop);
      li.append(badge, block, controls);
      return li;
    }
    function render(target) {
      container = target;
      selection();
      const root = el('section', 'cbench');
      root.setAttribute('aria-label', 'Pinned comparisons');
      const a = find(leftId),
        b = find(rightId);
      const main = el('div', 'cbench-main'),
        head = el('header', 'cbench-header');
      head.append(el('h2', '', 'Side-by-side comparison'));
      const add = button('Pin current contrast', 'cbench-strong', pin, 'pin');
      add.dataset.benchFocus = 'pin';
      head.append(add);
      main.append(head);
      const pair = el('div', 'cbench-pair');
      const swap = iconButton('Swap pinned contrasts', 'swap', () => {
        [leftId, rightId] = [rightId, leftId];
        pendingFocus = 'swap';
        persist();
        redraw();
      });
      swap.classList.add('cbench-swap');
      swap.disabled = !a || !b;
      swap.dataset.benchFocus = 'swap';
      pair.append(sideSelector('a', a), swap, sideSelector('b', b));
      main.append(pair);
      if (!a) {
        const empty = el('div', 'cbench-empty');
        empty.append(
          el('h3', '', 'Pin a contrast to start'),
          el(
            'p',
            '',
            'Keep one comparison here, then change the visit, group, or population and pin the next.',
          ),
        );
        if (onExplore) empty.append(button('Open workspace', 'cbench-text', onExplore, 'open'));
        main.append(empty);
      } else {
        if (!b) {
          const tip = el('div', 'cbench-tip');
          tip.append(
            el('span', '', 'Change the study contrast and pin a second result to compare.'),
          );
          if (onExplore) tip.append(button('Open workspace', 'cbench-text', onExplore, 'open'));
          main.append(tip);
        } else {
          const differences = [];
          if (a.snapshot.spec.design !== b.snapshot.spec.design)
            differences.push('Different study designs');
          const rightRows = new Map(b.snapshot.rows.map((r) => [r.id, r]));
          if (
            a.snapshot.rows.some((r) => {
              const right = rightRows.get(r.id);
              return right && rowUnit(r, a.snapshot) !== rowUnit(right, b.snapshot);
            })
          )
            differences.push('Units differ for some outcomes');
          if (differences.length)
            main.append(
              el(
                'div',
                'cbench-tip',
                differences.join(' · ') + '. Interpret each estimate on its own scale.',
              ),
            );
        }
        main.append(table(a, b));
      }
      root.append(main);
      const shelf = el('section', 'cbench-shelf');
      const shelfHead = el('div', 'cbench-shelf-head');
      shelfHead.append(
        el('h3', '', 'Pinned contrasts'),
        el('span', 'cbench-count', String(pins.length)),
      );
      if (undo) {
        const undoButton = button('Undo removal', 'cbench-text', restoreDeleted);
        undoButton.dataset.benchFocus = 'undo';
        shelfHead.append(undoButton);
      }
      shelf.append(shelfHead);
      if (pins.length) {
        const list = el('ul', 'cbench-pins');
        pins.forEach((p) => list.append(shelfRow(p)));
        shelf.append(list);
      }
      if (pins.length || !persistent)
        shelf.append(
          el(
            'p',
            'cbench-storage',
            persistent
              ? 'Example pins are saved in this browser.'
              : 'Study pins are kept for this open study only.',
          ),
        );
      if (pins.length || undo || !persistent) root.append(shelf);
      target.replaceChildren(root);
      if (pendingFocus) {
        const n = Array.from(root.querySelectorAll('[data-bench-focus]')).find(
          (n) => n.dataset.benchFocus === pendingFocus,
        );
        if (n) {
          n.focus();
          if (n.tagName === 'INPUT') n.select();
        }
        pendingFocus = null;
      }
    }
    return Object.freeze({ setStudy, pin, render, count: () => pins.length });
  }
  window.createContrastBench = createContrastBench;
})();
