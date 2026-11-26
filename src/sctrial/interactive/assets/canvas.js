(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const states = new WeakMap();
  const palettes = ['#5145bb', '#247778'];
  const fg = '#20283b',
    subdued = '#576278',
    grid = '#dce0e9',
    minor = '#edf0f5';
  const pathLimit = 48,
    dotLimit = 140;
  let serial = 0;

  function make(parent, tag, attrs = {}, value) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (value !== undefined) n.textContent = value;
    parent.append(n);
    return n;
  }
  function html(tag, cls, value) {
    const n = document.createElement(tag);
    n.className = cls;
    if (value !== undefined) n.textContent = value;
    return n;
  }
  const text = (p, x, y, value, attrs = {}) =>
    make(p, 'text', { x, y, fill: subdued, 'font-size': 11, ...attrs }, value);
  const line = (p, x1, y1, x2, y2, attrs = {}) =>
    make(p, 'line', { x1, y1, x2, y2, stroke: grid, ...attrs });
  const crisp = { 'vector-effect': 'non-scaling-stroke' };
  const safeColor = (s, fallback) => (/^#[0-9a-f]{3,8}$/i.test(s) ? s : fallback);
  function num(value, sign = false) {
    if (!Number.isFinite(value)) return '—';
    if (value !== 0 && (Math.abs(value) < 0.005 || Math.abs(value) >= 1e6))
      return (sign && value > 0 ? '+' : '') + value.toExponential(1).replace(/-/g, '−');
    return (
      (value < 0 ? '−' : sign && value > 0 ? '+' : '') +
      Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    );
  }
  function tickNum(v, step) {
    if (v !== 0 && (Math.abs(v) >= 1e5 || step < 1e-4)) return v.toExponential(1).replace(/-/g, '−');
    const digits = Math.max(
      0,
      Math.min(
        6,
        -Math.floor(Math.log10(step)) +
          (Math.round((step / 10 ** Math.floor(Math.log10(step))) * 10) === 25 ? 1 : 0),
      ),
    );
    return Number(v.toFixed(digits))
      .toLocaleString('en-US', { maximumFractionDigits: digits })
      .replace('-', '−');
  }
  function domain(values, includeZero = false) {
    let min = includeZero ? 0 : Infinity,
      max = includeZero ? 0 : -Infinity;
    values.forEach((v) => {
      if (Number.isFinite(v)) {
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1, step: 0.25 };
    const delta = max - min || Math.abs(max) * 0.25 || 1;
    const low = min - 0.07 * delta,
      high = max + 0.07 * delta,
      raw = (high - low) / 5;
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = ([1, 2, 2.5, 5, 10].find((a) => a * power >= raw) || 10) * power;
    const lowTick = min >= 0 && low <= 0 ? 0 : Math.floor(low / step) * step;
    const highTick = max <= 0 && high >= 0 ? 0 : Math.ceil(high / step) * step;
    return { min: lowTick, max: highTick > lowTick ? highTick : lowTick + step, step };
  }
  function ticks(d) {
    return Array.from(
      { length: Math.min(12, Math.round((d.max - d.min) / d.step) + 1) },
      (_, i) => d.min + i * d.step,
    );
  }
  function sampled(rows, n, selectedId) {
    if (rows.length <= n) return rows;
    const out = Array.from(
      { length: n },
      (_, i) => rows[Math.round((i * (rows.length - 1)) / (n - 1))],
    );
    const pinned = rows.find((r) => r.id === selectedId);
    if (pinned && !out.includes(pinned)) out[out.length - 1] = pinned;
    return out;
  }
  function trim(textValue, space, font = 11) {
    const s = String(textValue),
      max = Math.max(2, Math.floor(space / (font * 0.57)));
    return s.length <= max ? s : s.slice(0, Math.max(1, max - 1)) + '…';
  }
  function addTitle(n, value) {
    make(n, 'title', {}, value);
  }
  function marker(parent, x, y, size, color, groupIndex, attrs = {}) {
    if (groupIndex % 2)
      return make(parent, 'path', {
        d:
          'M ' +
          x +
          ' ' +
          (y - size) +
          ' l ' +
          size +
          ' ' +
          size +
          ' -' +
          size +
          ' ' +
          size +
          ' -' +
          size +
          ' -' +
          size +
          ' Z',
        fill: color,
        ...attrs,
      });
    return make(parent, 'circle', { cx: x, cy: y, r: size, fill: color, ...attrs });
  }
  function pointsPath(coords) {
    let prev = false,
      path = '';
    coords.forEach((p) => {
      if (!p) {
        prev = false;
        return;
      }
      path += (prev ? 'L' : 'M') + p[0].toFixed(2) + ',' + p[1].toFixed(2);
      prev = true;
    });
    return path;
  }
  function prep(model) {
    const visits = model.visits || [],
      names = new Set(visits);
    const groups = (model.groups || []).map((g, i) => ({
      name: g.name,
      color: safeColor(g.color, palettes[i % 2]),
      rows: [],
      totals: new Map(),
    }));
    const groupMap = new Map(groups.map((g) => [g.name, g]));
    const byId = new Map();
    for (const source of model.series || []) {
      const g = groupMap.get(source.arm);
      if (!g) continue;
      const points = new Map();
      for (const pt of source.points || [])
        if (names.has(pt.visit) && Number.isFinite(pt.value)) points.set(pt.visit, pt.value);
      if (!points.size) continue;
      const person = { id: source.id, arm: source.arm, points };
      byId.set(person.id, person);
      g.rows.push(person);
      for (const [visit, value] of points) {
        const t = g.totals.get(visit) || [0, 0];
        t[0] += value;
        t[1]++;
        g.totals.set(visit, t);
      }
    }
    return { groups, byId, visits };
  }
  const longitudinal = (design) => !['cross', 'observe'].includes(design);
  function change(row, m) {
    const v = row.points.get(m.visit),
      base = row.points.get(m.baseline);
    if (!Number.isFinite(v) || (longitudinal(m.design) && !Number.isFinite(base))) return null;
    return longitudinal(m.design) ? v - base : v;
  }

  function paint(state) {
    const { container, m, a, data } = state;
    if (!container.isConnected || container.clientWidth < 8) return;
    const active = document.activeElement,
      focusKey = container.contains(active) ? active.dataset.ccFocus : null;
    const w = Math.floor(container.clientWidth),
      tight = w < 490;
    const h = w < 400 ? 273 : w < 600 ? 294 : w < 850 ? 317 : 341;
    const bounds = {
      left: tight ? 39 : 52,
      right: w - (tight ? 21 : 36),
      top: 57,
      bottom: h - 52,
      width: w,
      height: h,
      tight,
    };
    state.lastWidth = w;
    container.replaceChildren();
    const root = html('div', 'cc-surface');
    container.append(root);
    const svg = make(root, 'svg', {
      viewBox: '0 0 ' + w + ' ' + h,
      role: 'group',
      'aria-label':
        m.feature.label +
        (m.mode === 'changes' ? ' · Participant distribution' : ' · Participant trajectories'),
    });
    const readout = html('div', 'cc-readout'),
      readMain = html('div', 'cc-readout-main'),
      legend = html('div', 'cc-legend');
    readout.append(readMain, legend);
    root.append(readout);
    const tip = html('div', 'cc-tip');
    tip.hidden = true;
    root.append(tip);
    const individuals = new Map();
    let roster = 0,
      shown = 0;
    const selected = data.byId.get(m.selectedId);
    const gMap = new Map(data.groups.map((g) => [g.name, g]));
    const k1 = html('span', ''),
      k2 = html('span', '');
    k1.append(
      html('i', 'cc-key ' + (m.mode === 'changes' ? 'cc-key-dot' : '')),
      document.createTextNode('Participant'),
    );
    k2.append(
      html('i', 'cc-key ' + (m.mode === 'changes' ? 'cc-key-diamond' : 'cc-key-mean')),
      document.createTextNode(m.mode === 'changes' ? 'Group mean' : 'Mean at each visit'),
    );
    legend.append(k1, k2);

    function read(person) {
      readMain.replaceChildren();
      if (!person) {
        readMain.append(
          html(
            'strong',
            '',
            shown < roster
              ? shown + ' of ' + roster + ' shown'
              : roster + ' ' + (roster === 1 ? 'participant' : 'participants'),
          ),
        );
        if (!tight)
          readMain.append(
            html('span', '', m.mode === 'changes' && longitudinal(m.design) ? 'Complete pairs' : ''),
          );
        return;
      }
      const dot = html('i', 'cc-readout-arm');
      dot.style.setProperty('--mark-color', gMap.get(person.arm).color);
      readMain.append(dot, html('strong', '', person.id), html('span', '', person.arm));
      if (longitudinal(m.design)) {
        const v0 = person.points.get(m.baseline),
          v1 = person.points.get(m.visit);
        readMain.append(html('span', '', m.baseline + ' ' + num(v0) + ' → ' + m.visit + ' ' + num(v1)));
        if (Number.isFinite(v0) && Number.isFinite(v1))
          readMain.append(html('strong', '', 'Δ ' + num(v1 - v0, true)));
      } else readMain.append(html('span', '', m.visit + ' ' + num(person.points.get(m.visit))));
    }
    function highlight(id) {
      for (const [key, n] of individuals) {
        n.classList.toggle('cc-active', key === id);
        n.classList.toggle('cc-muted', id != null && key !== id);
      }
      read(data.byId.get(id));
    }
    function tooltip(content, x, y) {
      tip.textContent = content;
      tip.hidden = false;
      const px = Math.max(4, Math.min(w - tip.offsetWidth - 4, x + 9));
      tip.style.left = px + 'px';
      tip.style.top = Math.max(3, Math.min(h - tip.offsetHeight - 2, y - tip.offsetHeight - 8)) + 'px';
    }
    function atAnchor(el) {
      const r = el.getBoundingClientRect(),
        b = root.getBoundingClientRect();
      return [r.x + r.width / 2 - b.x, r.y - b.y];
    }
    function bindPerson(g, person, anchor, description) {
      individuals.set(person.id, g);
      g.classList.add('cc-person', 'cc-interactive');
      g.setAttribute('tabindex', '0');
      g.setAttribute('role', 'button');
      g.setAttribute('aria-label', description);
      g.setAttribute('aria-pressed', String(m.selectedId === person.id));
      g.dataset.ccFocus = 'person:' + person.id;
      addTitle(g, description);
      const over = (e) => {
        highlight(person.id);
        const box = root.getBoundingClientRect();
        tooltip(description, e.clientX - box.left, e.clientY - box.top);
      };
      const focus = () => {
        highlight(person.id);
        tooltip(description, ...atAnchor(anchor));
      };
      const leave = () => {
        tip.hidden = true;
        highlight(m.selectedId);
      };
      const choose = () => a.onSelectParticipant?.(m.selectedId === person.id ? null : person.id);
      g.addEventListener('pointerenter', over);
      g.addEventListener('pointerleave', leave);
      g.addEventListener('focus', focus);
      g.addEventListener('blur', leave);
      g.addEventListener('click', choose);
      g.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          choose();
        } else if (e.key === 'Escape') {
          leave();
          if (m.selectedId != null) a.onSelectParticipant?.(null);
        } else if (['ArrowDown', 'ArrowUp'].includes(e.key)) {
          e.preventDefault();
          const nodes = [...individuals.values()],
            i = nodes.indexOf(g);
          nodes[(i + (e.key === 'ArrowDown' ? 1 : -1) + nodes.length) % nodes.length]?.focus();
        }
      });
    }
    function meanTarget(g, description, anchor, action) {
      g.setAttribute('class', 'cc-mean cc-interactive');
      g.setAttribute('role', action ? 'button' : 'img');
      g.setAttribute('aria-label', description);
      g.setAttribute('tabindex', '0');
      addTitle(g, description);
      const reveal = () => tooltip(description, ...atAnchor(anchor));
      if (action) {
        g.style.cursor = 'pointer';
        g.addEventListener('click', action);
        g.addEventListener('keydown', (e) => {
          if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            action();
          }
        });
      }
      g.addEventListener('pointerenter', reveal);
      g.addEventListener('focus', reveal);
      g.addEventListener('pointerleave', () => {
        tip.hidden = true;
      });
      g.addEventListener('blur', () => {
        tip.hidden = true;
      });
    }
    const ctx = { svg, m, a, data, b: bounds, bindPerson, meanTarget, tooltip, tip };
    if (m.mode === 'changes') {
      const summary = drawChanges(ctx);
      roster = summary.total;
      shown = summary.shown;
    } else {
      const summary = drawTraces(ctx);
      visitCounts(root, data, summary.axisVisits || []);
      roster = summary.total;
      shown = summary.shown;
    }
    highlight(selected?.id);
    if (focusKey) {
      const next = Array.from(root.querySelectorAll('[data-cc-focus]')).find(
        (n) => n.dataset.ccFocus === focusKey,
      );
      if (next) next.focus({ preventScroll: true });
    }
  }

  function visitCounts(root, data, visits) {
    if (!visits.length) return;
    const wrap = html('div', 'cc-counts');
    const table = html('table', '');
    const caption = html('caption', '', 'Participants measured at each visit');
    const head = table.createTHead().insertRow();
    const body = table.createTBody();
    const column = (value) => {
      const cell = html('th', '', value);
      cell.scope = 'col';
      head.append(cell);
    };
    column('Group');
    visits.forEach(column);
    for (const group of data.groups) {
      const row = body.insertRow();
      const name = html('th', '', group.name);
      name.scope = 'row';
      row.append(name);
      for (const visit of visits) {
        row.insertCell().textContent = String(group.totals.get(visit)?.[1] || 0);
      }
    }
    table.prepend(caption);
    wrap.append(table);
    root.append(wrap);
  }

  function drawTraces(ctx) {
    const { svg, m, a, b, data, bindPerson, meanTarget } = ctx;
    const { visits, groups } = data,
      { left, right, top, bottom, width, height, tight } = b;
    const x = (idx) =>
      visits.length < 2
        ? (left + right) / 2
        : left + 7 + (idx * (right - left - 14)) / (visits.length - 1);
    let roster = 0,
      shown = 0;
    const draws = groups.map((g) => {
      roster += g.rows.length;
      const show = sampled(g.rows, pathLimit, m.selectedId);
      shown += show.length;
      return show;
    });
    const values = [];
    for (const g of groups)
      for (const person of g.rows) for (const value of person.points.values()) values.push(value);
    if (!values.length || !visits.length) {
      text(svg, width / 2, height / 2, 'No observed scores for this selection', {
        class: 'cc-empty',
        'text-anchor': 'middle',
      });
      return { total: 0, shown: 0 };
    }
    const d = domain(values),
      y = (v) => bottom - ((v - d.min) * (bottom - top)) / (d.max - d.min);
    const baselineIndex = visits.indexOf(m.baseline),
      followIndex = visits.indexOf(m.visit);
    const reverse = longitudinal(m.design) && baselineIndex > followIndex;
    const selectedLabel = longitudinal(m.design) ? (reverse ? 'Comparison' : 'Follow-up') : 'Selected';
    const referenceLabel = reverse ? 'Reference' : 'Baseline';
    const editable = new Set(m.canSelectVisit ? m.editableVisits || [] : []);
    if (longitudinal(m.design) && baselineIndex !== -1 && followIndex !== -1) {
      const start = Math.min(x(baselineIndex), x(followIndex)),
        end = Math.max(x(baselineIndex), x(followIndex));
      make(svg, 'rect', {
        x: start,
        y: top,
        width: end - start,
        height: bottom - top,
        fill: '#6763c0',
        'fill-opacity': 0.055,
      });
    }
    const ys = ticks(d);
    ys.forEach((v, i) => {
      const yy = y(v);
      line(svg, left, yy, right, yy, { stroke: grid, 'stroke-width': 0.85 });
      if (i < ys.length - 1)
        line(svg, left, y(v + d.step / 2), right, y(v + d.step / 2), {
          stroke: minor,
          'stroke-width': 0.65,
          'stroke-dasharray': '2 4',
        });
      text(svg, left - 10, yy + 3.7, tickNum(v, d.step), {
        'text-anchor': 'end',
        class: 'cc-number',
        'font-size': 10,
      });
    });
    const headerRoom = right - left,
      unitRoom = Math.min(headerRoom * 0.4, 175);
    text(svg, left, 18, trim(m.unit || 'Score', unitRoom, 10), {
      'font-size': 10,
      'letter-spacing': '.35px',
      fill: '#586079',
    });
    const slot = Math.min(tight ? 73 : 129, (headerRoom - unitRoom) / Math.max(1, groups.length));
    const widths = groups.map((group) => Math.min(slot, 25 + String(group.name).length * 6.4));
    let gx = right - widths.reduce((sum, v) => sum + v, 0);
    groups.forEach((group, i) => {
      line(svg, gx, 14, gx + 12, 14, { stroke: group.color, 'stroke-width': 2.1 });
      marker(svg, gx + 6, 14, 2.7, '#ffffff', i, { stroke: group.color, 'stroke-width': 1.4 });
      const key = text(svg, gx + 17, 18, trim(group.name, widths[i] - 20, 10.5), {
        'font-size': 10.5,
        'font-weight': 600,
        fill: fg,
      });
      addTitle(key, group.name);
      gx += widths[i];
    });
    const maxLabels = Math.max(3, Math.round((right - left) / 80));
    const ticksToShow = new Set(
      sampled(
        visits.map((visit, i) => ({ id: visit, i })),
        maxLabels,
        null,
      ).map((v) => v.i),
    );
    if (followIndex >= 0) ticksToShow.add(followIndex);
    if (longitudinal(m.design) && baselineIndex >= 0) ticksToShow.add(baselineIndex);
    const always = new Set([followIndex, ...(longitudinal(m.design) ? [baselineIndex] : [])]);
    const others = [...ticksToShow].filter((i) => !always.has(i)).sort((a, b) => a - b);
    const shownIndexes = [...always].filter((i) => i >= 0);
    others.forEach((i) => {
      if (shownIndexes.every((j) => Math.abs(x(i) - x(j)) > 47)) shownIndexes.push(i);
    });
    shownIndexes.sort((a, b) => a - b);
    for (const i of shownIndexes) {
      const xx = x(i),
        isTime = always.has(i);
      line(svg, xx, top - 3, xx, bottom + 5, {
        stroke: isTime ? '#969bb3' : minor,
        'stroke-dasharray': isTime ? '3 5' : '2 5',
        'stroke-width': isTime ? 1 : 0.8,
      });
    }
    const traceLayer = make(svg, 'g'),
      means = make(svg, 'g');
    groups.forEach((group, k) => {
      draws[k].forEach((person) => {
        const coords = visits.map((v, i) =>
          person.points.has(v) ? [x(i), y(person.points.get(v))] : null,
        );
        const observed = coords.filter(Boolean);
        if (!observed.length) return;
        const g = make(traceLayer, 'g'),
          path = pointsPath(coords);
        make(g, 'path', {
          d: path,
          fill: 'none',
          stroke: group.color,
          'stroke-width': 1.35,
          opacity: 0.55,
          class: 'cc-ink',
          'pointer-events': 'none',
          ...crisp,
        });
        make(g, 'path', {
          d: path,
          fill: 'none',
          stroke: 'transparent',
          'stroke-width': 10,
          'pointer-events': 'stroke',
          ...crisp,
        });
        observed.forEach((p) => {
          marker(g, p[0], p[1], 2.15, group.color, k, {
            opacity: 0.7,
            class: 'cc-dot',
            'pointer-events': 'none',
          });
          make(g, 'circle', { cx: p[0], cy: p[1], r: 5.5, fill: 'transparent' });
        });
        const last = observed[observed.length - 1];
        const anchor = make(g, 'circle', {
          cx: last[0],
          cy: last[1],
          r: 5.1,
          fill: 'none',
          stroke: '#342a85',
          'stroke-width': 1.15,
          opacity: 0,
          class: 'cc-focus',
          'pointer-events': 'none',
        });
        const current = person.points.get(m.visit);
        bindPerson(
          g,
          person,
          anchor,
          person.id + ' · ' + group.name + ' · ' + m.visit + ' ' + num(current),
        );
      });
      const groupMeans = visits.map((v, i) => {
        const t = group.totals.get(v);
        return t ? [x(i), y(t[0] / t[1])] : null;
      });
      make(means, 'path', {
        d: pointsPath(groupMeans),
        fill: 'none',
        stroke: group.color,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'stroke-width': 3.35,
        'pointer-events': 'none',
        ...crisp,
      });
      groupMeans.forEach((pt, i) => {
        if (!pt) return;
        const g = make(means, 'g'),
          tot = group.totals.get(visits[i]);
        make(g, 'circle', { cx: pt[0], cy: pt[1], r: 6, fill: 'transparent' });
        marker(g, pt[0], pt[1], 4.8, '#ffffff', k, { stroke: group.color, 'stroke-width': 2.35 });
        const anchor = make(g, 'circle', {
          cx: pt[0],
          cy: pt[1],
          r: 7.8,
          fill: 'none',
          stroke: '#342a85',
          opacity: 0,
          class: 'cc-focus',
        });
        const meanValue = tot[0] / tot[1];
        const coincident = draws[k].filter(
          (person) => Math.abs(person.points.get(visits[i]) - meanValue) < 1e-10,
        );
        const next = coincident.length
          ? coincident[(coincident.findIndex((p) => p.id === m.selectedId) + 1) % coincident.length]
          : null;
        const detail =
          group.name +
          ' · ' +
          visits[i] +
          ' · Mean ' +
          num(meanValue) +
          ' · n = ' +
          tot[1] +
          (next ? ' · Inspect ' + next.id : '');
        meanTarget(
          g,
          detail,
          anchor,
          next
            ? () =>
                a.onSelectParticipant?.(
                  coincident.length === 1 && next.id === m.selectedId ? null : next.id,
                )
            : undefined,
        );
      });
    });
    // The rail and axis label select the same visit. Only the label contributes a tab stop.
    shownIndexes.forEach((i, pos) => {
      const visit = visits[i],
        xx = x(i),
        isActive = i === followIndex,
        isBase = i === baselineIndex && longitudinal(m.design);
      const prev = pos > 0 ? Math.abs(xx - x(shownIndexes[pos - 1])) : Infinity;
      const next = pos < shownIndexes.length - 1 ? Math.abs(xx - x(shownIndexes[pos + 1])) : Infinity;
      const room = Math.min(prev, next, tight ? 79 : 126);
      const vicinity = Number.isFinite(room) ? room : tight ? 68 : 96;
      const label = trim(visit, Math.min(vicinity - 8, tight ? 74 : 112), 11);
      const cw = Math.max(25, Math.min(vicinity - 3, Math.max(label.length * 6.4 + 14, 35), width - 8));
      const cx = Math.max(cw / 2 + 2, Math.min(width - cw / 2 - 2, xx));
      const g = make(svg, 'g');
      const cy = bottom + 18;
      const hitWidth = Math.max(44, cw);
      const hitX = Math.max(0, Math.min(width - hitWidth, cx - hitWidth / 2));
      if (editable.has(visit)) {
        make(g, 'rect', { x: hitX, y: bottom + 1, width: hitWidth, height: 46, fill: 'transparent' });
      }
      if (isActive || isBase || editable.has(visit)) {
        make(g, 'rect', {
          x: cx - cw / 2,
          y: cy - 11,
          width: cw,
          height: 23,
          rx: 4,
          fill: isActive ? '#e7e5ff' : isBase ? '#f1f2f7' : '#ffffff',
          stroke: isActive ? '#b5b0ee' : '#c6cada',
          class: isActive ? '' : 'cc-chip',
        });
      }
      text(g, cx, cy + 4, label, {
        'text-anchor': 'middle',
        'font-size': 11,
        'font-weight': isActive || isBase ? 650 : 500,
        fill: isActive ? '#3a317b' : fg,
      });
      if (isActive || isBase)
        text(
          svg,
          cx,
          bottom + 42,
          (isActive ? selectedLabel : referenceLabel).toUpperCase(),
          {
            'text-anchor': 'middle',
            'font-size': 8.4,
            'letter-spacing': 1.0,
            'font-weight': 600,
            fill: '#626b80',
            'pointer-events': 'none',
          },
        );
      if (!editable.has(visit)) return;
      make(g, 'rect', {
        x: cx - cw / 2 - 2,
        y: cy - 13,
        width: cw + 4,
        height: 27,
        rx: 5,
        fill: 'none',
        stroke: '#4538a2',
        opacity: 0,
        class: 'cc-focus',
      });
      g.setAttribute('class', 'cc-visit cc-interactive');
      g.setAttribute('role', 'button');
      g.setAttribute('tabindex', 0);
      g.setAttribute('aria-pressed', String(isActive));
      g.setAttribute(
        'aria-label',
        isActive
          ? visit + ' ' + (longitudinal(m.design) ? selectedLabel.toLowerCase() + ', selected visit' : 'selected visit')
          : visit + ', select visit',
      );
      g.dataset.ccFocus = 'visit:' + visit;
      const tipCopy = isActive ? visit + ' · Selected visit' : 'Select ' + visit;
      const showTip = () => ctx.tooltip(tipCopy, xx, bottom + 3);
      const hideTip = () => {
        ctx.tip.hidden = true;
      };
      const selectVisit = () => {
        if (!isActive) a.onVisit?.(visit);
      };
      addTitle(g, tipCopy);
      for (const event of ['focus', 'pointerenter']) g.addEventListener(event, showTip);
      for (const event of ['blur', 'pointerleave']) g.addEventListener(event, hideTip);
      g.addEventListener('click', selectVisit);
      if (!isActive && editable.size > 1) {
        const hy = top - 10;
        const target = make(svg, 'g', { class: 'cc-visit-head', 'aria-hidden': 'true' });
        target.dataset.ccTargetVisit = visit;
        make(target, 'rect', {
          x: Math.max(0, Math.min(width - 44, xx - 22)),
          y: hy - 23,
          width: 44,
          height: 44,
          fill: 'transparent',
        });
        make(target, 'circle', {
          cx: xx,
          cy: hy,
          r: 4.7,
          fill: '#f3f2fb',
          stroke: '#959bb5',
          'stroke-width': 1.1,
          class: 'cc-destination',
        });
        line(target, xx, hy + 5, xx, top + 1, { stroke: '#959bb5', 'pointer-events': 'none' });
        addTitle(target, tipCopy);
        target.addEventListener('click', selectVisit);
        target.addEventListener('pointerenter', () => ctx.tooltip(tipCopy, xx, hy + 40));
        target.addEventListener('pointerleave', hideTip);
        for (const event of ['focus', 'pointerenter'])
          g.addEventListener(event, () => target.classList.add('cc-aim'));
        for (const event of ['blur', 'pointerleave'])
          g.addEventListener(event, () => target.classList.remove('cc-aim'));
      }
      g.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (!isActive) a.onVisit?.(visit);
        }
      });
    });
    if (followIndex >= 0 && editable.size > 1) {
      const valid = visits.map((v, i) => (editable.has(v) ? i : null)).filter((i) => i != null);
      const xx = x(followIndex),
        hy = top - 9;
      const h = make(svg, 'g', {
        class: 'cc-handle cc-interactive',
        role: 'slider',
        tabindex: 0,
        'aria-label': 'Selected visit',
        'aria-valuemin': 1,
        'aria-valuemax': valid.length,
        'aria-valuenow': Math.max(1, valid.indexOf(followIndex) + 1),
        'aria-valuetext': m.visit,
      });
      h.dataset.ccFocus = 'visit-handle';
      make(h, 'rect', {
        x: Math.max(0, Math.min(width - 48, xx - 24)),
        y: hy - 23,
        width: 48,
        height: 44,
        fill: 'transparent',
      });
      make(h, 'path', {
        d:
          'M ' +
          (xx - 12) +
          ' ' +
          (hy - 7) +
          ' h24 q3 0 3 3 v6 q0 3 -3 3 h-8 l-4 5 -4 -5 h-8 q-3 0 -3 -3 v-6 q0 -3 3 -3 z',
        fill: '#e7e5ff',
        class: 'cc-grip',
      });
      make(h, 'path', {
        d: 'M ' + (xx - 5) + ' ' + (hy - 4) + ' l-3 3 3 3 m10 -6 3 3 -3 3',
        fill: 'none',
        stroke: '#403984',
        'stroke-width': 1.45,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'pointer-events': 'none',
      });
      line(h, xx, hy - 3.7, xx, hy + 1.7, { stroke: '#585295', 'stroke-width': 1.3 });
      make(h, 'rect', {
        x: xx - 18,
        y: hy - 10,
        width: 36,
        height: 24,
        rx: 3,
        fill: 'none',
        stroke: '#4538a2',
        opacity: 0,
        class: 'cc-focus',
      });
      const help = 'Drag to change the visit, or use the arrow keys.';
      addTitle(h, help);
      h.setAttribute('aria-description', help);
      const showHelp = () => {
        if (!drag) ctx.tooltip(help, xx, hy + 43);
      };
      const hideHelp = () => {
        if (!drag) ctx.tip.hidden = true;
      };
      h.addEventListener('pointerenter', showHelp);
      h.addEventListener('pointerleave', hideHelp);
      h.addEventListener('focus', showHelp);
      h.addEventListener('blur', hideHelp);
      h.addEventListener('keydown', (e) => {
        const current = valid.indexOf(followIndex);
        let index;
        if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') index = Math.max(0, current - 1);
        else if (e.key === 'ArrowRight' || e.key === 'ArrowUp')
          index = Math.min(valid.length - 1, current + 1);
        else if (e.key === 'Home') index = 0;
        else if (e.key === 'End') index = valid.length - 1;
        else return;
        e.preventDefault();
        if (valid[index] !== followIndex) a.onVisit?.(visits[valid[index]]);
      });
      let pending = followIndex,
        drag = false;
      const ruler = line(svg, xx, top, xx, bottom, {
        stroke: '#5b50a9',
        'stroke-width': 1.5,
        'stroke-dasharray': '3 3',
        visibility: 'hidden',
        'pointer-events': 'none',
      });
      function move(e) {
        const rect = svg.getBoundingClientRect(),
          px = ((e.clientX - rect.left) * width) / rect.width;
        pending = valid.reduce(
          (best, j) => (Math.abs(x(j) - px) < Math.abs(x(best) - px) ? j : best),
          valid[0],
        );
        const vx = x(pending);
        h.setAttribute('transform', 'translate(' + (vx - xx) + ' 0)');
        ruler.setAttribute('x1', vx);
        ruler.setAttribute('x2', vx);
        ruler.setAttribute('visibility', 'visible');
        ctx.tooltip(visits[pending], vx, top + 28);
      }
      h.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        drag = true;
        h.classList.add('cc-dragging');
        h.setPointerCapture(e.pointerId);
        move(e);
      });
      h.addEventListener('pointermove', (e) => {
        if (drag) move(e);
      });
      function end(commit) {
        if (!drag) return;
        drag = false;
        h.classList.remove('cc-dragging');
        ctx.tip.hidden = true;
        ruler.setAttribute('visibility', 'hidden');
        h.removeAttribute('transform');
        if (commit && pending !== followIndex) a.onVisit?.(visits[pending]);
      }
      h.addEventListener('pointerup', () => end(true));
      h.addEventListener('pointercancel', () => end(false));
    }
    return { total: roster, shown, axisVisits: shownIndexes.map((i) => visits[i]) };
  }

  function drawChanges(ctx) {
    const { svg, m, b, data, bindPerson, meanTarget } = ctx;
    const { width, height, tight } = b;
    const groups = data.groups,
      top = 28,
      bottom = height - 55,
      left = tight ? 28 : 42,
      right = width - (tight ? 20 : 35);
    const paired = longitudinal(m.design),
      values = [];
    const lanes = groups.map((g) => {
      const rows = g.rows
        .map((row) => ({ id: row.id, row, value: change(row, m) }))
        .filter((p) => Number.isFinite(p.value));
      rows.forEach((p) => values.push(p.value));
      return {
        group: g,
        rows,
        mean: rows.length ? rows.reduce((sum, p) => sum + p.value, 0) / rows.length : null,
      };
    });
    if (!values.length) {
      text(
        svg,
        width / 2,
        height / 2 - 3,
        paired ? 'No complete participant pairs' : 'No observed scores',
        { 'text-anchor': 'middle', class: 'cc-empty' },
      );
      text(
        svg,
        width / 2,
        height / 2 + 19,
        trim(paired ? m.baseline + ' → ' + m.visit : m.visit, width - 20),
        { 'text-anchor': 'middle' },
      );
      return { total: 0, shown: 0 };
    }
    const d = domain(values, true),
      x = (v) => left + ((v - d.min) * (right - left)) / (d.max - d.min);
    const laneHeight = (bottom - top) / Math.max(1, lanes.length);
    const zero = x(0);
    ticks(d).forEach((v) => {
      line(svg, x(v), top, x(v), bottom + 6, {
        stroke: Math.abs(v) < d.step / 1000 ? '#7d8397' : grid,
        'stroke-width': v === 0 ? 1.3 : 0.75,
        'stroke-dasharray': v === 0 ? '4 4' : '2 5',
      });
      text(svg, x(v), bottom + 21, tickNum(v, d.step), {
        'text-anchor': 'middle',
        class: 'cc-number',
        'font-size': 10,
      });
    });
    const unit = paired ? 'Change in ' + (m.unit || 'score').toLowerCase() : m.unit || 'Score';
    text(svg, (left + right) / 2, bottom + 42, trim(unit, right - left, 11), {
      'text-anchor': 'middle',
      'font-size': 10.6,
      fill: fg,
    });
    let total = 0,
      shown = 0;
    lanes.forEach(({ group, rows, mean }, k) => {
      const y0 = top + k * laneHeight,
        center = y0 + laneHeight * 0.58;
      make(svg, 'rect', {
        x: left,
        y: y0 + 2,
        width: right - left,
        height: Math.max(1, laneHeight - 5),
        rx: 4,
        fill: group.color,
        'fill-opacity': 0.045,
        'pointer-events': 'none',
      });
      line(svg, left, center, right, center, {
        stroke: group.color,
        'stroke-opacity': 0.23,
        'stroke-width': 0.7,
      });
      const name = trim(group.name, Math.max(50, (right - left) * 0.4), 12);
      marker(svg, left + 7, y0 + 14, 3.2, group.color, k);
      text(svg, left + 17, y0 + 18, name, { 'font-size': 11.5, 'font-weight': 650, fill: fg });
      text(svg, right - 3, y0 + 18, 'n = ' + rows.length, {
        'text-anchor': 'end',
        'font-size': 10.5,
        class: 'cc-number',
        fill: subdued,
      });
      total += rows.length;
      if (!rows.length) {
        text(svg, (left + right) / 2, center + 3, paired ? 'No complete pairs' : 'No observations', {
          'text-anchor': 'middle',
          'font-size': 11,
        });
        return;
      }
      const show = sampled(rows, dotLimit, m.selectedId)
        .slice()
        .sort((p, q) => p.value - q.value || String(p.id).localeCompare(String(q.id)));
      shown += show.length;
      // Jitter acts only vertically. Positions on the score axis retain the measured value.
      const occupied = [];
      const limit = Math.max(4, Math.floor((laneHeight * 0.32 - 7) / 8));
      const offsets = [0];
      for (let z = 1; z <= limit; z++) offsets.push(-z * 8, z * 8);
      for (const p of show) {
        const xx = x(p.value),
          recent = occupied.filter((q) => Math.abs(q.x - xx) < 10);
        const scoring = (off) =>
          recent.reduce((sum, q) => sum + Math.max(0, 100 - ((q.x - xx) ** 2 + (q.off - off) ** 2)), 0);
        const offset = offsets.reduce(
          (best, off) => (scoring(off) < scoring(best) ? off : best),
          offsets[0],
        );
        occupied.push({ x: xx, off: offset });
        const yy = center + offset,
          g = make(svg, 'g');
        make(g, 'circle', { cx: xx, cy: yy, r: 8, fill: 'transparent' });
        const pt = marker(g, xx, yy, 4.6, group.color, 0, {
          stroke: '#ffffff',
          'stroke-width': 1.1,
          opacity: 0.91,
          class: 'cc-dot',
        });
        make(g, 'circle', {
          cx: xx,
          cy: yy,
          r: 8.2,
          fill: 'none',
          stroke: group.color,
          'stroke-width': 1.15,
          opacity: 0,
          class: 'cc-focus',
        });
        const msg =
          p.id + ' · ' + group.name + ' · ' + (paired ? 'Change ' : 'Score ') + num(p.value, paired);
        bindPerson(g, p.row, pt, msg);
      }
      const xx = x(mean),
        ym = y0 + laneHeight - 12;
      const gm = make(svg, 'g');
      line(gm, xx, center - 8, xx, ym - 6, {
        stroke: group.color,
        'stroke-dasharray': '2 4',
        'stroke-opacity': 0.65,
        'pointer-events': 'none',
      });
      const mk = marker(gm, xx, ym, 5.8, group.color, 1, { stroke: '#ffffff', 'stroke-width': 1.2 });
      make(gm, 'circle', {
        cx: xx,
        cy: ym,
        r: 8.6,
        fill: 'none',
        stroke: fg,
        opacity: 0,
        class: 'cc-focus',
      });
      const labelRight = xx < right - 67;
      text(gm, xx + (labelRight ? 10 : -10), ym + 3.4, num(mean, paired), {
        'text-anchor': labelRight ? 'start' : 'end',
        class: 'cc-number',
        'font-size': 10.4,
        'font-weight': 650,
        fill: group.color,
      });
      meanTarget(
        gm,
        group.name +
          ' · Mean ' +
          (paired ? 'change ' : 'score ') +
          num(mean, paired) +
          ' · n = ' +
          rows.length,
        mk,
      );
    });
    return { total, shown };
  }

  function disposeContrastCanvas(container) {
    const prior = states.get(container);
    if (!prior) return;
    prior.observer?.disconnect();
    if (prior.frame) cancelAnimationFrame(prior.frame);
    states.delete(container);
  }
  function renderContrastCanvas(container, model, actions = {}) {
    if (!container) return;
    let state = states.get(container);
    if (!state) {
      state = { container, lastWidth: 0, frame: 0 };
      states.set(container, state);
      if (typeof ResizeObserver === 'function') {
        state.observer = new ResizeObserver(() => {
          if (
            !container.isConnected ||
            Math.floor(container.clientWidth) === state.lastWidth ||
            state.frame
          )
            return;
          state.frame = requestAnimationFrame(() => {
            state.frame = 0;
            if (states.has(container)) paint(state);
          });
        });
        state.observer.observe(container);
      }
    }
    state.m = model;
    state.a = actions;
    state.data = prep(model);
    paint(state);
  }
  window.renderContrastCanvas = renderContrastCanvas;
  window.disposeContrastCanvas = disposeContrastCanvas;
})();
