(() => {
  'use strict';

  const ns = 'http://www.w3.org/2000/svg';
  const finite = Number.isFinite;
  const limit = 36;
  const sessions = new WeakMap();
  const focusHandlers = new WeakMap();
  const color = ['#2969d5', '#b86120'];

  function svg(tag, attributes = {}, value) {
    const node = document.createElementNS(ns, tag);
    for (const [name, attribute] of Object.entries(attributes)) node.setAttribute(name, attribute);
    if (value != null) node.textContent = String(value);
    return node;
  }

  function add(parent, tag, attributes, value) {
    const node = svg(tag, attributes, value);
    parent.append(node);
    return node;
  }

  function word(parent, x, y, value, cls, anchor = 'start') {
    return add(parent, 'text', { x, y, class: cls, 'text-anchor': anchor }, value);
  }

  function fitWord(parent, x, y, value, cls, room, anchor = 'middle') {
    const label = word(parent, x, y, value, cls, anchor);
    const original = String(value ?? '');
    if (label.getComputedTextLength() > room && original.length > 2) {
      let a = 0;
      let b = original.length;
      while (b - a > 1) {
        const mid = Math.floor((a + b) / 2);
        label.textContent = original.slice(0, mid) + '…';
        if (label.getComputedTextLength() <= room) a = mid;
        else b = mid;
      }
      label.textContent = original.slice(0, a) + '…';
      add(label, 'title', {}, original);
    }
    return label;
  }

  function rule(parent, x1, y1, x2, y2, cls, more) {
    return add(parent, 'line', { x1, y1, x2, y2, class: cls, ...more });
  }

  function number(value, signed = false, precision = 4) {
    if (!finite(value)) return '—';
    if (!value) return '0';
    const magnitude = Math.abs(value);
    const display = magnitude >= 1e6 || magnitude < 0.0001
      ? Number(magnitude.toPrecision(precision)).toExponential()
      : magnitude.toLocaleString('en-US', { maximumSignificantDigits: precision });
    return (value < 0 ? '−' : signed ? '+' : '') + display;
  }

  function exact(value) {
    return String(value).replace(/^-/, '−');
  }

  function numericScale(values, includeZero) {
    let min = includeZero ? 0 : Infinity;
    let max = includeZero ? 0 : -Infinity;
    for (const value of values) {
      if (finite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
    }
    if (!finite(min) || !finite(max)) return null;
    if (min === max) {
      const padding = min === 0 ? 1 : Math.abs(min) * 0.08 || Number.MIN_VALUE;
      min -= padding; max += padding;
    }
    const width = max - min;
    if (!(width > 0) || !finite(width)) return null;
    const rough = width / 4 || width;
    const power = 10 ** Math.floor(Math.log10(rough)) || Number.MIN_VALUE;
    const step = ([1, 2, 2.5, 5, 10].find((x) => x >= rough / power) || 10) * power;
    const first = Math.floor(min / step + 1e-12);
    const last = Math.ceil(max / step - 1e-12);
    let low = first * step;
    let high = last * step;
    if (!finite(low) || !finite(high) || high <= low) return null;
    const ticks = [];
    for (let i = 0; i <= last - first && i < 9; i++) ticks.push((first + i) * step);
    const pad = (high - low) * 0.055;
    low -= pad; high += pad;
    const tickText = (v) => {
      const rounded = +(v.toPrecision(14));
      const digits = Math.max(0, Math.ceil(-Math.log10(step)) + (step / power === 2.5 ? 1 : 0));
      if (digits > 7 || Math.abs(v) >= 1e7) return number(rounded, false, 7);
      return (rounded < 0 ? '−' : '') + Math.abs(rounded).toLocaleString('en-US', { maximumFractionDigits: digits });
    };
    return { ticks, tickText, y: (value, top, bottom) => bottom - (value - low) / (high - low) * (bottom - top) };
  }

  function hash(value) {
    let h = 2166136261;
    for (const ch of String(value)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
    h ^= h >>> 16;
    h = Math.imul(h, 0x85ebca6b);
    return (h ^ (h >>> 13)) >>> 0;
  }

  function sample(records, paired) {
    if (records.length <= limit) return records.slice();
    const retained = new Set();
    for (const key of paired ? ['baselineValue', 'followupValue'] : ['followupValue']) {
      let low = records[0], high = records[0];
      for (const p of records) {
        if (p[key] < low[key]) low = p;
        if (p[key] > high[key]) high = p;
      }
      retained.add(low); retained.add(high);
    }
    const randomOrder = records.map((p) => ({ p, order: hash(p.id) }))
      .sort((a, b) => a.order - b.order || String(a.p.id).localeCompare(String(b.p.id)));
    for (const { p } of randomOrder) { if (retained.size === limit) break; retained.add(p); }
    return [...retained];
  }

  function prepare(config) {
    const arms = Array.isArray(config.arms) ? config.arms : [];
    const haveRecords = Array.isArray(config.records);
    const paired = Boolean(config.longitudinal && haveRecords);
    const groups = arms.map((arm, index) => {
      const records = haveRecords ? config.records.filter((p) => p.arm === arm.name && p.contributes === true &&
        finite(p.followupValue) && (!paired || finite(p.baselineValue))) :
        (arm.values || []).flatMap((v, i) => finite(v) ? [{ id: arm.ids?.[i], followupValue: v }] : []);
      const ix = arm.colorIndex ?? arm.armIndex ?? index;
      return { name: arm.name, color: color[Math.abs(ix) % color.length] || color[0], count: records.length,
        participants: sample(records, paired).sort((a, b) => String(a.id ?? '').localeCompare(String(b.id ?? ''))) };
    });
    const drawn = groups.reduce((n, g) => n + g.participants.length, 0);
    const total = groups.reduce((n, g) => n + g.count, 0);
    const armNames = new Set(arms.map((arm) => arm.name));
    const excluded = haveRecords ? config.records.filter((p) => armNames.has(p.arm)).length - total : 0;
    const interval = finite(config.estimate) && Array.isArray(config.interval) && config.interval.length === 2 &&
      config.interval.every(finite) && config.interval[0] <= config.interval[1] ? config.interval : null;
    const raw = groups.flatMap((g) => g.participants.flatMap((p) => paired
      ? [p.baselineValue, p.followupValue] : [p.followupValue]));
    const rawScale = raw.length ? numericScale(raw, Boolean(config.longitudinal && !paired)) : null;
    const effectScale = finite(config.estimate) ? numericScale([config.estimate, ...(interval || [])], true) : null;
    const between = groups.length > 1;
    const estimand = config.longitudinal ? (between ? 'Difference in change' : 'Mean paired change')
      : (between ? 'Between-arm difference' : 'Participant mean');
    return { groups, paired, drawn, total, excluded, interval, rawScale, effectScale, estimand };
  }

  // A small horizontal displacement separates coincident observations on each categorical visit.
  function positions(readings, center, y, room) {
    const ordered = readings.map((v, i) => ({ i, y: y(v) })).sort((a, b) => a.y - b.y || a.i - b.i);
    const result = Array(readings.length);
    const placed = [];
    const lanes = [0];
    for (let dx = 7; dx <= Math.min(35, room); dx += 7) lanes.push(dx, -dx);
    for (const point of ordered) {
      const near = placed.filter((p) => Math.abs(p.y - point.y) < 8.5);
      let best = lanes[0];
      let bestDistance = -1;
      for (const lane of lanes) {
        const distance = near.reduce((d, p) => Math.min(d, Math.hypot(p.dx - lane, p.y - point.y)), Infinity);
        if (distance > bestDistance) { best = lane; bestDistance = distance; }
        if (distance >= 8.5) break;
      }
      placed.push({ dx: best, y: point.y }); result[point.i] = center + best;
    }
    return result;
  }

  function drawAxis(layer, scale, box, effect) {
    if (!scale) return;
    for (const value of scale.ticks) {
      const y = scale.y(value, box.top, box.bottom);
      rule(layer, box.left, y, box.right, y, effect && value === 0 ? 'sce-zero' : 'sce-grid');
      rule(layer, box.left - 4, y, box.left, y, 'sce-tick-line');
      word(layer, box.left - 8, y + 4, scale.tickText(value), 'sce-tick', 'end');
    }
    rule(layer, box.left, box.top, box.left, box.bottom, 'sce-axis-line');
    rule(layer, box.left, box.bottom, box.right, box.bottom, 'sce-axis-line');
  }

  function readingBox(width, scale) {
    const labelWidth = Math.max(0, ...(scale?.ticks || []).map((v) => scale.tickText(v).length));
    return {
      left: Math.max(42, 11 + labelWidth * 7),
      right: width - 12,
      top: 86,
      bottom: Math.max(224, Math.min(350, window.innerHeight * 0.32)),
    };
  }

  function drawRaw(chart, config, model, box, items) {
    const { paired, groups, rawScale } = model;
    word(chart, 5, 20, config.longitudinal && !paired ? 'Participant changes' : 'Participant measurements', 'sce-heading');
    fitWord(chart, 5, 39, config.unit || 'Recorded score', 'sce-subtitle', box.right - 10, 'start');
    drawAxis(chart, rawScale, box, false);
    if (!groups.length || !rawScale) {
      word(chart, (box.left + box.right) / 2, (box.top + box.bottom) / 2,
        paired ? 'No complete pairs' : 'No measured participants', 'sce-empty', 'middle');
    }
    const span = (box.right - box.left) / (groups.length || 1);
    let personIndex = 0;
    groups.forEach((group, index) => {
      const start = box.left + index * span;
      const mid = start + span / 2;
      const base = start + span * 0.24;
      const last = paired ? start + span * 0.76 : mid;
      if (index) rule(chart, start, box.top, start, box.bottom, 'sce-arm-divider');
      const section = add(chart, 'g', { role: 'group', 'aria-label': group.name + ' participant observations',
        class: 'sce-arm', style: '--arm:' + group.color });
      const links = add(section, 'g', { class: 'sce-lines' });
      const dots = add(section, 'g', {});
      if (rawScale) {
        const y = (v) => rawScale.y(v, box.top, box.bottom);
        const people = group.participants;
        const reference = paired ? positions(people.map((p) => p.baselineValue), base, y, span * 0.14) : [];
        const comparison = positions(people.map((p) => p.followupValue), last, y, span * (paired ? 0.14 : 0.32));
        people.forEach((p, i) => {
          const subject = personIndex++;
          const id = p.id;
          const actionable = id !== undefined && id !== null && typeof config.onSelect === 'function';
          function register(node, detail, x, yy, kind) {
            node.dataset.subject = String(subject);
            if (actionable) {
              node.dataset.item = String(items.length);
              node.setAttribute('role', 'button');
              node.setAttribute('aria-label', 'Inspect ' + detail);
              node.setAttribute('tabindex', items.length ? '-1' : '0');
              items.push({ node, id, detail, x, y: yy, subject, kind });
            } else add(node, 'title', {}, detail);
          }
          const idText = id == null ? group.name : String(id);
          const y1 = paired ? y(p.baselineValue) : 0;
          const y2 = y(p.followupValue);
          if (paired) {
            const d = 'M' + reference[i] + ' ' + y1 + 'L' + comparison[i] + ' ' + y2;
            const line = add(links, 'g', { class: 'sce-link' });
            add(line, 'path', { d, class: 'sce-line-hit' });
            add(line, 'path', { d, class: 'sce-person-line' });
            register(line, idText + ' · ' + config.baseline + ' ' + exact(p.baselineValue) + ' → ' +
              config.visit + ' ' + exact(p.followupValue), (reference[i] + comparison[i]) / 2, (y1 + y2) / 2, 'pair');
          }
          function point(x, yy, value, visit, referencePoint) {
            const positions = referencePoint ? reference : comparison;
            const scoreKey = referencePoint ? 'baselineValue' : 'followupValue';
            let hitRadius = 12;
            people.forEach((other, j) => {
              if (i !== j) {
                const distance = Math.hypot(x - positions[j], yy - y(other[scoreKey]));
                hitRadius = Math.min(hitRadius, Math.max(4.3, distance - .7));
              }
            });
            const node = add(dots, 'g', { class: 'sce-observation' });
            add(node, 'circle', { cx: x, cy: yy, r: hitRadius, class: 'sce-point-hit' });
            add(node, 'circle', { cx: x, cy: yy, r: 4.3,
              class: 'sce-arm-mark ' + (referencePoint ? 'sce-open' : 'sce-solid') });
            register(node, idText + ' · ' + visit + ' · ' + exact(value), x, yy, referencePoint ? 'reference' : 'comparison');
          }
          if (paired) point(reference[i], y1, p.baselineValue, config.baseline, true);
          const visitLabel = !paired && config.longitudinal ? 'Paired change' : config.visit;
          point(comparison[i], y2, p.followupValue, visitLabel, false);
        });
      }
      const visitWidth = span * (paired ? 0.47 : 0.92);
      if (paired) {
        rule(section, base, box.bottom, base, box.bottom + 4, 'sce-tick-line');
        fitWord(section, base, box.bottom + 21, config.baseline, 'sce-visit', visitWidth);
      }
      rule(section, last, box.bottom, last, box.bottom + 4, 'sce-tick-line');
      fitWord(section, last, box.bottom + 21, config.longitudinal && !paired ? 'Change' : config.visit,
        'sce-visit', visitWidth);
      fitWord(section, mid, box.top - 24, group.name, 'sce-arm-label', span - 10);
      const count = group.participants.length < group.count ? group.participants.length + ' of ' + group.count.toLocaleString() + ' shown' :
        'n = ' + group.count.toLocaleString() + (config.longitudinal ? (group.count === 1 ? ' pair' : ' pairs') : '');
      word(section, mid, box.top - 7, count, 'sce-arm-n', 'middle');
    });
  }

  function effectHeader(config, model) {
    const summary = document.createElement('div');
    summary.className = 'sce-summary';
    const title = document.createElement('h3');
    title.textContent = model.estimand;
    const value = document.createElement('strong');
    value.className = 'sce-summary-value';
    const detail = document.createElement('span');
    detail.className = 'sce-summary-detail';
    const signed = config.longitudinal || model.groups.length > 1;
    let precision = 4;
    if (model.interval) {
      const [low, high] = model.interval;
      while (precision < 15 && low !== high && number(low, signed, precision) === number(high, signed, precision)) precision++;
      detail.textContent = '95% CI  ' + number(low, signed, precision) + ' to ' + number(high, signed, precision);
    } else if (finite(config.estimate)) {
      detail.textContent = config.descriptive ? 'Descriptive estimate' : '95% CI unavailable';
    } else {
      detail.textContent = config.status || 'Estimate unavailable';
    }
    value.textContent = number(config.estimate, signed, precision);
    const output = document.createElement('div');
    output.className = 'sce-summary-reading';
    output.append(value, detail);
    summary.append(title, output);
    return summary;
  }

  function drawInterval(chart, width, config, model) {
    const scale = model.effectScale;
    if (!scale) return;
    const axisMargin = Math.max(20, 5 + Math.max(...scale.ticks.map(v => scale.tickText(v).length)) * 3.5);
    // The scale's first argument after the value is the high-end coordinate.
    const x = v => scale.y(v, width - axisMargin, axisMargin);
    const y = 16;
    rule(chart, axisMargin, y, width - axisMargin, y, 'sce-summary-track');
    for (const v of scale.ticks) {
      const xx = x(v);
      rule(chart, xx, v === 0 ? 3 : y - 2, xx, v === 0 ? 29 : y + 4,
        v === 0 ? 'sce-zero' : 'sce-summary-tick');
      word(chart, xx, 43, scale.tickText(v), 'sce-tick', 'middle');
    }
    if (model.interval) {
      const [low, high] = model.interval;
      const ci = add(chart, 'g', { class: 'sce-ci', role: 'img',
        'aria-label': '95% confidence interval ' + exact(low) + ' to ' + exact(high) });
      rule(ci, x(low), y, x(high), y, 'sce-whisker');
      for (const v of [low, high]) rule(ci, x(v), y - 5, x(v), y + 5, 'sce-whisker');
    }
    const center = x(config.estimate);
    add(chart, 'path', { d: 'M' + center + ' ' + (y - 6) + 'l6 6-6 6-6-6z',
      class: 'sce-estimate', role: 'img', 'aria-label': 'Estimate ' + exact(config.estimate) });
  }

  function wire(chart, figure, tooltip, items, config) {
    let active = null;
    let pointed = null;
    const get = (target) => {
      const hit = target?.closest?.('[data-item]');
      return hit ? items[Number(hit.dataset.item)] : null;
    };
    const nodes = new Map();
    for (const item of items) {
      if (!nodes.has(item.subject)) nodes.set(item.subject, []);
      nodes.get(item.subject).push(item.node);
    }
    function highlight(item, event) {
      if (active !== item) {
        for (const node of nodes.get(active?.subject) || []) node.classList.remove('sce-emphasis');
        if (item) for (const node of nodes.get(item.subject)) node.classList.add('sce-emphasis');
        active = item;
        tooltip.hidden = !item;
        if (item) tooltip.textContent = item.detail;
      }
      if (!item) return;
      const rect = figure.getBoundingClientRect();
      const x = event ? event.clientX - rect.left : item.x;
      const y = event ? event.clientY - rect.top : item.y;
      tooltip.style.left = Math.max(4, Math.min(x + 12, rect.width - tooltip.offsetWidth - 4)) + 'px';
      tooltip.style.top = Math.max(2, y - tooltip.offsetHeight - 10) + 'px';
    }
    chart.onpointermove = (event) => { pointed = get(event.target); highlight(pointed || get(document.activeElement), pointed ? event : null); };
    chart.onpointerleave = () => { pointed = null; highlight(get(document.activeElement)); };
    focusHandlers.get(chart)?.abort();
    const controller = new AbortController();
    focusHandlers.set(chart, controller);
    chart.addEventListener('focusin', (event) => {
      const item = get(event.target);
      if (!item) return;
      chart.querySelector('[tabindex="0"]')?.setAttribute('tabindex', '-1');
      item.node.setAttribute('tabindex', '0');
      highlight(item);
    }, { signal: controller.signal });
    chart.addEventListener('focusout', (event) => highlight(get(event.relatedTarget) || pointed),
      { signal: controller.signal });
    chart.onclick = (event) => {
      const item = get(event.target);
      if (item) { item.node.focus({ preventScroll: true }); config.onSelect(item.id); }
    };
    chart.onkeydown = (event) => {
      const item = get(event.target);
      if (!item) return;
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); config.onSelect(item.id); return; }
      if (event.key === 'Escape') { highlight(null); return; }
      const index = items.indexOf(item);
      let target;
      if (event.key === 'Home') target = items[0];
      if (event.key === 'End') target = items[items.length - 1];
      if (event.key === 'ArrowRight') target = items[Math.min(items.length - 1, index + 1)];
      if (event.key === 'ArrowLeft') target = items[Math.max(0, index - 1)];
      if (event.key === 'ArrowDown') target = items.find((p) => p.subject > item.subject && p.kind === item.kind) || item;
      if (event.key === 'ArrowUp') target = items.slice(0, index).reverse().find((p) => p.subject < item.subject && p.kind === item.kind) || item;
      if (target) { event.preventDefault(); target.node.focus({ preventScroll: true }); }
    };
  }

  function render(container, config) {
    if (!container || !config) return;
    sessions.get(container)?.disconnect();
    const model = prepare(config);
    const figure = document.createElement('figure');
    figure.className = 'sce';
    figure.setAttribute('aria-label', [config.feature, model.estimand].filter(Boolean).join(' · '));
    const chart = svg('svg', { class: 'sce-svg', role: 'group',
      'aria-label': 'Participant measurements and ' + model.estimand.toLowerCase(),
      'aria-description': 'Arrow keys explore the observations. Enter opens a participant record.' });
    const tooltip = document.createElement('span');
    tooltip.className = 'sce-point-tip';
    tooltip.setAttribute('aria-hidden', 'true'); tooltip.hidden = true;
    const caption = document.createElement('figcaption');
    caption.className = 'sce-caption';
    const notes = [];
    if (model.paired) notes.push('Lines join measurements from the same participant.');
    if (model.excluded) notes.push(model.excluded.toLocaleString() + (model.excluded === 1 ? ' participant excluded.' : ' participants excluded.'));
    if (model.drawn < model.total) notes.push('Trace sample includes observed extremes. The estimate uses all contributing participants.');
    if (!finite(config.estimate) && config.status) notes.push(config.status);
    caption.textContent = notes.join(' ');
    const summary = effectHeader(config, model);
    const intervalPlot = svg('svg', { class: 'sce-interval-strip', role: 'group',
      'aria-label': model.estimand + ' on the recorded measurement scale' });
    intervalPlot.hidden = !model.effectScale;
    if (model.effectScale) summary.append(intervalPlot);
    figure.append(summary, chart, tooltip, caption);
    container.replaceChildren(figure);
    let previousLayout = '';
    function draw() {
      const width = Math.max(230, Math.round(figure.getBoundingClientRect().width));
      const sizeKey = width + 'x' + window.innerHeight;
      if (previousLayout === sizeKey) return;
      const prior = document.activeElement?.closest?.('[data-item]');
      const label = chart.contains(prior) ? prior?.getAttribute('aria-label') : null;
      previousLayout = sizeKey;
      const box = readingBox(width, model.rawScale);
      intervalPlot.setAttribute('viewBox', [0, 0, width, 48].join(' '));
      intervalPlot.replaceChildren();
      drawInterval(intervalPlot, width, config, model);
      const items = [];
      chart.setAttribute('viewBox', [0, 0, width, box.bottom + 33].join(' '));
      chart.setAttribute('height', box.bottom + 33);
      chart.replaceChildren();
      tooltip.hidden = true;
      drawRaw(chart, config, model, box, items);
      wire(chart, figure, tooltip, items, config);
      if (label) items.find((item) => item.node.getAttribute('aria-label') === label)?.node.focus({ preventScroll: true });
    }
    draw();
    if (typeof ResizeObserver !== 'undefined') {
      const disconnect = () => { observer.disconnect(); window.removeEventListener('resize', resize); };
      const resize = () => { if (figure.isConnected) draw(); else disconnect(); };
      const observer = new ResizeObserver(resize);
      observer.observe(figure);
      window.addEventListener('resize', resize, { passive: true });
      sessions.set(container, { disconnect });
    }
    if (typeof chart.animate === 'function' && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      chart.animate([{ opacity: 0.48, transform: 'translateY(2px)' }, { opacity: 1, transform: 'none' }],
        { duration: 145, easing: 'ease-out' });
    }
  }

  window.SctrialEvidence = Object.freeze({ render });
})();
