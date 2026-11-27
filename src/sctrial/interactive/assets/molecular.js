(() => {
  'use strict';

  const sessions = new WeakMap();
  const ns = 'http://www.w3.org/2000/svg';
  const finite = Number.isFinite;
  const compare = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
  const hues = ['#638fff', '#ffb267', '#9caaff', '#e99ede'];
  const ramp = ['#143b49', '#1a6772', '#2f9c9c', '#69cdb1', '#dcf3a0'];
  const kinds = { genes: 'gene', pathways: 'pathway', modules: 'module' };
  const names = { genes: 'Gene expression', pathways: 'Pathways', modules: 'Module scores' };
  let sequence = 0;

  function html(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function node(parent, tag, attrs = {}, text) {
    const el = document.createElementNS(ns, tag);
    for (const [key, v] of Object.entries(attrs)) el.setAttribute(key, v);
    if (text != null) el.textContent = text;
    parent.append(el);
    return el;
  }

  function label(parent, x, y, text, cls = 'ml-label', anchor = 'start') {
    return node(parent, 'text', { x, y, class: cls, 'text-anchor': anchor }, text);
  }

  function clipped(parent, x, y, text, cls, room, anchor = 'start') {
    const el = label(parent, x, y, text, cls, anchor);
    const original = String(text ?? '');
    if (el.getComputedTextLength() > room) {
      let low = 0,
        high = original.length;
      while (high - low > 1) {
        const mid = (high + low) >> 1;
        el.textContent = original.slice(0, mid) + '…';
        if (el.getComputedTextLength() > room) high = mid;
        else low = mid;
      }
      el.textContent = original.slice(0, low) + '…';
      node(el, 'title', {}, original);
    }
    return el;
  }

  function line(parent, x1, y1, x2, y2, cls, extra = {}) {
    return node(parent, 'line', { x1, y1, x2, y2, class: cls, ...extra });
  }

  function button(labelText, action, cls = '') {
    const el = html('button', cls, labelText);
    el.type = 'button';
    el.addEventListener('click', action);
    return el;
  }

  function selectControl(name, entries, value, cb) {
    const box = html('label', 'ml-control');
    box.append(html('span', '', name));
    const input = html('select');
    input.setAttribute('aria-label', name);
    for (const entry of entries) {
      const option = html('option', '', entry.label);
      option.value = entry.id;
      input.append(option);
    }
    input.value = value;
    input.addEventListener('change', () => cb?.(input.value));
    box.append(input);
    return box;
  }

  const featureKind = (f) => f.kind || 'score';
  const nameOf = (f) => f.label || f.id;
  const measurement = (r, id) =>
    r && (r.cells == null || r.cells > 0) && finite(r.scores?.[id]) ? r.scores[id] : null;
  const exact = (value) => String(value).replace(/^-/, '−');
  const fmt = (value, precision = 4) =>
    finite(value)
      ? Number(value.toPrecision(precision)).toLocaleString('en-US', {
          maximumSignificantDigits: precision,
        })
      : '—';
  const unitOf = (f) =>
    f.unit ||
    (f.kind === 'gene'
      ? 'Recorded expression'
      : f.kind === 'pathway'
        ? 'Pathway score'
        : f.kind === 'module'
          ? 'Module score'
          : 'Recorded score');
  const key = (a, b) => JSON.stringify([a, b]);

  function studyIndex(study) {
    const obs = study.observations || [];
    const people = new Map(
      (study.participants || []).map((p) => [
        String(p.id),
        { id: String(p.id), arm: String(p.arm ?? '') },
      ]),
    );
    const visits = [...(study.visits || [])];
    const pops = [...(study.populations || [])];
    const byPopulation = new Map();
    for (const r of obs) {
      const id = String(r.id);
      if (!people.has(id)) people.set(id, { id, arm: String(r.arm ?? '') });
      if (!study.visits && !visits.includes(r.visit)) visits.push(r.visit);
      if (!study.populations && !pops.includes(r.population)) pops.push(r.population);
      if (!byPopulation.has(r.population)) byPopulation.set(r.population, new Map());
      byPopulation.get(r.population).set(key(id, r.visit), r);
    }
    const arms = [...new Set([...people.values()].map((p) => p.arm))];
    const participants = [...people.values()].sort(
      (a, b) => arms.indexOf(a.arm) - arms.indexOf(b.arm) || compare.compare(a.id, b.id),
    );
    return { participants, arms, visits, pops, byPopulation };
  }

  function hash(text) {
    let n = 2166136261;
    for (const ch of String(text)) n = Math.imul(n ^ ch.charCodeAt(0), 16777619);
    return (n ^ (n >>> 15)) >>> 0;
  }

  function colorAt(t) {
    const pos = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
    const i = Math.min(ramp.length - 2, Math.floor(pos));
    const rgb = [1, 3, 5].map((k) =>
      Math.round(
        parseInt(ramp[i].slice(k, k + 2), 16) * (i + 1 - pos) +
          parseInt(ramp[i + 1].slice(k, k + 2), 16) * (pos - i),
      ),
    );
    return 'rgb(' + rgb.join(',') + ')';
  }

  function domain(values) {
    let min = Infinity,
      max = -Infinity;
    for (const value of values)
      if (finite(value)) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    return Number.isFinite(min) ? [min, max] : null;
  }

  function ticks(extent) {
    let [lo, hi] = extent;
    if (lo === hi) {
      const delta = Math.abs(lo) * 0.07 || 1;
      lo -= delta;
      hi += delta;
    }
    const width = hi - lo,
      rough = width / 4;
    if (!finite(rough) || rough <= 0)
      return {
        values: [lo, hi],
        y: (v, t, b) => b - ((v / 2 - lo / 2) / (hi / 2 - lo / 2)) * (b - t),
        format: exact,
      };
    const power = 10 ** Math.floor(Math.log10(rough));
    const step = ([1, 2, 2.5, 5, 10].find((n) => n >= rough / power) || 10) * power;
    const a = Math.floor(lo / step),
      b = Math.ceil(hi / step);
    const values = Array.from(
      { length: Math.max(2, Math.min(8, b - a + 1)) },
      (_, i) => (a + i) * step,
    );
    const bottom = a * step,
      top = b * step;
    let precision = 4;
    while (precision < 15 && new Set(values.map((v) => fmt(v, precision))).size < values.length)
      precision++;
    return {
      values,
      y: (v, t, b) => b - ((v - bottom) / (top - bottom || 1)) * (b - t),
      format: (v) => fmt(v, precision),
    };
  }

  function defs(chart, uid) {
    const defs = node(chart, 'defs');
    const pattern = node(defs, 'pattern', {
      id: uid + '-na',
      patternUnits: 'userSpaceOnUse',
      width: 6,
      height: 6,
    });
    node(pattern, 'rect', { width: 6, height: 6, fill: '#112d35' });
    node(pattern, 'path', {
      d: 'M-1,1L1,-1M0,6L6,0M5,7L7,5',
      stroke: '#35535c',
      'stroke-width': 1,
    });
    const gradient = node(defs, 'linearGradient', { id: uid + '-ramp' });
    ramp.forEach((stop, i) =>
      node(gradient, 'stop', { offset: (i / (ramp.length - 1)) * 100 + '%', 'stop-color': stop }),
    );
    const core = node(defs, 'radialGradient', {
      id: uid + '-core',
      cx: '38%',
      cy: '26%',
      r: '80%',
    });
    for (const [offset, v] of [
      ['0%', '#397d74'],
      ['58%', '#163f46'],
      ['100%', '#09272e'],
    ])
      node(core, 'stop', { offset, 'stop-color': v });
  }

  function inspect(s, item, detail, action, x, y) {
    item.dataset.mlItem = String(s.items.length);
    item.setAttribute('tabindex', s.items.length ? '-1' : '0');
    item.setAttribute('role', 'button');
    item.setAttribute('aria-label', detail);
    s.items.push({ item, detail, action, x, y });
  }

  function interactions(s) {
    const { chart, figure, tip } = s;
    let hot = null,
      hover = null;
    const get = (target) => {
      const n = target?.closest?.('[data-ml-item]');
      return n ? s.items[Number(n.dataset.mlItem)] : null;
    };
    function show(item, event) {
      if (hot !== item) {
        hot?.item.classList.remove('ml-hot');
        item?.item.classList.add('ml-hot');
        hot = item;
        tip.hidden = !item;
        if (item) tip.textContent = item.detail;
      }
      if (!item) return;
      const box = figure.getBoundingClientRect(),
        plot = chart.getBoundingClientRect();
      const x = event ? event.clientX - box.left : item.x + plot.left - box.left;
      const y = event ? event.clientY - box.top : item.y + plot.top - box.top;
      tip.style.left = Math.max(3, Math.min(x + 10, box.width - tip.offsetWidth - 4)) + 'px';
      tip.style.top = Math.max(0, y - tip.offsetHeight - 10) + 'px';
    }
    chart.onpointermove = (e) => {
      hover = get(e.target);
      show(hover || get(document.activeElement), hover ? e : null);
    };
    chart.onpointerleave = () => {
      hover = null;
      show(get(document.activeElement));
    };
    chart.addEventListener(
      'focusin',
      (e) => {
        const i = get(e.target);
        if (!i) return;
        chart.querySelector('[tabindex="0"]')?.setAttribute('tabindex', '-1');
        i.item.setAttribute('tabindex', '0');
        show(i);
      },
      { signal: s.abort.signal },
    );
    chart.addEventListener('focusout', (e) => show(get(e.relatedTarget) || hover), {
      signal: s.abort.signal,
    });
    chart.onclick = (e) => {
      const i = get(e.target);
      if (i) {
        i.item.focus({ preventScroll: true });
        i.action();
      }
    };
    chart.onkeydown = (e) => {
      const i = get(e.target);
      if (!i) return;
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        i.action();
        return;
      }
      if (e.key === 'Escape') {
        hot?.item.classList.remove('ml-hot');
        hot = null;
        tip.hidden = true;
        return;
      }
      const a = s.items.indexOf(i);
      const index = {
        ArrowRight: a + 1,
        ArrowLeft: a - 1,
        ArrowDown: a + (s.columns || 1),
        ArrowUp: a - (s.columns || 1),
        Home: 0,
        End: s.items.length - 1,
      }[e.key];
      if (index !== undefined) {
        e.preventDefault();
        s.items[Math.min(s.items.length - 1, Math.max(0, index))].item.focus({
          preventScroll: true,
        });
      }
    };
  }

  function selectMark(s, text, node) {
    s.insight.textContent = text;
    s.chart.querySelectorAll('.ml-picked').forEach((el) => el.classList.remove('ml-picked'));
    if (node) node.classList.add('ml-picked');
  }

  function pager(s, title, page, count, size, change) {
    if (count <= size) return;
    const wrap = html('span', 'ml-pager');
    wrap.append(
      html(
        'span',
        '',
        title + ' ' + (page * size + 1) + '–' + Math.min(count, (page + 1) * size) + ' of ' + count,
      ),
    );
    for (const [delta, name, icon] of [
      [-1, 'Previous', '‹'],
      [1, 'Next', '›'],
    ]) {
      const b = button(icon, () => {
        change(page + delta);
        s.redraw(true);
      });
      b.setAttribute('aria-label', name + ' ' + title.toLowerCase());
      b.disabled = page + delta < 0 || (page + delta) * size >= count;
      wrap.append(b);
    }
    s.actions.append(wrap);
  }

  function genes(s, w) {
    const { chart, features, active, index, visit, pop, view } = s;
    const per = Math.max(1, Math.min(9, Math.floor((w - 115) / 83)));
    const perRow = w < 480 ? 12 : 14;
    view.col = Math.min(Math.floor((features.length - 1) / per), Math.max(0, view.col));
    view.row = Math.min(
      Math.max(0, Math.floor((index.participants.length - 1) / perRow)),
      Math.max(0, view.row),
    );
    const fs = features.slice(view.col * per, (view.col + 1) * per),
      people = index.participants.slice(view.row * perRow, (view.row + 1) * perRow);
    s.columns = fs.length;
    const left = Math.min(w * 0.28, 128),
      right = 12,
      top = 101,
      row = 31;
    const step = Math.min(fs.length >= 4 ? 230 : 158, (w - left - right) / fs.length),
      gridW = fs.length * step;
    const gridX = left + Math.max(0, (w - left - right - gridW) / 2);
    const h = top + row * people.length + 43;
    s.size(w, h);
    const rs = index.byPopulation.get(pop),
      at = (id, f) => measurement(rs?.get(key(id, visit)), f.id);
    const ranges = fs.map((f) => domain(index.participants.map((p) => at(p.id, f))));
    label(chart, 2, 20, 'Expression by participant', 'ml-chart-title');
    const uniformUnit = features.every((f) => unitOf(f) === unitOf(active));
    clipped(
      chart,
      2,
      41,
      pop + '  ·  ' + visit + '  ·  ' + (uniformUnit ? unitOf(active) : 'Feature-specific units'),
      'ml-note',
      w - 8,
    );
    fs.forEach((f, j) => {
      const x = gridX + (j + 0.5) * step,
        r = ranges[j];
      const count = index.participants.filter((p) => at(p.id, f) !== null).length;
      clipped(
        chart,
        x,
        top - 36,
        nameOf(f),
        f.id === active.id ? 'ml-gene ml-active' : 'ml-gene',
        step - 4,
        'middle',
      );
      let precision = 3;
      if (r && r[0] !== r[1])
        while (precision < 14 && fmt(r[0], precision) === fmt(r[1], precision)) precision++;
      const range = r
        ? r[0] === r[1]
          ? fmt(r[0], precision)
          : fmt(r[0], precision) + '–' + fmt(r[1], precision)
        : '—';
      clipped(chart, x, top - 21, range, 'ml-note', step - 4, 'middle');
      label(chart, x, top - 6, 'n = ' + count, 'ml-axis', 'middle');
    });
    const personX = gridX - left;
    label(chart, personX + 2, top - 8, 'Participant', 'ml-axis');
    people.forEach((p, i) => {
      const y = top + i * row;
      const armColor = hues[index.arms.indexOf(p.arm) % hues.length];
      node(chart, 'rect', {
        x: personX + 2,
        y: y + 8,
        width: 3,
        height: 13,
        rx: 1.5,
        fill: armColor,
      });
      clipped(chart, personX + 12, y + 13, p.id, 'ml-person', left - 15);
      clipped(chart, personX + 12, y + 26, p.arm, 'ml-person-arm', left - 15);
      fs.forEach((f, j) => {
        const x = gridX + j * step,
          v = at(p.id, f),
          r = ranges[j],
          constant = r && r[0] === r[1],
          t = v === null || constant ? 0.5 : (v - r[0]) / (r[1] - r[0]);
        const tile = node(chart, 'g', {
          'data-gene-cell': '',
          'data-id': p.id,
          'data-feature': f.id,
          'data-value': v === null ? '' : v,
        });
        node(tile, 'rect', {
          x: x + 2,
          y: y + 2,
          width: Math.max(2, step - 4),
          height: row - 4,
          rx: 3,
          class: 'ml-cell-shape',
          fill: v === null ? 'url(#' + s.uid + '-na)' : constant ? '#356269' : colorAt(t),
        });
        if (v !== null && step >= 52) {
          const txt = label(tile, x + step / 2, y + 19, fmt(v, 3), 'ml-cell-value', 'middle');
          txt.style.fill = !constant && t > 0.54 ? '#092d30' : '#f0fcf7';
        }
        const detail =
          p.id +
          ' · ' +
          p.arm +
          ' · ' +
          nameOf(f) +
          ' · ' +
          (v === null ? 'Missing' : exact(v) + ' ' + unitOf(f));
        inspect(s, tile, detail, () => selectMark(s, detail, tile), x + step / 2, y + row / 2);
      });
    });
    const footer = top + people.length * row + 23;
    label(chart, 2, footer, 'Within each gene', 'ml-axis');
    const bx = w < 400 ? 108 : 129;
    label(chart, bx, footer, 'Low', 'ml-note');
    const barW = Math.min(86, w * 0.2);
    node(chart, 'rect', {
      x: bx + 23,
      y: footer - 9,
      width: barW,
      height: 7,
      rx: 2,
      fill: 'url(#' + s.uid + '-ramp)',
    });
    label(chart, bx + barW + 30, footer, 'High', 'ml-note');
    if (w > 450) {
      const nx = bx + barW + 86;
      node(chart, 'rect', {
        x: nx,
        y: footer - 12,
        width: 12,
        height: 12,
        fill: 'url(#' + s.uid + '-na)',
        rx: 2,
      });
      label(chart, nx + 18, footer, 'Missing', 'ml-note');
    }
    s.caption.textContent =
      'Column labels give the observed expression range and measured participant count for each gene. Color is scaled within each gene. Hatching marks missing values.';
    pager(s, 'Participants', view.row, index.participants.length, perRow, (x) => {
      view.row = x;
    });
    pager(s, 'Genes', view.col, features.length, per, (x) => {
      view.col = x;
    });
    if (!people.length)
      label(chart, w / 2, 95, 'No participants in this study', 'ml-note', 'middle');
  }

  function membership(s, w) {
    const { chart, active, index, pop, visit, view } = s;
    const listed = [
      ...new Set(
        active.members
          .map(String)
          .map((g) => g.trim())
          .filter(Boolean),
      ),
    ];
    const size = w > 800 ? 8 : w > 490 ? 6 : 4;
    view.members = Math.min(
      Math.max(0, view.members),
      Math.max(0, Math.ceil(listed.length / size) - 1),
    );
    const genes = listed.slice(view.members * size, (view.members + 1) * size);
    const h = w < 500 ? 387 : Math.min(547, Math.max(410, (window.innerHeight || 800) - 325));
    s.size(w, h);
    label(chart, 2, 20, 'Gene-set membership', 'ml-chart-title');
    clipped(
      chart,
      2,
      41,
      nameOf(active) + '  ·  ' + listed.length + ' listed genes  ·  ' + unitOf(active),
      'ml-note',
      w - 5,
    );
    let legendX = 2;
    index.arms.forEach((arm, i) => {
      const room = Math.min(145, (w - 8) / Math.max(1, index.arms.length));
      node(chart, 'circle', { cx: legendX + 3, cy: 59, r: 3.4, fill: hues[i % hues.length] });
      clipped(chart, legendX + 13, 63, arm, 'ml-note', room - 18);
      legendX += room;
    });
    const cx = w / 2,
      cy = h * 0.54,
      rx = Math.min(w * 0.25, 277),
      ry = Math.min(h * 0.285, 145);
    for (const r of [0.45, 0.76, 1])
      node(chart, 'ellipse', { cx, cy, rx: rx * r, ry: ry * r, class: 'ml-orbit' });
    const match = new Map();
    for (const f of s.config.study.features || [])
      if (featureKind(f) === 'gene') {
        for (const sym of [f.id, nameOf(f), f.symbol].filter(Boolean))
          match.set(String(sym).toLowerCase(), f);
      }
    const obs = index.byPopulation.get(pop);
    const readings = (f) =>
      index.participants.flatMap((p) => {
        const v = measurement(obs?.get(key(p.id, visit)), f.id);
        return v === null ? [] : [{ ...p, v }];
      });
    genes.forEach((g, i) => {
      const angle = ((i + 0.27) * Math.PI * 2) / genes.length - Math.PI / 2,
        dx = Math.cos(angle),
        dy = Math.sin(angle);
      const x = cx + rx * dx,
        y = cy + ry * dy,
        dir = dx >= 0 ? 1 : -1;
      const f = match.get(g.toLowerCase()),
        values = f ? readings(f) : [],
        measured = values.length > 0;
      const edge = node(chart, 'path', {
        d:
          'M' +
          cx +
          ' ' +
          cy +
          'Q' +
          (cx + rx * dx * 0.33 - ry * dy * 0.09) +
          ' ' +
          (cy + ry * dy * 0.36 + rx * dx * 0.04) +
          ' ' +
          x +
          ' ' +
          y,
        class: 'ml-member-line',
        pathLength: 1,
        'data-membership': g,
        'aria-hidden': 'true',
      });
      const group = node(chart, 'g', {
        'data-member': g,
        class: 'ml-member' + (measured ? ' ml-measured' : ''),
      });
      node(group, 'circle', { cx: x, cy: y, r: 13, class: 'ml-hit' });
      node(group, 'circle', { cx: x, cy: y, r: 10, class: 'ml-member-ring' });
      node(group, 'circle', { cx: x, cy: y, r: 6, class: 'ml-member-node' });
      const mw = w >= 700 ? 135 : w >= 480 ? 106 : 70;
      const left = Math.max(3, Math.min(w - mw - 3, dir > 0 ? x + 15 : x - 15 - mw)),
        right = left + mw;
      clipped(
        group,
        dir > 0 ? left : right,
        y - 12,
        g,
        'ml-member-symbol',
        mw,
        dir > 0 ? 'start' : 'end',
      );
      const detail =
        g +
        ' · ' +
        nameOf(active) +
        ' member · ' +
        values.length +
        ' participants with measured expression at ' +
        visit +
        ' · ' +
        pop;
      inspect(
        s,
        group,
        detail,
        () => {
          selectMark(s, detail, group);
          chart
            .querySelectorAll('.ml-edge-picked')
            .forEach((e) => e.classList.remove('ml-edge-picked'));
          edge.classList.add('ml-edge-picked');
          if (f && measured && s.config.onAnalyze)
            s.insight.append(
              button('Analyze ' + nameOf(f), () => s.config.onAnalyze(f.id), 'ml-insight-link'),
            );
        },
        x,
        y,
      );
      group.onpointerenter = () => edge.classList.add('ml-edge-hot');
      group.onpointerleave = () => edge.classList.remove('ml-edge-hot');
      if (!measured) {
        clipped(
          chart,
          dir > 0 ? left : right,
          y + 6,
          'No expression',
          'ml-note',
          mw,
          dir > 0 ? 'start' : 'end',
        );
        return;
      }
      const sorted = values.slice().sort((a, b) => a.v - b.v),
        extent = [sorted[0].v, sorted[sorted.length - 1].v];
      const shown =
        sorted.length <= 32
          ? sorted
          : Array.from(
              { length: 32 },
              (_, j) => sorted[Math.round((j * (sorted.length - 1)) / 31)],
            );
      line(chart, left, y, right, y, 'ml-micro-axis');
      for (const item of shown) {
        const xx =
          extent[1] === extent[0]
            ? (left + right) / 2
            : left + ((item.v - extent[0]) / (extent[1] - extent[0])) * mw;
        const arm = index.arms.indexOf(item.arm),
          yy = y + (arm % 2 === 0 ? -2.4 : 2.4);
        const mark = node(chart, 'g', {
          'data-micro-reading': g,
          'data-id': item.id,
          'data-value': item.v,
        });
        node(mark, 'circle', { cx: xx, cy: yy, r: 4, class: 'ml-hit' });
        node(mark, 'circle', {
          cx: xx,
          cy: yy,
          r: 2.6,
          fill: hues[arm % hues.length],
          class: 'ml-micro-point',
        });
        const raw =
          item.id +
          ' · ' +
          item.arm +
          ' · ' +
          g +
          ' · ' +
          exact(item.v) +
          ' ' +
          unitOf(f) +
          ' · ' +
          visit;
        inspect(s, mark, raw, () => selectMark(s, raw, mark), xx, yy);
      }
      const range = fmt(extent[0], 3) + '–' + fmt(extent[1], 3);
      clipped(
        chart,
        dir > 0 ? left : right,
        y + 17,
        w < 480 ? 'n = ' + values.length : range + '  ·  n = ' + values.length,
        'ml-micro-n',
        mw + 4,
        dir > 0 ? 'start' : 'end',
      );
    });
    const observed = readings(active),
      mean = observed.length ? observed.reduce((sum, p) => sum + p.v / observed.length, 0) : null;
    const radius = Math.min(58, Math.max(38, w * 0.095));
    const hub = node(chart, 'g', { 'data-hub-mean': mean ?? '', 'data-n': observed.length });
    node(hub, 'circle', { cx, cy, r: radius + 6, class: 'ml-core-ring' });
    node(hub, 'circle', { cx, cy, r: radius, fill: 'url(#' + s.uid + '-core)', class: 'ml-core' });
    label(hub, cx, cy - 13, 'Score mean', 'ml-core-label', 'middle');
    clipped(hub, cx, cy + 9, fmt(mean, 4), 'ml-core-n', radius * 1.8, 'middle');
    label(hub, cx, cy + 27, 'n = ' + observed.length, 'ml-core-label', 'middle');
    const detail =
      nameOf(active) +
      ' · Participant mean ' +
      (mean === null ? 'unavailable' : exact(mean) + ' ' + unitOf(active)) +
      ' · n = ' +
      observed.length +
      ' at ' +
      visit;
    inspect(s, hub, detail, () => selectMark(s, detail, hub), cx, cy);
    s.caption.textContent =
      'Spokes indicate gene-set membership. The hub is the participant mean of the selected pathway score. Member strips show up to 32 observed participant values on each gene’s own expression scale.';
    if (active.membershipSource?.name)
      s.caption.textContent += ' Gene-set definition from ' + active.membershipSource.name + '.';
    pager(s, 'Members', view.members, listed.length, size, (x) => {
      view.members = x;
    });
  }

  function stableSubset(series, slots) {
    if (series.length <= slots) return series;
    const keep = new Set();
    let low = Infinity,
      high = -Infinity,
      lowRecord,
      highRecord;
    for (const p of series)
      for (const v of p.values)
        if (v !== null) {
          if (v < low) {
            low = v;
            lowRecord = p;
          }
          if (v > high) {
            high = v;
            highRecord = p;
          }
        }
    if (lowRecord) keep.add(lowRecord);
    if (highRecord) keep.add(highRecord);
    for (const { p } of series
      .map((p) => ({ p, n: hash(p.id) }))
      .sort((a, b) => a.n - b.n || compare.compare(a.p.id, b.p.id))) {
      if (keep.size >= slots) break;
      keep.add(p);
    }
    return [...keep];
  }

  function modules(s, w) {
    const { chart, index, pop, visit, active, features, view } = s;
    const compact = w < 520,
      per = w < 500 ? 4 : 6;
    view.lane = Math.min(
      Math.max(0, view.lane || 0),
      Math.max(0, Math.ceil(features.length / per) - 1),
    );
    const fs = features.slice(view.lane * per, (view.lane + 1) * per);
    const observation = index.byPopulation.get(pop);
    const rows = fs.map((f) => {
      const vals = index.participants.flatMap((p) => {
        const v = measurement(observation?.get(key(p.id, visit)), f.id);
        return v === null ? [] : [{ ...p, v, values: [v] }];
      });
      const range = domain(vals.map((p) => p.v));
      return { f, vals, scale: range ? ticks(range) : null };
    });
    const armH = 23,
      firstY = 69,
      rowH = (compact ? 47 : 29) + armH * Math.max(index.arms.length, 1) + 23;
    const h = firstY + rowH * rows.length + 10;
    s.size(w, h);
    label(chart, 2, 20, 'Score distributions', 'ml-chart-title');
    clipped(
      chart,
      2,
      42,
      pop + '  ·  ' + visit + '  ·  Independent scale per score',
      'ml-note',
      w - 6,
    );
    let sampled = false;
    rows.forEach(({ f, vals, scale }, r) => {
      const y = firstY + r * rowH;
      const labelW = compact ? w * 0.65 : Math.min(218, w * 0.205);
      const armX = compact ? 2 : labelW + 14,
        left = compact ? Math.min(97, w * 0.27) : armX + Math.min(94, w * 0.11);
      const right = w - 12,
        rowStart = y + (compact ? 43 : 15);
      const selectedScore = f.id === active.id;
      const head = node(chart, 'g', {
        'data-module-id': f.id,
        'aria-pressed': String(selectedScore),
      });
      node(head, 'rect', {
        x: 0,
        y: y - 12,
        width: labelW,
        height: compact ? 39 : rowH - 8,
        fill: 'transparent',
      });
      if (selectedScore)
        node(head, 'rect', {
          x: 0,
          y: y - 7,
          width: 3,
          height: compact ? 30 : Math.max(27, armH * index.arms.length),
          rx: 1.5,
          fill: '#79e6c5',
        });
      clipped(
        head,
        9,
        y + 8,
        nameOf(f),
        selectedScore ? 'ml-lane-name ml-active' : 'ml-lane-name',
        labelW - 15,
      );
      clipped(head, 9, y + 25, unitOf(f), 'ml-note', labelW - 14);
      inspect(
        s,
        head,
        'Select ' + nameOf(f) + ' · ' + unitOf(f),
        () => {
          if (f.id !== active.id) s.config.onFeature?.(f.id);
          else selectMark(s, nameOf(f) + ' · ' + unitOf(f), head);
        },
        labelW / 2,
        y + 6,
      );
      if (!scale) {
        label(chart, (left + right) / 2, rowStart + 16, 'No measured scores', 'ml-note', 'middle');
        line(chart, 0, y + rowH - 12, w, y + rowH - 12, 'ml-lane-divider');
        return;
      }
      const lo = scale.values[0],
        hi = scale.values[scale.values.length - 1];
      const xx = (v) => left + ((v - lo) / (hi - lo || 1)) * (right - left);
      const axisBottom = rowStart + armH * (index.arms.length - 1) + 12;
      const tickWidth =
        Math.max(...scale.values.map((tick) => scale.format(tick).length * 6.8)) + 8;
      const skip = Math.max(1, Math.ceil((tickWidth * (scale.values.length - 1)) / (right - left)));
      scale.values.forEach((tick, ti) => {
        const x = xx(tick);
        line(chart, x, rowStart - 10, x, axisBottom, 'ml-grid');
        const last = scale.values.length - 1;
        if (ti !== 0 && ti !== last && (ti % skip !== 0 || last - ti < skip)) return;
        const anchor = ti === 0 ? 'start' : ti === last ? 'end' : 'middle';
        label(chart, x, axisBottom + 14, scale.format(tick), 'ml-axis', anchor);
      });
      index.arms.forEach((arm, ai) => {
        const metrics = vals.filter((p) => p.arm === arm).sort((a, b) => a.v - b.v),
          n = metrics.length,
          yy = rowStart + armH * ai,
          shade = hues[ai % hues.length];
        const shown = stableSubset(metrics, 36).sort((a, b) => a.v - b.v);
        if (shown.length < n) sampled = true;
        node(chart, 'circle', { cx: armX + 3, cy: yy, r: 3.2, fill: shade });
        clipped(
          chart,
          armX + 12,
          yy + 4,
          arm + ' · ' + (n > shown.length ? shown.length + '/' + n : n),
          'ml-arm-n',
          left - armX - 15,
        );
        if (!n) return;
        line(chart, left, yy, right, yy, 'ml-lane-axis');
        const mid = Math.floor(n / 2),
          median = n % 2 ? metrics[mid].v : metrics[mid - 1].v / 2 + metrics[mid].v / 2;
        const mx = xx(median),
          med = node(chart, 'g', { 'data-median': f.id, 'data-arm': arm, 'data-value': median });
        line(med, mx, yy - 9, mx, yy + 9, 'ml-median-halo');
        line(med, mx, yy - 9, mx, yy + 9, 'ml-median');
        const packed = [];
        for (const d of shown) {
          const x = xx(d.v),
            neighbor = packed.filter((v) => x - v.x < 7);
          const lane =
            [0, -4.6, 4.6, -7, 7].find((dy) =>
              neighbor.every((v) => Math.hypot(x - v.x, dy - v.dy) > 5.8),
            ) ?? 0;
          const py = yy + lane;
          packed.push({ x, dy: lane });
          const mark = node(chart, 'g', {
            'data-reading': '',
            'data-id': d.id,
            'data-feature': f.id,
            'data-visit': visit,
            'data-value': d.v,
            class: 'ml-reading',
          });
          node(mark, 'circle', { cx: x, cy: py, r: 4.3, class: 'ml-hit' });
          node(mark, 'circle', { cx: x, cy: py, r: 3.3, fill: shade, class: 'ml-score-point' });
          const desc =
            d.id +
            ' · ' +
            arm +
            ' · ' +
            visit +
            ' · ' +
            nameOf(f) +
            ' · ' +
            exact(d.v) +
            ' ' +
            unitOf(f);
          inspect(s, mark, desc, () => selectMark(s, desc, mark), x, py);
        }
      });
      line(chart, 0, y + rowH - 12, w, y + rowH - 12, 'ml-lane-divider');
    });
    s.caption.textContent =
      'One point per measured participant. The white notch marks the arm median. Each score has its own numeric axis.' +
      (sampled
        ? ' Up to 36 individuals per arm and score are shown. Medians use all measured participants.'
        : '');
    pager(s, 'Scores', view.lane, features.length, per, (x) => {
      view.lane = x;
    });
  }

  function pathwayScores(s, w) {
    const { chart, index, active, pop } = s,
      dates = index.visits;
    const by = index.byPopulation.get(pop),
      palette = (arm) => hues[index.arms.indexOf(arm) % hues.length];
    const series = index.participants
      .map((p) => ({
        ...p,
        values: dates.map((v) => measurement(by?.get(key(p.id, v)), active.id)),
      }))
      .filter((p) => p.values.some((v) => v !== null));
    const all = series.flatMap((p) => p.values),
      extent = domain(all);
    const h = w < 500 ? 362 : Math.min(520, Math.max(360, (window.innerHeight || 800) - 325));
    s.size(w, h);
    label(chart, 2, 20, 'Pathway scores by visit', 'ml-chart-title');
    clipped(
      chart,
      2,
      42,
      nameOf(active) + '  ·  ' + unitOf(active) + '  ·  ' + pop,
      'ml-note',
      w - 6,
    );
    if (!extent) {
      label(
        chart,
        w / 2,
        h * 0.55,
        'No measured scores for this population',
        'ml-empty-copy',
        'middle',
      );
      s.caption.textContent = 'Missing measurements contribute no points.';
      return;
    }
    const scale = ticks(extent);
    const span = Math.max(1, Math.min(dates.length, w < 480 ? 5 : 9));
    s.view.visitPage = Math.min(
      Math.max(0, s.view.visitPage),
      Math.max(0, Math.ceil(dates.length / span) - 1),
    );
    const first = s.view.visitPage * span,
      selected = dates.slice(first, first + span);
    const left = Math.max(
        45,
        Math.max(...scale.values.map((v) => scale.format(v).length * 7)) + 12,
      ),
      right = w - 12,
      top = 82,
      bottom = h - 52;
    const column = (right - left) / selected.length,
      xs = selected.map((_, i) => left + column * (i + 0.5)),
      y = (v) => scale.y(v, top + 5, bottom - 5);
    for (const tick of scale.values) {
      const yy = y(tick);
      line(chart, left, yy, right, yy, 'ml-grid');
      label(chart, left - 8, yy + 4, scale.format(tick), 'ml-axis', 'end');
    }
    selected.forEach((v, i) => {
      line(chart, xs[i], top, xs[i], bottom, 'ml-visit-line');
      clipped(chart, xs[i], bottom + 21, v, 'ml-visit', column - 4, 'middle');
      label(
        chart,
        xs[i],
        bottom + 39,
        'n = ' + series.filter((p) => p.values[first + i] !== null).length,
        'ml-note',
        'middle',
      );
    });
    const shown = index.arms.flatMap((a) =>
      stableSubset(
        series.filter((p) => p.arm === a),
        35,
      ),
    );
    let lx = left;
    for (const a of index.arms) {
      const count = series.filter((p) => p.arm === a).length;
      if (!count) continue;
      const legend = a + ' · ' + count;
      const room = Math.min(140, (right - left) / index.arms.length - 10);
      node(chart, 'circle', { cx: lx + 3, cy: 61, r: 3.5, fill: palette(a) });
      clipped(chart, lx + 13, 65, legend, 'ml-note', room - 12);
      lx += room + 10;
    }
    const points = node(chart, 'g', {});
    shown.forEach((p) => {
      const selectedValues = p.values.slice(first, first + selected.length),
        arm = index.arms.indexOf(p.arm);
      const stroke = palette(p.arm);
      const offset = ((arm + 0.5) / index.arms.length - 0.5) * column * 0.72;
      const jitter = ((hash(p.id) % 101) / 100 - 0.5) * Math.min(12, column * 0.09);
      const xx = (i) => xs[i] + offset + jitter;
      selectedValues.forEach((v, i) => {
        if (v === null) return;
        const x = xx(i),
          yy = y(v),
          visit = selected[i];
        const mark = node(points, 'g', {
          'data-reading': '',
          'data-id': p.id,
          'data-visit': visit,
          'data-value': v,
          class: 'ml-reading',
        });
        node(mark, 'circle', { cx: x, cy: yy, r: 8, class: 'ml-hit' });
        node(mark, 'circle', { cx: x, cy: yy, r: 4, fill: stroke, class: 'ml-score-point' });
        const detail =
          p.id + ' · ' + p.arm + ' · ' + visit + ' · ' + exact(v) + ' ' + unitOf(active);
        inspect(s, mark, detail, () => selectMark(s, detail, mark), x, yy);
      });
    });
    s.caption.textContent =
      'Points are measured participant scores.' +
      (shown.length < series.length
        ? ' Showing ' +
          shown.length +
          ' of ' +
          series.length +
          ' participants. Visit counts include all measured participants.'
        : ' Visit counts include only participants with a measured score.');
    pager(s, 'Visits', s.view.visitPage, dates.length, span, (x) => {
      s.view.visitPage = x;
    });
  }

  function dispose(host) {
    const s = sessions.get(host);
    if (!s) return;
    s.observer?.disconnect();
    window.removeEventListener('resize', s.resize);
    s.abort?.abort();
    sessions.delete(host);
  }

  function render(host, config) {
    if (!host || !config?.study || !kinds[config.mode]) return;
    const old = sessions.get(host),
      reuse = old && old.config.study === config.study;
    const index = reuse ? old.index : studyIndex(config.study);
    const mode = config.mode;
    const view =
      reuse && old.config.mode === mode ? old.view : { row: 0, col: 0, members: 0, visitPage: 0 };
    const features = (config.study.features || []).filter(
      (f) => featureKind(f) === kinds[mode] || (mode === 'modules' && featureKind(f) === 'score'),
    );
    const wanted =
      typeof config.selectedFeature === 'object'
        ? config.selectedFeature?.id
        : config.selectedFeature;
    const active = features.find((f) => f.id === wanted) || features[0];
    const featureChanged = !old || old.active?.id !== active?.id;
    dispose(host);
    const root = html('section', 'ml-stage ml-' + mode);
    const top = html('header', 'ml-header'),
      titles = html('div', 'ml-titles');
    titles.append(
      html(
        'div',
        'ml-source',
        config.source || config.study.study?.source || config.study.source || 'Study',
      ),
    );
    const generic = mode === 'modules' && active && featureKind(active) === 'score';
    titles.append(html('h1', 'ml-title', generic ? 'Recorded scores' : names[mode]));
    top.append(titles);
    root.append(top);
    host.replaceChildren(root);
    const s = {
      config,
      mode,
      view,
      index,
      features,
      active,
      root,
      uid: 'ml-' + ++sequence,
      abort: new AbortController(),
    };
    sessions.set(host, s);
    if (!active) {
      const empty = html('div', 'ml-empty');
      const what = {
        genes: 'gene-expression measurements',
        pathways: 'pathway scores or gene-set definitions',
        modules: 'module or recorded scores',
      }[mode];
      empty.append(html('h2', '', 'This study has no ' + what + '.'));
      const help = {
        genes: 'Open an H5AD with selected genes, or a participant-level CSV with gene: columns.',
        pathways: 'Map a measured score named pathway:NAME in a CSV or H5AD study.',
        modules: 'Open a study with module scores or numeric participant-level measurements.',
      }[mode];
      empty.append(html('p', '', help));
      if (config.onOpenStudy)
        empty.append(button('Open study', () => config.onOpenStudy(), 'ml-action'));
      root.append(empty);
      return;
    }
    config.onDisplayedFeature?.(active.id);
    s.pop = index.pops.includes(config.selectedPopulation)
      ? config.selectedPopulation
      : index.pops[0];
    s.visit = index.visits.includes(config.selectedVisit) ? config.selectedVisit : index.visits[0];
    const memberLens =
      mode === 'pathways' &&
      Array.isArray(active.members) &&
      active.members.some((g) => String(g).trim());
    const controls = html('div', 'ml-controls');
    controls.append(
      selectControl(
        mode === 'genes' ? 'Gene' : mode === 'pathways' ? 'Pathway' : 'Score',
        features.map((f) => ({
          id: f.id,
          label:
            nameOf(f) +
            (mode === 'modules' && featureKind(f) === 'score' ? ' · recorded score' : ''),
        })),
        active.id,
        config.onFeature,
      ),
    );
    if (index.pops.length)
      controls.append(
        selectControl(
          'Cell population',
          index.pops.map((id) => ({ id, label: id })),
          s.pop,
          config.onPopulation,
        ),
      );
    if ((mode === 'genes' || mode === 'modules' || memberLens) && index.visits.length)
      controls.append(
        selectControl(
          'Visit',
          index.visits.map((id) => ({ id, label: id })),
          s.visit,
          config.onVisit,
        ),
      );
    top.append(controls);
    if (mode === 'pathways' && typeof config.onLinkGeneSet === 'function') {
      controls.append(
        button(
          active.membershipSource ? 'Replace gene set…' : 'Link gene set…',
          () => config.onLinkGeneSet(active.id),
          'ml-link-set',
        ),
      );
    }
    if (config.onAnalyze) {
      const analyze = button(
        'Analyze ' + (mode === 'genes' ? 'gene' : mode === 'pathways' ? 'pathway' : 'score'),
        () => config.onAnalyze(active.id),
        'ml-action',
      );
      if (config.analysisContext) {
        analyze.title = 'Open comparison · ' + config.analysisContext;
        analyze.setAttribute('aria-description', analyze.title);
      }
      controls.append(analyze);
      if (config.analysisRequiresExistingVisit && config.analysisContext) {
        controls.append(html('span', 'ml-destination', 'Opens comparison · ' + config.analysisContext));
      }
    }
    const figure = html('figure', 'ml-figure');
    s.figure = figure;
    const chart = document.createElementNS(ns, 'svg');
    s.chart = chart;
    chart.setAttribute('class', 'ml-chart');
    chart.setAttribute('role', 'group');
    chart.setAttribute(
      'aria-label',
      (generic ? 'Recorded scores' : names[mode]) + ' · ' + nameOf(active),
    );
    chart.setAttribute(
      'aria-description',
      'Use arrow keys to explore values. Enter selects a measurement or member.',
    );
    const tip = html('div', 'ml-tooltip');
    tip.hidden = true;
    tip.setAttribute('aria-hidden', 'true');
    s.tip = tip;
    const foot = html('figcaption', 'ml-footer');
    s.caption = html('p', 'ml-caption');
    s.actions = html('div', 'ml-actions');
    s.insight = html('p', 'ml-insight');
    s.insight.setAttribute('role', 'status');
    s.insight.setAttribute('aria-live', 'polite');
    foot.append(s.caption, s.actions, s.insight);
    figure.append(chart, tip, foot);
    root.append(figure);
    let lastSize = '',
      firstDraw = true;
    s.size = (w, h) => {
      chart.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      chart.setAttribute('height', h);
    };
    s.redraw = (force = false) => {
      const w = Math.max(220, Math.round(figure.getBoundingClientRect().width));
      if (!force && lastSize === w + 'x' + window.innerHeight) return;
      lastSize = w + 'x' + window.innerHeight;
      const focus = chart.contains(document.activeElement)
        ? document.activeElement.getAttribute('aria-label')
        : null;
      s.abort.abort();
      s.abort = new AbortController();
      s.items = [];
      s.columns = 1;
      chart.replaceChildren();
      defs(chart, s.uid);
      s.actions.replaceChildren();
      s.insight.textContent = '';
      tip.hidden = true;
      if (mode === 'genes') {
        if (firstDraw && featureChanged) {
          const per = Math.max(1, Math.min(9, Math.floor((w - 115) / 83)));
          view.col = Math.floor(features.indexOf(active) / per);
        }
        genes(s, w);
      } else if (memberLens) membership(s, w);
      else if (mode === 'modules') {
        if (firstDraw && featureChanged)
          view.lane = Math.floor(features.indexOf(active) / (w < 500 ? 4 : 6));
        modules(s, w);
      } else pathwayScores(s, w);
      firstDraw = false;
      interactions(s);
      if (focus)
        s.items
          .find((i) => i.item.getAttribute('aria-label') === focus)
          ?.item.focus({ preventScroll: true });
    };
    s.redraw();
    s.resize = () => {
      if (!root.isConnected) {
        dispose(host);
        return;
      }
      s.redraw();
    };
    if (typeof ResizeObserver !== 'undefined') {
      s.observer = new ResizeObserver(s.resize);
      s.observer.observe(figure);
    }
    window.addEventListener('resize', s.resize, { passive: true });
    if (chart.animate && !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
      chart.animate(
        [
          { opacity: 0.48, transform: 'translateY(3px)' },
          { opacity: 1, transform: 'none' },
        ],
        { duration: 190, easing: 'ease-out' },
      );
  }

  window.SctrialMolecular = Object.freeze({ render, dispose });
})();
