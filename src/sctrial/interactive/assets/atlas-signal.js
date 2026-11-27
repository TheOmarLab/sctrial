(() => {
  'use strict';

  const svgNS = 'http://www.w3.org/2000/svg';
  const numeric = new Intl.NumberFormat('en', { maximumSignificantDigits: 3 });
  const counts = new Intl.NumberFormat('en');
  const miniatureKeys = new WeakMap();
  const zeroColor = [80, 91, 101];
  const missingColor = [125, 124, 138];
  const positiveStops = [
    [0, [34, 87, 103]],
    [0.075, [44, 150, 157]],
    [0.24, [112, 202, 175]],
    [0.55, [215, 201, 129]],
    [1, [255, 202, 113]],
  ];
  const negativeStops = [
    [0, zeroColor],
    [0.12, [63, 111, 138]],
    [0.44, [72, 166, 197]],
    [1, [150, 223, 239]],
  ];
  const signedPositiveStops = [
    [0, zeroColor],
    [0.12, [140, 121, 88]],
    [0.44, [209, 155, 81]],
    [1, [255, 215, 135]],
  ];
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));

  // Halving before subtraction also covers a scale spanning opposite finite extremes.
  function position(value, domain) {
    const [lo, hi] = domain;
    if (lo === hi) return 0.5;
    const width = hi - lo;
    return clamp(
      Number.isFinite(width) ? (value - lo) / width : (value / 2 - lo / 2) / (hi / 2 - lo / 2),
    );
  }

  function at(domain, t) {
    return domain[0] * (1 - t) + domain[1] * t;
  }

  function mix(a, b, t) {
    return a.map((v, i) => Math.round(v + (b[i] - v) * clamp(t))).join(',');
  }

  function colorAt(level, stops) {
    const end = stops.findIndex(([p]) => p >= level);
    if (end <= 0) return (end === 0 ? stops[0][1] : stops.at(-1)[1]).join(',');
    const [a, first] = stops[end - 1],
      [b, last] = stops[end];
    return mix(first, last, (level - a) / (b - a));
  }

  function color(value, domain) {
    if (!domain || !Number.isFinite(value)) return missingColor.join(',');
    if (value === 0) return zeroColor.join(',');
    if (domain[0] < 0) {
      const amplitude = Math.max(Math.abs(domain[0]), Math.abs(domain[1]));
      return colorAt(
        Math.min(1, Math.abs(value) / amplitude),
        value < 0 ? negativeStops : signedPositiveStops,
      );
    }
    return colorAt(position(value, domain), positiveStops);
  }

  function gradient(domain) {
    if (!domain) return 'none';
    if (domain[0] === domain[1]) return 'rgb(' + color(domain[0], domain) + ')';
    const signed = domain[0] < 0;
    const pivots = new Set([0, 1]);
    if (signed) {
      const amplitude = Math.max(Math.abs(domain[0]), Math.abs(domain[1]));
      for (const [p] of negativeStops) pivots.add(position(-p * amplitude, domain));
      for (const [p] of signedPositiveStops) pivots.add(position(p * amplitude, domain));
    } else {
      for (const [p] of positiveStops) pivots.add(p);
    }
    const stops = Array.from(pivots)
      .sort((a, b) => a - b)
      .map((p) => {
        const rgb = signed ? color(at(domain, p), domain) : colorAt(p, positiveStops);
        return 'rgb(' + rgb + ') ' + p * 100 + '%';
      });
    return 'linear-gradient(90deg, ' + stops.join(', ') + ')';
  }

  function svg(tag, attributes = {}) {
    const el = document.createElementNS(svgNS, tag);
    for (const [key, val] of Object.entries(attributes)) el.setAttribute(key, val);
    return el;
  }

  function html(tag, cls) {
    const el = document.createElement(tag);
    el.className = cls;
    return el;
  }

  function mini(host, group, domain) {
    const { min, q1, median, q3, max } = group;
    const measured = domain && Number.isFinite(median);
    const key = measured ? [min, q1, median, q3, max, ...domain].join('|') : '';
    if (miniatureKeys.get(host) === key) return;
    miniatureKeys.set(host, key);
    host.replaceChildren();
    host.hidden = !measured;
    if (!measured) return;
    const x = (v) => 3 + 82 * position(v, domain);
    const shape = svg('svg', {
      viewBox: '0 0 88 11',
      preserveAspectRatio: 'none',
      'aria-hidden': 'true',
    });
    const medianColor = 'rgb(' + color(median, domain) + ')';
    shape.append(
      svg('path', { d: 'M3 5.5H85', stroke: '#476064', 'stroke-width': 0.7 }),
      svg('path', { d: 'M' + x(min) + ' 5.5H' + x(max), stroke: '#9caeac', 'stroke-width': 1.3 }),
      svg('rect', {
        x: x(q1),
        y: 3.2,
        width: Math.max(1, x(q3) - x(q1)),
        height: 4.6,
        rx: 1.7,
        fill: medianColor,
      }),
      svg('path', { d: 'M' + x(median) + ' 2V9', stroke: '#f0f4e9', 'stroke-width': 1.8 }),
    );
    host.title =
      'Participant range ' +
      numeric.format(min) +
      ' to ' +
      numeric.format(max) +
      ' · middle 50% ' +
      numeric.format(q1) +
      ' to ' +
      numeric.format(q3) +
      ' · median ' +
      numeric.format(median);
    host.append(shape);
  }

  // Each small-sample dot stays at the participant's actual value. A crowd of coincident
  // marks switches to counted bins, preserving ties without invented horizontal jitter.
  function marks(values, domain, width) {
    const left = 5;
    const right = width - 5;
    const x = (v) => left + (right - left) * position(v, domain);
    const ends = [];
    const dots = [];
    if (values.length <= 48) {
      for (const value of values) {
        const px = x(value);
        let lane = ends.findIndex((end) => px - end >= 7.2);
        if (lane < 0) lane = ends.length;
        if (lane === 5) break;
        ends[lane] = px;
        dots.push({ x: px, lane, value });
      }
      if (dots.length === values.length) return { dots, lanes: ends.length };
    }
    if (domain[0] === domain[1]) {
      const bar = Math.min(right - left, 14);
      return {
        bins: [
          {
            from: domain[0],
            to: domain[1],
            x: (width - bar) / 2,
            width: bar,
            count: values.length,
          },
        ],
        highest: values.length,
      };
    }
    const nBins = clamp(Math.floor((right - left) / 12), 5, 42);
    const bins = Array.from({ length: nBins }, (_, i) => ({
      from: at(domain, i / nBins),
      to: at(domain, (i + 1) / nBins),
      x: left + (i * (right - left)) / nBins,
      width: (right - left) / nBins,
      count: 0,
    }));
    for (const value of values)
      bins[Math.min(nBins - 1, Math.floor(position(value, domain) * nBins))].count++;
    return { bins, highest: Math.max(...bins.map((b) => b.count)) };
  }

  // Merge adjacent original bins only when needed to keep bars distinguishable at this width.
  function cellMarks(histogram, width) {
    const { counts, domain } = histogram;
    if (domain[0] === domain[1]) {
      const span = Math.min(18, width - 10);
      const n = counts.reduce((a, b) => a + b, 0);
      return {
        bins: [{ x: (width - span) / 2, width: span, from: domain[0], to: domain[1], count: n }],
        highest: n,
      };
    }
    const capacity = Math.max(1, Math.floor((width - 10) / 4));
    let step = Math.max(1, Math.ceil(counts.length / capacity));
    while (counts.length % step !== 0) step++;
    const bins = [];
    let highest = 0;
    for (let i = 0; i < counts.length; i += step) {
      const end = Math.min(counts.length, i + step);
      const count = counts.slice(i, end).reduce((a, b) => a + b, 0);
      const x = 5 + ((width - 10) * i) / counts.length;
      const span = ((width - 10) * (end - i)) / counts.length;
      bins.push({
        from: at(domain, i / counts.length),
        to: at(domain, end / counts.length),
        x,
        width: span,
        count,
      });
      highest = Math.max(highest, count);
    }
    return { bins, highest };
  }

  function mount(host) {
    host.classList.add('at-signal');
    const body = html('div', 'at-signal-body');
    const label = html('div', 'at-signal-label');
    const name = html('strong', 'at-signal-population');
    const detail = html('span', 'at-signal-detail');
    const options = html('div', 'at-signal-options');
    options.setAttribute('role', 'group');
    options.setAttribute('aria-label', 'Inspect distribution');
    const cellButton = html('button', 'at-signal-option');
    const personButton = html('button', 'at-signal-option');
    for (const [button, text, aria] of [
      [cellButton, 'Cell values', 'Cell distribution'],
      [personButton, 'Participants', 'Participant distribution'],
    ]) {
      button.type = 'button';
      button.textContent = text;
      button.setAttribute('aria-label', aria);
      options.append(button);
    }
    label.append(name, detail);
    const chart = html('div', 'at-signal-chart');
    chart.tabIndex = 0;
    chart.setAttribute('role', 'group');
    const plot = svg('svg', { 'aria-hidden': 'true', preserveAspectRatio: 'none' });
    const ticks = html('div', 'at-signal-ticks');
    chart.append(plot, ticks);
    const stat = html('div', 'at-signal-stat');
    const metric = html('span', 'at-signal-metric');
    const number = html('strong', 'at-signal-number');
    stat.append(metric, number);
    const empty = html('span', 'at-signal-empty');
    const announcement = html('span', 'at-signal-announcement');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    body.append(label, options, chart, stat, empty, announcement);
    host.replaceChildren(body);
    let group, scale, feature, visit, arm, layout, cursor, participantMeans, selection, hasCells;
    let activeIndex = -1,
      width = 0,
      view = 'cells';
    const events = new AbortController();
    const on = (target, type, fn) => target.addEventListener(type, fn, { signal: events.signal });
    const cellView = () => hasCells && view === 'cells';
    const domain = () => (cellView() ? group?.cellBins?.domain : scale);

    function interval(bin, last) {
      if (bin.from === bin.to) return String(bin.from);
      return '[' + String(bin.from) + ', ' + String(bin.to) + (last ? ']' : ')');
    }
    function meanLabel() {
      if (!group.n) return 'No participant values';
      const total =
        counts.format(group.n) +
        (group.participants > group.n ? '/' + counts.format(group.participants) : '');
      return (
        total + ' participant ' + (participantMeans ? 'mean' : 'value') + (group.n === 1 ? '' : 's')
      );
    }
    function cellLabel() {
      const n = group?.cellBins?.measured ?? 0;
      const total = counts.format(n) + (group?.count > n ? '/' + counts.format(group.count) : '');
      return total + ' measured cell' + (n === 1 ? '' : 's');
    }
    function readout(index, announce = false) {
      activeIndex = index;
      const cells = cellView(),
        selected = index >= 0,
        bins = layout?.bins;
      const suffix = feature?.unit ? ' ' + feature.unit : '';
      let px, description;
      if (selected && bins) {
        const bin = bins[index];
        metric.textContent = cells ? 'Cells in bin' : 'People in bin';
        number.textContent = counts.format(bin.count);
        description =
          counts.format(bin.count) +
          (cells ? ' cells in ' : ' participants in ') +
          interval(bin, index === bins.length - 1) +
          suffix;
        detail.textContent =
          bin.from === bin.to
            ? numeric.format(bin.from)
            : numeric.format(bin.from) + ' – ' + numeric.format(bin.to);
        px = bin.x + bin.width / 2;
      } else if (!cells) {
        const value = selected ? group.values[index] : group.median;
        metric.textContent = selected ? 'Participant' : 'Median';
        number.textContent = numeric.format(value);
        detail.textContent = meanLabel();
        description = metric.textContent + ' · ' + String(value) + suffix;
        px = 5 + (width - 10) * position(value, scale);
      } else {
        const hist = group.cellBins;
        if (feature.kind === 'gene' && hist.domain[0] >= 0) {
          metric.textContent = 'Nonzero cells';
          number.textContent =
            numeric.format((100 * (hist.measured - hist.zero)) / hist.measured) + '%';
          description =
            counts.format(hist.measured - hist.zero) +
            ' of ' +
            counts.format(hist.measured) +
            ' measured cells are nonzero.';
        } else {
          metric.textContent = 'Cells measured';
          number.textContent = numeric.format((100 * hist.measured) / group.count) + '%';
          description =
            counts.format(hist.measured) +
            ' of ' +
            counts.format(group.count) +
            ' cells have a recorded score.';
        }
        detail.textContent = cellLabel();
      }
      stat.title = description;
      if (announce) announcement.textContent = description;
      else if (!selected && announcement.textContent) announcement.textContent = '';
      if (cursor) {
        cursor.setAttribute('visibility', px == null ? 'hidden' : 'visible');
        if (px != null) cursor.setAttribute('transform', 'translate(' + px + ' 0)');
        cursor.setAttribute('class', selected ? 'at-signal-cursor is-active' : 'at-signal-cursor');
      }
    }
    function tick(value, pct, align) {
      const mark = html('span', 'at-signal-tick ' + align);
      mark.style.left = pct + '%';
      mark.textContent = numeric.format(value);
      mark.title = String(value) + (feature?.unit ? ' ' + feature.unit : '');
      ticks.append(mark);
    }
    function paint() {
      if (host.hidden) return;
      const cells = cellView(),
        hist = group?.cellBins;
      const range = domain(),
        hasData = cells ? hist?.measured > 0 : group?.n > 0;
      host.dataset.unit = cells ? 'cells' : 'participants';
      options.hidden = !hasCells;
      cellButton.setAttribute('aria-pressed', String(cells));
      personButton.setAttribute('aria-pressed', String(!cells));
      name.textContent = group?.name || selection || '';
      name.title = (feature?.label || '') + ' · ' + name.textContent;
      chart.hidden = stat.hidden = !hasData;
      empty.hidden = !!hasData;
      if (!hasData || !range) {
        activeIndex = -1;
        layout = cursor = null;
        delete host.dataset.measured;
        detail.textContent = group ? (cells ? cellLabel() : meanLabel()) : '';
        empty.textContent = 'No measured values at this visit';
        return;
      }
      width = Math.max(30, Math.round(chart.clientWidth));
      const valuesN = cells ? hist.measured : group.n;
      host.dataset.measured = String(valuesN);
      layout = cells ? cellMarks(hist, width) : marks(group.values, range, width);
      const stage = document.createDocumentFragment();
      const base = 38;
      const symbols = svg('g', { class: 'at-signal-symbols' });
      stage.append(
        svg('path', {
          d: 'M5 ' + base + 'H' + (width - 5),
          stroke: '#54616a',
          'stroke-width': 0.75,
        }),
      );
      const signed = range[0] < 0 && range[1] > 0;
      if (signed) {
        const x = 5 + (width - 10) * position(0, range);
        stage.append(
          svg('path', {
            d: 'M' + x + ' 3V' + base,
            stroke: '#a2afb1',
            'stroke-opacity': 0.62,
            'stroke-dasharray': '2.5 3',
          }),
        );
      }
      if (layout.dots) {
        for (const dot of layout.dots) {
          symbols.append(
            svg('circle', {
              cx: dot.x,
              cy: base - 4.2 - dot.lane * 6.4,
              r: 3.4,
              fill: 'rgb(' + color(dot.value, range) + ')',
              stroke: '#d5dfdb',
              'stroke-width': 0.7,
            }),
          );
        }
        // The interval across the top is the middle half of participant values.
        const a = 5 + (width - 10) * position(group.q1, range),
          b = 5 + (width - 10) * position(group.q3, range);
        stage.append(
          svg('path', {
            d: 'M' + a + ' 5V8H' + b + 'V5',
            stroke: '#90a9ac',
            'stroke-width': 1,
            fill: 'none',
          }),
        );
      } else {
        for (const [index, b] of layout.bins.entries()) {
          const height = (33 * b.count) / layout.highest;
          let v = at([b.from, b.to], 0.5);
          if (cells && b.from <= 0 && b.to >= 0 && hist.measured === hist.zero) v = 0;
          symbols.append(
            svg('rect', {
              x: b.x + 0.45,
              y: base - height,
              width: Math.max(0.5, b.width - 0.9),
              height,
              rx: Math.min(1.8, b.width / 5),
              fill: 'rgb(' + color(v, range) + ')',
              'data-count': b.count,
            }),
          );
          const containsZero =
            b.from <= 0 && (b.to > 0 || (index === layout.bins.length - 1 && b.to === 0));
          if (cells && containsZero && hist.zero > 0 && hist.zero <= b.count) {
            const zeroHeight = (33 * hist.zero) / layout.highest;
            symbols.append(
              svg('rect', {
                x: b.x + 0.45,
                y: base - zeroHeight,
                width: Math.max(0.5, b.width - 0.9),
                height: zeroHeight,
                fill: 'rgb(' + zeroColor.join(',') + ')',
              }),
            );
          }
        }
        const yLabel = svg('text', {
          x: width - 5,
          y: 9,
          'text-anchor': 'end',
          class: 'at-signal-height-label',
        });
        yLabel.textContent = numeric.format((100 * layout.highest) / valuesN) + '%';
        stage.append(
          svg('path', { d: 'M5 5H' + (width - 38), stroke: '#343f44', 'stroke-dasharray': '2 4' }),
        );
        symbols.append(yLabel);
      }
      stage.append(symbols);
      cursor = svg('g', { class: 'at-signal-cursor' });
      cursor.append(
        svg('path', { d: 'M0 5V' + base, stroke: '#e9dfbd', 'stroke-width': 1 }),
        svg('path', { d: 'M-3.2 1h6.4L0 5z', fill: '#edcf91' }),
      );
      stage.append(cursor);
      plot.setAttribute('viewBox', '0 0 ' + width + ' 40');
      plot.replaceChildren(stage);
      ticks.replaceChildren();
      if (range[0] === range[1]) tick(range[0], 50, 'middle');
      else {
        tick(range[0], 0, 'start');
        const pct = 100 * position(0, range);
        if (signed && (pct * width) / 100 > 25 && ((100 - pct) * width) / 100 > 25)
          tick(0, pct, 'middle');
        tick(range[1], 100, 'end');
      }
      const nLabel = cells ? cellLabel() : meanLabel();
      const markDefinition = cells
        ? 'Bar height gives the share of measured cells in each interval. The top line gives the largest bin share.'
        : layout.dots
          ? 'One dot per participant. The bracket spans the middle 50%.'
          : 'Bars count participants in each interval.';
      const common = ' The scale is shared across visits, arms and cell types.';
      chart.title = markDefinition + common;
      chart.setAttribute(
        'aria-label',
        [
          name.textContent,
          feature?.label || '',
          visit,
          arm,
          nLabel,
          markDefinition,
          feature?.unit,
          'Use the arrow keys to inspect ' + (layout.bins ? 'bins.' : 'values.'),
        ]
          .filter(Boolean)
          .join('. '),
      );
      detail.title = nLabel + '. ' + markDefinition + common;
      readout(-1);
    }
    function pointer(event) {
      if (!layout || !domain()) return;
      const px = clamp(event.clientX - chart.getBoundingClientRect().left, 5, width - 5);
      if (layout.bins) {
        let active = layout.bins.findIndex((b) => px <= b.x + b.width);
        if (active < 0) active = layout.bins.length - 1;
        readout(active);
      } else {
        const py =
          ((event.clientY - plot.getBoundingClientRect().top) * 40) /
          (plot.getBoundingClientRect().height || 40);
        let nearest = 0,
          distance = Infinity;
        layout.dots.forEach((dot, i) => {
          const d = Math.hypot(dot.x - px, 33.8 - dot.lane * 6.4 - py);
          if (d < distance) {
            distance = d;
            nearest = i;
          }
        });
        readout(nearest);
      }
    }
    for (const type of ['pointermove', 'pointerdown']) on(chart, type, pointer);
    for (const type of ['pointerleave', 'blur'])
      on(chart, type, () => {
        if (activeIndex >= 0) readout(-1);
      });
    on(chart, 'keydown', (e) => {
      if (!layout) return;
      const length = layout.bins?.length || layout.dots.length;
      const focus = activeIndex < 0 ? Math.floor((length - 1) / 2) : activeIndex;
      const keys = { ArrowLeft: focus - 1, ArrowRight: focus + 1, Home: 0, End: length - 1 };
      if (e.key === 'Escape') readout(-1);
      else if (Object.hasOwn(keys, e.key)) {
        e.preventDefault();
        readout(clamp(keys[e.key], 0, length - 1), true);
      }
    });
    for (const [button, unit] of [
      [cellButton, 'cells'],
      [personButton, 'participants'],
    ]) {
      on(button, 'click', () => {
        if (view !== unit) {
          view = unit;
          paint();
        }
      });
    }
    function update(args) {
      const next = args.selectedGroup ?? args.groups?.find((g) => g.name === args.selected);
      const hide = args.mode === 'cells' || !args.feature;
      const available = !!(
        args.cellDomain ||
        next?.cellBins ||
        args.groups?.some((g) => g.cellBins)
      );
      const same =
        host.hidden === hide &&
        hasCells === available &&
        feature === args.feature &&
        visit === args.visit &&
        arm === args.arm &&
        participantMeans === args.participantMeans &&
        selection === args.selected &&
        group?.values === next?.values &&
        group?.cellBins === next?.cellBins &&
        scale?.[0] === args.scale?.[0] &&
        scale?.[1] === args.scale?.[1];
      ({ feature, visit, arm, participantMeans, scale } = args);
      selection = args.selected;
      group = next;
      host.hidden = hide;
      hasCells = available;
      if (!hide && !same) paint();
    }
    const observer = new ResizeObserver(() => {
      if (!host.hidden && !chart.hidden && Math.max(30, Math.round(chart.clientWidth)) !== width)
        paint();
    });
    observer.observe(chart);
    return {
      update,
      destroy() {
        events.abort();
        observer.disconnect();
        host.replaceChildren();
      },
    };
  }

  window.SctrialAtlasSignal = Object.freeze({
    color,
    gradient,
    mini,
    mount,
    zeroColor: 'rgb(' + zeroColor.join(',') + ')',
    missingColor: 'rgb(' + missingColor.join(',') + ')',
  });
})();
