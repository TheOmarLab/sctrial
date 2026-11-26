(() => {
  'use strict';

  const finite = Number.isFinite;
  const svgNS = 'http://www.w3.org/2000/svg';

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined && content !== null) node.textContent = content;
    return node;
  }

  function shape(tag, attributes, title) {
    const node = document.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    if (title) {
      const label = document.createElementNS(svgNS, 'title');
      label.textContent = title;
      node.append(label);
    }
    return node;
  }

  function number(value, signed = false, digits = 3) {
    if (!finite(value)) return '—';
    if (value === 0) return '0';
    const magnitude = Math.abs(value);
    const formatted =
      magnitude < 0.001 || magnitude >= 1e6
        ? magnitude.toExponential(Math.max(0, digits - 1)).replace(/\.?(0+)(?=e)/, '')
        : magnitude.toLocaleString('en-US', { maximumSignificantDigits: digits });
    return (value < 0 ? '−' : signed ? '+' : '') + formatted;
  }

  function scale(values, includeZero) {
    let lo = includeZero ? 0 : Infinity;
    let hi = includeZero ? 0 : -Infinity;
    for (const value of values) {
      if (!finite(value)) continue;
      lo = Math.min(lo, value);
      hi = Math.max(hi, value);
    }
    if (!finite(lo) || !finite(hi)) return null;
    if (lo === hi) {
      const padding = lo === 0 ? 1 : Math.abs(lo) * 0.1;
      lo -= padding;
      hi += padding;
    }
    const width = hi - lo;
    if (!finite(width) || width === 0) return null;
    const rough = width / 4;
    const power = 10 ** Math.floor(Math.log10(rough));
    if (!finite(power) || power === 0) return null;
    const multiple = [1, 2, 2.5, 5, 10].find((step) => step >= rough / power) || 10;
    const step = multiple * power;
    const first = Math.floor(lo / step);
    const last = Math.ceil(hi / step);
    const ticks = [];
    for (let index = first; index <= last && ticks.length <= 7; index++) {
      ticks.push(index === 0 ? 0 : +(index * step).toPrecision(14));
    }
    const min = first * step;
    const max = last * step;
    if (!finite(min) || !finite(max) || min === max) return null;
    return { ticks, position: (value) => 100 * ((value - min) / (max - min)) + '%' };
  }

  function axis(mapping, unit) {
    const row = element('div', 'sce-axis');
    row.setAttribute('aria-hidden', 'true');
    for (const value of mapping.ticks) {
      const label = element(
        'span',
        value === 0 ? 'sce-tick sce-zero' : 'sce-tick',
        number(value, false, 5),
      );
      label.style.left = mapping.position(value);
      row.append(label);
    }
    if (unit) row.title = unit;
    return row;
  }

  function track(mapping, height) {
    const plot = shape('svg', { class: 'sce-track', width: '100%', height, 'aria-hidden': 'true' });
    for (const tick of mapping.ticks) {
      const x = mapping.position(tick);
      plot.append(
        shape('line', {
          class: tick === 0 ? 'sce-grid sce-zero-line' : 'sce-grid',
          x1: x,
          x2: x,
          y1: 0,
          y2: height,
        }),
      );
    }
    return plot;
  }

  function ciFor(estimate, interval) {
    return finite(estimate) &&
      Array.isArray(interval) &&
      interval.length === 2 &&
      interval.every(finite) &&
      interval[0] <= interval[1]
      ? interval
      : null;
  }

  function effectChart(config, interval) {
    const block = element('div', 'sce-effect');
    const summary = element('div', 'sce-summary');
    const between = config.arms.length > 1;
    const estimand = config.longitudinal
      ? between
        ? 'Difference in change'
        : 'Mean paired change'
      : between
        ? 'Between-arm difference'
        : 'Participant mean';
    summary.append(
      element(
        'h3',
        'sce-label',
        config.descriptive ? 'Descriptive ' + estimand.toLowerCase() : estimand,
      ),
    );
    const sign = config.longitudinal || config.arms.length > 1;
    summary.append(element('strong', 'sce-effect-number', number(config.estimate, sign)));
    if (config.unit) summary.append(element('span', 'sce-summary-unit', config.unit));
    block.append(summary);

    const visual = element('div', 'sce-model-chart');
    if (!finite(config.estimate)) {
      const status = element('div', 'sce-no-estimate', config.status || 'Estimate unavailable');
      status.setAttribute('role', 'status');
      visual.append(status);
      block.append(visual);
      return block;
    }

    if (interval) {
      const caption = element('div', 'sce-ci-caption');
      caption.append(
        element('span', '', '95% CI'),
        element('strong', '', number(interval[0], sign) + ' to ' + number(interval[1], sign)),
      );
      visual.append(caption);
    } else {
      visual.append(
        element(
          'div',
          'sce-ci-caption',
          config.descriptive ? 'Observed summary' : '95% CI unavailable',
        ),
      );
    }

    const mapping = scale([config.estimate, ...(interval || [])], true);
    if (mapping) {
      const plot = track(mapping, 29);
      const x = mapping.position(config.estimate);
      if (interval) {
        const left = mapping.position(interval[0]);
        const right = mapping.position(interval[1]);
        plot.append(shape('line', { class: 'sce-whisker', x1: left, x2: right, y1: 15, y2: 15 }));
        for (const end of [left, right]) {
          plot.append(shape('line', { class: 'sce-cap', x1: end, x2: end, y1: 10, y2: 20 }));
        }
      }
      plot.append(shape('circle', { class: 'sce-estimate-point', cx: x, cy: 15, r: 4.8 }));
      visual.append(plot, axis(mapping, config.unit));
    }

    block.append(visual);
    return block;
  }

  function pointLayout(group, mapping, width) {
    if (group.readings.length > 90) return null;
    const lanes = [0, -1, 1, -2, 2, -3, 3];
    const occupied = lanes.map(() => -Infinity);
    const points = [];
    let extent = 0;
    for (const reading of group.readings) {
      const x = parseFloat(mapping.position(reading.value));
      const pixel = (width * x) / 100;
      const lane = occupied.findIndex((last) => pixel - last >= 9.5);
      if (lane < 0) return null;
      occupied[lane] = pixel;
      extent = Math.max(extent, Math.abs(lanes[lane]));
      points.push({ reading, x, pixel, offset: lanes[lane] * 9.5, hitRadius: 12 });
    }
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const a = points[i];
        const b = points[j];
        if (b.pixel - a.pixel > 13) break;
        const separation = Math.hypot(b.pixel - a.pixel, b.offset - a.offset);
        const radius = Math.max(8.7, separation - 0.8);
        a.hitRadius = Math.min(a.hitRadius, radius);
        b.hitRadius = Math.min(b.hitRadius, radius);
      }
    }
    return { height: Math.max(32, 18 + 19 * extent), points };
  }

  function binsFor(readings, mapping) {
    const counts = new Uint32Array(44);
    for (const { value } of readings) {
      const fraction = parseFloat(mapping.position(value)) / 100;
      const index = Math.max(0, Math.min(counts.length - 1, Math.floor(fraction * counts.length)));
      counts[index] += 1;
    }
    return counts;
  }

  function showBins(plot, bins, largest, markClass) {
    bins.forEach((count, i) => {
      if (!count) return;
      const height = (24 * count) / largest;
      plot.append(
        shape(
          'rect',
          {
            class: markClass,
            x: (100 * (i + 0.05)) / bins.length + '%',
            y: 29 - height,
            width: (100 * 0.9) / bins.length + '%',
            height,
          },
          count + (count === 1 ? ' participant' : ' participants'),
        ),
      );
    });
  }

  function wireReadings(plot, items, tooltip, onSelect) {
    if (!items.size) return;
    const buttons = [...items.keys()];
    plot.removeAttribute('aria-hidden');
    plot.setAttribute('role', 'group');
    plot.setAttribute(
      'aria-description',
      'Use the arrow keys to move between participants. Press Enter to open a record.',
    );
    buttons[0].setAttribute('tabindex', '0');

    const markAt = (target) => target?.closest?.('.sce-observation');
    function identify(button) {
      const item = items.get(button);
      if (!item) {
        tooltip.hidden = true;
        return;
      }
      tooltip.textContent = item.label;
      tooltip.style.left = item.x + '%';
      tooltip.dataset.side = item.x < 18 ? 'start' : item.x > 82 ? 'end' : 'center';
      tooltip.hidden = false;
    }
    plot.addEventListener('pointerover', (event) => identify(markAt(event.target)));
    plot.addEventListener('pointerout', (event) =>
      identify(markAt(event.relatedTarget) || markAt(document.activeElement)),
    );
    plot.addEventListener('focusin', (event) => {
      const current = markAt(event.target);
      for (const button of buttons)
        button.setAttribute('tabindex', current === button ? '0' : '-1');
      identify(current);
    });
    plot.addEventListener('focusout', (event) => identify(markAt(event.relatedTarget)));
    plot.addEventListener('click', (event) => {
      const item = items.get(markAt(event.target));
      if (item) onSelect(item.id);
    });
    plot.addEventListener('keydown', (event) => {
      const button = markAt(event.target);
      const index = buttons.indexOf(button);
      if (index < 0) return;
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect(items.get(button).id);
        return;
      }
      if (event.key === 'Escape' && !tooltip.hidden) {
        tooltip.hidden = true;
        event.stopPropagation();
        return;
      }
      const next = {
        ArrowRight: index + 1,
        ArrowLeft: index - 1,
        ArrowDown: index + 1,
        ArrowUp: index - 1,
        Home: 0,
        End: buttons.length - 1,
      }[event.key];
      if (next !== undefined) {
        event.preventDefault();
        buttons[Math.min(buttons.length - 1, Math.max(0, next))].focus();
      }
    });
  }

  function showPoints(plot, layout, markClass, group, config, tooltip) {
    const items = new Map();
    for (const { reading, x, offset, hitRadius } of layout.points) {
      const y = layout.height / 2 + offset;
      const label = String(reading.id ?? group.name) + ' · ' + number(reading.value, true, 5);
      const dot = shape('circle', { class: markClass, cx: x + '%', cy: y, r: 3.7 });
      if (
        reading.id === undefined ||
        reading.id === null ||
        typeof config.onSelect !== 'function'
      ) {
        const staticMark = shape('g', {}, label);
        staticMark.append(dot);
        plot.append(staticMark);
        continue;
      }
      const button = shape('g', {
        class: 'sce-observation',
        role: 'button',
        tabindex: '-1',
        'aria-label': 'Inspect ' + label,
      });
      button.append(shape('circle', { class: 'sce-hit', cx: x + '%', cy: y, r: hitRadius }), dot);
      items.set(button, { id: reading.id, label, x });
      plot.append(button);
    }
    plot.setAttribute('aria-label', group.name + ' participant observations');
    wireReadings(plot, items, tooltip, config.onSelect);
    return items.size;
  }

  function matrixLink(config) {
    const table = document.getElementById('participantMatrix');
    if (!table && !config.onOpenMatrix) return null;
    const button = element('button', 'sce-open-matrix', 'Inspect participants');
    button.type = 'button';
    button.onclick = () => {
      if (config.onOpenMatrix) {
        config.onOpenMatrix();
      } else {
        table.scrollIntoView({ block: 'nearest' });
        table.querySelector('input, button')?.focus({ preventScroll: true });
      }
    };
    return button;
  }

  function groupSummary(arm) {
    const supplied = Array.isArray(arm.values);
    const readings = [];
    if (supplied)
      arm.values.forEach((value, index) => {
        if (finite(value)) readings.push({ value, id: arm.ids?.[index] });
      });
    const n = supplied ? readings.length : Number.isInteger(arm.n) && arm.n >= 0 ? arm.n : null;
    const mean =
      finite(arm.mean) && n !== 0
        ? arm.mean
        : readings.length
          ? readings.reduce((sum, reading) => sum + reading.value / readings.length, 0)
          : null;
    if (readings.length <= 90) readings.sort((a, b) => a.value - b.value);
    return { name: String(arm.name ?? ''), colorIndex: arm.colorIndex, n, mean, readings };
  }

  const observers = new WeakMap();
  function observations(panel, config) {
    const area = element('section', 'sce-observations');
    const groups = config.arms.map(groupSummary);
    const all = groups.flatMap((group) =>
      group.readings.map((reading) => reading.value).concat(finite(group.mean) ? [group.mean] : []),
    );
    const mapping = all.length ? scale(all, config.longitudinal) : null;
    const head = element('div', 'sce-observed-head');
    head.append(
      element(
        'h4',
        'sce-label',
        config.longitudinal ? 'Paired participant changes' : 'Participant scores',
      ),
    );
    const visit = config.longitudinal
      ? [config.visit, config.baseline].filter(Boolean).join(' − ')
      : config.visit;
    if (visit) head.append(element('span', 'sce-visit', visit));
    area.append(head);
    const data = element('div', 'sce-data');
    const lanes = groups.map((group) => {
      const label = element('div', 'sce-arm-label');
      label.append(
        element('strong', '', group.name),
        element('small', '', 'n = ' + (group.n ?? '—')),
      );
      const lane = element('div', 'sce-obs-lane');
      const amount = element('div', 'sce-group-mean', number(group.mean, config.longitudinal));
      amount.setAttribute(
        'aria-label',
        group.name + ' mean ' + number(group.mean, config.longitudinal, 5),
      );
      data.append(label, lane, amount);
      return lane;
    });
    if (mapping)
      data.append(
        element('span'),
        axis(mapping, config.unit),
        element('small', 'sce-mean-caption', 'Mean'),
      );
    const legend = element('div', 'sce-legend');
    area.append(data, legend);
    panel.append(area);

    let drawnWidth = -1;
    function draw() {
      const width = Math.round(lanes[0].getBoundingClientRect().width);
      if (drawnWidth === width) return;
      drawnWidth = width;
      const focused = area.contains(document.activeElement)
        ? document.activeElement.getAttribute('aria-label')
        : '';
      const layouts = groups.map((group) => mapping && pointLayout(group, mapping, width));
      const binned = mapping && layouts.some((layout) => !layout);
      const counts = binned ? groups.map((group) => binsFor(group.readings, mapping)) : [];
      const largest = counts.length ? Math.max(...counts.map((bins) => Math.max(...bins))) : 0;
      let individual = false;
      groups.forEach((group, index) => {
        const lane = lanes[index];
        lane.replaceChildren();
        if (!mapping || (!group.readings.length && !finite(group.mean))) {
          lane.append(
            element(
              'span',
              'sce-obs-empty',
              group.n === 0 ? 'No contributing participants' : 'Values unavailable',
            ),
          );
          return;
        }
        const height = binned ? 32 : layouts[index].height;
        const plot = track(mapping, height);
        const tooltip = element('span', 'sce-point-tip');
        tooltip.setAttribute('aria-hidden', 'true');
        tooltip.hidden = true;
        const markClass = 'sce-arm-mark sce-arm-' + ((group.colorIndex ?? index) % 2);
        if (binned) {
          showBins(plot, counts[index], largest, markClass);
        } else {
          const count = showPoints(plot, layouts[index], markClass, group, config, tooltip);
          individual = individual || count > 0;
        }
        if (finite(group.mean)) {
          const x = mapping.position(group.mean);
          for (const cls of ['sce-mean-halo', 'sce-mean']) {
            plot.append(shape('line', { class: cls, x1: x, x2: x, y1: 5, y2: height - 5 }));
          }
        }
        lane.append(plot, tooltip);
      });
      if (focused) {
        const replacement = [...area.querySelectorAll('.sce-observation')].find(
          (node) => node.getAttribute('aria-label') === focused,
        );
        replacement?.focus({ preventScroll: true });
      }
      legend.replaceChildren();
      if (!groups.some((group) => group.readings.length)) return;
      const marker = element('i', binned ? 'sce-bin-key' : 'sce-dot-key');
      const meanKey = element('i', 'sce-mean-key');
      marker.setAttribute('aria-hidden', 'true');
      meanKey.setAttribute('aria-hidden', 'true');
      legend.append(
        marker,
        element(
          'span',
          '',
          binned
            ? 'Participant counts per bin'
            : individual
              ? 'Select a participant'
              : 'One participant',
        ),
        meanKey,
        element('span', '', 'Group mean'),
      );
      if (!individual) {
        const link = matrixLink(config);
        if (link) legend.append(link);
      }
    }
    draw();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      if (area.isConnected) draw();
      else observer.disconnect();
    });
    observer.observe(area);
    observers.set(panel.parentElement, observer);
  }

  function render(container, config) {
    if (!container || !config) return;
    const options = { ...config, arms: Array.isArray(config.arms) ? config.arms : [] };
    const panel = element('section', 'sce');
    panel.setAttribute(
      'aria-label',
      ['Effect estimate', options.feature].filter(Boolean).join(' for '),
    );
    if (options.model) {
      const details = element('details', 'sce-details');
      const summary = element('summary', 'sce-context');
      if (options.contrast) summary.append(element('strong', '', options.contrast));
      summary.append(
        element('span', 'sce-method-key', options.descriptive ? 'Calculation' : 'Model details'),
      );
      details.append(summary, element('p', 'sce-model-description', options.model));
      panel.append(details);
    } else if (options.contrast) {
      const head = element('div', 'sce-context');
      head.append(element('strong', '', options.contrast));
      panel.append(head);
    }
    panel.append(effectChart(options, ciFor(options.estimate, options.interval)));
    observers.get(container)?.disconnect();
    container.replaceChildren(panel);
    if (options.arms.length) observations(panel, options);
  }

  window.SctrialEvidence = Object.freeze({ render });
})();
