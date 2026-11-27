const fmtCount = new Intl.NumberFormat('en-US');
const visits = ['pre', 'post'];
const cache = new WeakMap();
const memberViews = new WeakMap();

function safe(value) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]
  );
}
function num(value, digits = 2) {
  return Number.isFinite(value)
    ? value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : 'n/a';
}
function pct(value, digits = 1) {
  return Number.isFinite(value) ? num(value * 100, digits) + '%' : 'n/a';
}
function signed(value, digits) {
  if (!Number.isFinite(value)) return 'n/a';
  const rounded = Number(value.toFixed(digits));
  return (rounded > 0 ? '+' : rounded < 0 ? '−' : '') + num(Math.abs(rounded), digits);
}
function cap(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  return ([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((step) => step * power >= value) || 10) * power;
}
function metric(label, value, unit = '') {
  return (
    '<div class="ev-metric"><span>' +
    safe(label) +
    '</span><strong>' +
    safe(value) +
    (unit ? '<small>' + safe(unit) + '</small>' : '') +
    '</strong></div>'
  );
}
function empty(message) {
  return '<p class="ev-empty">' + safe(message) + '</p>';
}
function visitLabel(study, id) {
  return study.visits.find((item) => item.id === id)?.label || id;
}
function legend(study) {
  return (
    '<div class="ev-legend">' +
    visits
      .map((visit) => '<span class="ev-' + visit + '"><i></i>' + safe(visitLabel(study, visit)) + '</span>')
      .join('') +
    '</div>'
  );
}
function selector(features, feature, layer) {
  const key = layer === 'expression' ? 'gene' : 'pathway';
  const label = key === 'gene' ? 'Gene' : 'Pathway';
  if (!feature) return '';
  return (
    '<label class="ev-select"><span>' +
    label +
    '</span><select aria-label="' +
    label +
    '" data-ev-select="' +
    key +
    '">' +
    features
      .map(
        (f) =>
          '<option value="' +
          safe(f.id) +
          '"' +
          (feature.id === f.id ? ' selected' : '') +
          '>' +
          safe(f.label || f.id) +
          '</option>'
      )
      .join('') +
    '</select></label>'
  );
}
function groupCounts(groups, study, value, label) {
  return (
    '<div class="ev-endpoints">' +
    groups
      .map(
        (g, i) =>
          '<div class="ev-end ev-' +
          visits[i] +
          '"><span><i></i>' +
          safe(visitLabel(study, visits[i])) +
          '</span><strong>' +
          safe(g ? value(g) : 'n/a') +
          '</strong><small>' +
          safe(g ? label(g) : 'Unavailable') +
          '</small></div>'
      )
      .join('') +
    '</div>'
  );
}

function observedPairs(groups) {
  return (groups.find((g) => g?.participantPairs)?.participantPairs || []).filter(
    (p) => Number.isFinite(p.pre?.fraction) && Number.isFinite(p.post?.fraction)
  );
}
function fractions(state, groups) {
  const [pre, post] = groups;
  const current = groups[state.visit === 'post' ? 1 : 0];
  const pairs = observedPairs(groups);
  if (!current || !groups.some((g) => Number.isFinite(g?.fraction)))
    return empty('No composition measurements for this selection.');
  const both = Number.isFinite(pre?.fraction) && Number.isFinite(post?.fraction);
  const difference = both ? (post.fraction - pre.fraction) * 100 : null;
  const values = groups
    .filter(Boolean)
    .map((g) => g.fraction)
    .concat(pairs.flatMap((p) => [p.pre.fraction, p.post.fraction]));
  const max = Math.min(1, cap(values.reduce((max, v) => Math.max(max, v ?? 0), 0)));
  const y = (v) => 94 - (77 * v) / max;
  const chart =
    '<svg viewBox="0 0 272 106" role="img" aria-label="Pooled cell fractions for both visits.' +
    (pairs.length ? ' Thin lines connect observations from each paired participant.' : '') +
    '">' +
    [0, max / 2, max]
      .map(
        (v) =>
          '<text x="1" y="' +
          (y(v) + 3) +
          '" class="ev-axis">' +
          pct(v, max < 0.1 ? 1 : 0) +
          '</text><path d="M40 ' +
          y(v) +
          ' H260" class="ev-grid"/>'
      )
      .join('') +
    pairs
      .map(
        (p) =>
          '<path d="M45 ' +
          y(p.pre.fraction) +
          ' L251 ' +
          y(p.post.fraction) +
          '" data-y-pre="' +
          y(p.pre.fraction) +
          '" data-y-post="' +
          y(p.post.fraction) +
          '" class="ev-pair"/>'
      )
      .join('') +
    (both
      ? '<path d="M45 ' + y(pre.fraction) + ' L251 ' + y(post.fraction) + '" class="ev-pooled"/>'
      : '') +
    groups
      .map((g, i) =>
        Number.isFinite(g?.fraction)
          ? '<circle cx="' +
            (i ? 251 : 45) +
            '" cy="' +
            y(g.fraction) +
            '" r="5" class="ev-dot ev-' +
            visits[i] +
            '"/>'
          : ''
      )
      .join('') +
    '</svg>';
  return (
    '<h2>Cell fraction</h2><div class="ev-metrics">' +
    metric(visitLabel(state.study, state.visit), pct(current.fraction)) +
    metric('Change across visits', signed(difference, 1), 'pp') +
    '</div>' +
    '<figure class="ev-chart"><div data-ev-slopes' +
    (pairs.length
      ? ' tabindex="0" role="slider" aria-label="Inspect paired participants" aria-valuemin="0" aria-valuemax="' +
        pairs.length +
        '" aria-valuenow="0" aria-valuetext="No participant selected"'
      : '') +
    '>' +
    chart +
    '</div><figcaption><span class="ev-line-key"></span>Pooled cells' +
    (pairs.length
      ? '<span class="ev-line-key ev-dim"></span>' + pairs.length + ' paired participants'
      : '') +
    '</figcaption></figure>' +
    (pairs.length
      ? '<div class="ev-pair-controls"><span data-ev-pair-caption>Inspect a participant line</span><button type="button" data-ev-step="-1" aria-label="Inspect previous participant">‹</button><button type="button" data-ev-step="1" aria-label="Inspect next participant">›</button></div><div class="ev-pair-detail" data-ev-pair-detail hidden></div>'
      : '') +
    '<h3 class="ev-pool-heading">Pooled cell counts</h3>' +
    groupCounts(
      groups,
      state.study,
      (g) => pct(g.fraction),
      (g) => fmtCount.format(g.cells) + ' / ' + fmtCount.format(g.totalCells) + ' cells'
    )
  );
}

function expressionPlot(state, groups, feature) {
  const ss = groups.map((g) => g?.signals?.[feature.id]);
  if (!groups.some((g) => g?.cells) || !ss.some((s) => s?.bins?.length))
    return empty('No cells were captured in this population at either visit.');
  const size = ss.find((s) => s?.bins?.length).bins.length;
  const hi = feature.domain[1];
  const lows = feature.domain[0];
  if (!(hi > lows))
    return empty('Every observed cell has zero expression for this gene at the selected visits.');
  const sets = ss.map((s, i) => {
    if (!s || !groups[i].cells) return null;
    return s.bins.map((count, j) => count - (j === 0 ? s.zeros : 0));
  });
  const rates = sets.map((counts, j) => (counts ? counts.map((v) => v / groups[j].cells) : []));
  const max = cap(Math.max(0, ...rates.flat()));
  const x0 = 38,
    x1 = 266,
    top = 13,
    bottom = 95;
  const y = (p) => bottom - ((bottom - top) * p) / max;
  const x = (i) => x0 + (i * (x1 - x0)) / size;
  let marks = '';
  for (let j = 0; j < size; j++) {
    const step = (x1 - x0) / size;
    for (let i = 0; i < 2; i++) {
      if (!sets[i]) continue;
      const h = rates[i][j];
      marks +=
        '<rect class="ev-bar ev-' +
        visits[i] +
        '" x="' +
        (x(j) + step * (i ? 0.49 : 0.02)) +
        '" width="' +
        step * 0.46 +
        '" y="' +
        y(h) +
        '" height="' +
        (bottom - y(h)) +
        '"/>';
    }
  }
  const ticks = [0, max / 2, max]
    .map(
      (v) =>
        '<path class="ev-grid" d="M' +
        x0 +
        ' ' +
        y(v) +
        ' H' +
        x1 +
        '"/><text x="0" y="' +
        (y(v) + 3) +
        '" class="ev-axis">' +
        pct(v, max < 0.1 ? 1 : 0) +
        '</text>'
    )
    .join('');
  const axes = [0, 0.5, 1]
    .map(
      (f) =>
        '<text x="' +
        (x0 + f * (x1 - x0)) +
        '" y="112" class="ev-axis" text-anchor="' +
        (f === 0 ? 'start' : f === 1 ? 'end' : 'middle') +
        '">' +
        num(lows + (hi - lows) * f, 1) +
        '</text>'
    )
    .join('');
  const focus =
    '<rect class="ev-bin-highlight" data-ev-highlight fill="currentColor" opacity=".13" x="' +
    x0 +
    '" y="' +
    top +
    '" width="' +
    (x1 - x0) / size +
    '" height="' +
    (bottom - top) +
    '" visibility="hidden"/>';
  const chart = '<svg viewBox="0 0 272 119" aria-hidden="true">' + ticks + marks + focus + axes + '</svg>';
  const detected = groupCounts(
    groups,
    state.study,
    (g) => {
      const s = g.signals[feature.id];
      return s ? pct((g.cells - s.zeros) / g.cells) : 'n/a';
    },
    (g) => {
      const s = g.signals[feature.id];
      return s
        ? fmtCount.format(g.cells - s.zeros) + ' / ' + fmtCount.format(g.cells) + ' cells'
        : 'Unavailable';
    }
  );
  return (
    '<div class="ev-section"><h3>Cells with nonzero expression</h3>' +
    detected +
    '</div>' +
    '<div class="ev-section"><div class="ev-chart-head"><h3>Positive expression</h3>' +
    legend(state.study) +
    '</div>' +
    '<p class="ev-axis-label">Cells per bin (% of all cells)</p><div class="ev-hist" tabindex="0" role="slider" aria-label="Inspect expression bins" aria-valuemin="1" aria-valuemax="' +
    size +
    '" aria-valuenow="1" data-ev-hist>' +
    chart +
    '</div><div class="ev-xunit">' +
    safe(feature.unit || 'expression units') +
    '</div><div class="ev-bin-read" aria-live="off" data-ev-bin>Inspect a bin to compare cell counts.</div></div>'
  );
}

function expression(state, groups, feature, choices) {
  if (!feature) return empty('No measured genes in this study.');
  const i = state.visit === 'post' ? 1 : 0;
  const current = groups[i]?.signals?.[feature.id];
  return (
    selector(choices, feature, 'expression') +
    '<div class="ev-metrics ev-one">' +
    metric('Mean expression across cells', num(current?.mean, 2), feature.unit || '') +
    '</div>' +
    expressionPlot(state, groups, feature)
  );
}

function memberUnit(feature) {
  return feature.memberUnit || feature.unit?.replace(/^mean member /i, '') || 'expression';
}
function memberRows(feature, groups, study) {
  const ss = groups.map((g) => g?.signals?.[feature.id]);
  const both = groups.every((g) => g?.cells > 0);
  const rank = (row) => (both ? Math.abs(row.post - row.pre) : (row.pre ?? row.post));
  const rows = (feature.members || [])
    .map((gene, i) => ({ gene, pre: ss[0]?.memberMeans?.[i], post: ss[1]?.memberMeans?.[i] }))
    .filter((r) => Number.isFinite(r.pre) || Number.isFinite(r.post))
    .sort((a, b) => rank(b) - rank(a) || a.gene.localeCompare(b.gene));
  if (!rows.length) return empty('No cells were captured in this population at either visit.');
  const limit = cap(rows.reduce((mx, row) => Math.max(mx, row.pre ?? 0, row.post ?? 0), 0));
  const color =
    '<div class="ev-member-scale"><span>Gene</span><span>0</span><span>' +
    safe(num(limit, 1)) +
    '</span></div>';
  return (
    '<div class="ev-section"><div class="ev-chart-head"><h3>Member expression</h3>' +
    legend(study) +
    '</div>' +
    '<p class="ev-note">Mean ' +
    safe(memberUnit(feature)) +
    ' per cell, ordered by ' +
    (both ? 'absolute change' : 'expression at the observed visit') +
    '.</p>' +
    color +
    '<div class="ev-member-list" role="region" tabindex="0" aria-label="Expression across all ' +
    rows.length +
    ' measured pathway members">' +
    rows
      .map((row, index) => {
        const a = Number.isFinite(row.pre) ? (100 * row.pre) / limit : null;
        const b = Number.isFinite(row.post) ? (100 * row.post) / limit : null;
        const link =
          a != null && b != null
            ? '<i class="ev-segment" style="left:' +
              Math.min(a, b) +
              '%;width:' +
              Math.abs(a - b) +
              '%"></i>'
            : '';
        const dots = [a, b]
          .map((v, i) =>
            v == null ? '' : '<i class="ev-marker ev-' + visits[i] + '" style="left:' + v + '%"></i>'
          )
          .join('');
        const values = [row.pre, row.post];
        const label =
          row.gene +
          ' · ' +
          visits
            .map(
              (v, i) =>
                visitLabel(study, v) +
                ' ' +
                (Number.isFinite(values[i]) ? Number(values[i].toPrecision(4)) : 'n/a')
            )
            .join(' · ');
        const exact = visits
          .map(
            (v, i) => visitLabel(study, v) + ' ' + (Number.isFinite(values[i]) ? String(values[i]) : 'n/a')
          )
          .join(' · ');
        return (
          '<button type="button" class="ev-member" data-ev-member data-ev-exact="' +
          safe(exact) +
          '" tabindex="' +
          (index === 0 ? '0' : '-1') +
          '" aria-pressed="' +
          (index === 0) +
          '" aria-label="' +
          safe(label) +
          '"><span>' +
          safe(row.gene) +
          '</span><span class="ev-rail">' +
          link +
          dots +
          '</span></button>'
        );
      })
      .join('') +
    '</div><details class="ev-member-precision"><summary class="ev-member-read" data-ev-member-read></summary><p data-ev-member-exact></p></details><div class="ev-row-count">' +
    rows.length +
    ' measured members</div></div>'
  );
}
function pathway(state, groups, feature, choices) {
  if (!feature) return empty('No measured gene sets in this study.');
  const source = /^R-[A-Z]{3}-[0-9]+$/.test(feature.accession) ? feature.accession : '';
  const value = (g) => num(g.signals[feature.id]?.mean, 3);
  const memberCount = feature.memberCount ?? feature.members?.length;
  const coverage = Number.isFinite(feature.sourceMemberCount)
    ? memberCount + ' of ' + feature.sourceMemberCount
    : String(memberCount);
  return (
    selector(choices, feature, 'pathway') +
    '<div class="ev-section"><h3>Mean gene set score</h3>' +
    groupCounts(groups, state.study, value, (g) => fmtCount.format(g.cells) + ' cells') +
    '</div>' +
    memberRows(feature, groups, state.study) +
    '<details class="ev-method"><summary>Score and gene set</summary><p>Each cell’s score averages ' +
    safe(memberUnit(feature)) +
    ' across ' +
    safe(coverage) +
    ' member genes. Each visit value averages the scores across cells.</p>' +
    (source
      ? '<a href="https://reactome.org/content/detail/' +
        encodeURIComponent(source) +
        '" target="_blank" rel="noopener noreferrer">Reactome ' +
        safe(source) +
        ' ↗</a>'
      : '') +
    '</details>'
  );
}
function bindHistogram(element, groups, feature, study) {
  const hist = element.querySelector('[data-ev-hist]');
  if (!hist) return;
  const signals = groups.map((g) => g?.signals?.[feature.id]);
  const size = signals.find((s) => s?.bins)?.bins.length;
  if (!size) return;
  const highlight = hist.querySelector('[data-ev-highlight]');
  const readout = element.querySelector('[data-ev-bin]');
  const xl = feature.domain[0],
    dx = (feature.domain[1] - xl) / size;
  let last = 0;
  const show = (i) => {
    last = Math.max(0, Math.min(size - 1, i));
    const a = num(xl + dx * last),
      b = num(xl + dx * (last + 1));
    const counts = signals.map((s) =>
      s ? fmtCount.format(s.bins[last] - (last === 0 ? s.zeros : 0)) : 'n/a'
    );
    const range = (last === 0 ? '(' : '[') + a + ', ' + b + (last === size - 1 ? ']' : ')');
    const label =
      range +
      ' · ' +
      visitLabel(study, 'pre') +
      ' ' +
      counts[0] +
      ' · ' +
      visitLabel(study, 'post') +
      ' ' +
      counts[1];
    highlight.setAttribute('visibility', 'visible');
    highlight.setAttribute('x', String(38 + ((266 - 38) * last) / size));
    hist.setAttribute('aria-valuenow', String(last + 1));
    hist.setAttribute('aria-valuetext', label);
    readout.textContent = label;
  };
  const atPointer = (event) => {
    const svg = hist.querySelector('svg');
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    show(Math.floor((size * (point.x - 38)) / (266 - 38)));
  };
  hist.addEventListener('pointermove', atPointer);
  hist.addEventListener('pointerdown', atPointer);
  hist.addEventListener('focus', () => show(last));
  hist.addEventListener('keydown', (event) => {
    const next = { ArrowLeft: last - 1, ArrowRight: last + 1, Home: 0, End: size - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    show(next);
  });
}

function bindPairs(element, groups, study, initialIndex) {
  const chart = element.querySelector('[data-ev-slopes]');
  const values = observedPairs(groups);
  if (!chart || !values.length) return;
  const tracks = [...chart.querySelectorAll('.ev-pair')];
  const detail = element.querySelector('[data-ev-pair-detail]');
  const caption = element.querySelector('[data-ev-pair-caption]');
  let current = -1;
  const select = (index) => {
    if (index === current) return;
    current = index;
    tracks.forEach((line, i) => line.classList.toggle('is-inspected', i === index));
    chart.setAttribute('aria-valuenow', String(index + 1));
    if (index < 0) {
      chart.setAttribute('aria-valuetext', 'No participant selected');
      caption.textContent = 'Inspect a participant line';
      detail.hidden = true;
      return;
    }
    const pair = values[index];
    caption.textContent = 'Paired participant ' + (index + 1) + ' of ' + values.length;
    detail.innerHTML = visits
      .map((visit) => {
        const g = pair[visit];
        const count = fmtCount.format(g.cells) + ' / ' + fmtCount.format(g.totalCells);
        return (
          '<div class="ev-' +
          visit +
          '"><span>' +
          safe(visitLabel(study, visit)) +
          '</span><strong>' +
          pct(g.fraction, 2) +
          '</strong><small>' +
          count +
          ' cells</small></div>'
        );
      })
      .join('');
    detail.hidden = false;
    chart.setAttribute(
      'aria-valuetext',
      caption.textContent +
        '. ' +
        visits
          .map(
            (visit) =>
              visitLabel(study, visit) +
              ' ' +
              pct(pair[visit].fraction, 2) +
              ', ' +
              pair[visit].cells +
              ' of ' +
              pair[visit].totalCells +
              ' cells'
          )
          .join('. ')
    );
  };
  const nearest = (event) => {
    const svg = chart.querySelector('svg');
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const t = Math.max(0, Math.min(1, (p.x - 45) / 206));
    let best = 0,
      distance = Infinity;
    tracks.forEach((line, i) => {
      const y = Number(line.dataset.yPre) * (1 - t) + Number(line.dataset.yPost) * t;
      const gap = Math.abs(p.y - y);
      if (gap < distance) {
        best = i;
        distance = gap;
      }
    });
    if (distance <= 13 && p.x >= 32 && p.x <= 264) select(best);
  };
  chart.addEventListener('pointermove', (event) => {
    if (event.pointerType !== 'touch') nearest(event);
  });
  chart.addEventListener('pointerdown', nearest);
  const step = (delta) =>
    select(
      current < 0 ? (delta > 0 ? 0 : values.length - 1) : (current + delta + values.length) % values.length
    );
  element
    .querySelectorAll('[data-ev-step]')
    .forEach((button) => button.addEventListener('click', () => step(Number(button.dataset.evStep))));
  chart.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown', 'Home', 'End', 'Escape'].includes(event.key))
      return;
    event.preventDefault();
    if (event.key === 'Escape') select(-1);
    else if (event.key === 'Home') select(0);
    else if (event.key === 'End') select(values.length - 1);
    else step(event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 1);
  });
  if (initialIndex >= 0 && initialIndex < values.length) select(initialIndex);
}
export function selectEvidenceGene(element, gene) {
  return memberViews.get(element)?.(gene) ?? false;
}
function bindMembers(element, lastGene, hadFocus, onChange) {
  const rows = [...element.querySelectorAll('[data-ev-member]')];
  if (!rows.length) return;
  const output = element.querySelector('[data-ev-member-read]');
  const list = element.querySelector('.ev-member-list');
  const reveal = (row) => {
    const edge = list.getBoundingClientRect();
    const bounds = row.getBoundingClientRect();
    if (bounds.top < edge.top) list.scrollTop -= edge.top - bounds.top;
    else if (bounds.bottom > edge.bottom) list.scrollTop += bounds.bottom - edge.bottom;
  };
  const choose = (index, notify = false, bringIntoView = false) => {
    rows.forEach((button, i) => {
      button.tabIndex = i === index ? 0 : -1;
      button.setAttribute('aria-pressed', String(i === index));
    });
    output.textContent = rows[index].getAttribute('aria-label');
    const row = rows[index];
    element.querySelector('[data-ev-member-exact]').textContent = row.dataset.evExact;
    if (bringIntoView) reveal(row);
    if (notify) onChange({ member: row.firstElementChild.textContent });
  };
  memberViews.set(element, (gene) => {
    const index = rows.findIndex((row) => row.firstElementChild.textContent === gene);
    if (index < 0) return false;
    choose(index, false, true);
    return true;
  });
  rows.forEach((button, i) => {
    button.addEventListener('click', () => choose(i, true));
    button.addEventListener('keydown', (event) => {
      const next = {
        ArrowUp: Math.max(0, i - 1),
        ArrowDown: Math.min(rows.length - 1, i + 1),
        Home: 0,
        End: rows.length - 1
      }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      choose(next, true, true);
      rows[next].focus({ preventScroll: true });
    });
  });
  const selected = Math.max(
    0,
    rows.findIndex((row) => row.firstElementChild.textContent === lastGene)
  );
  choose(selected);
  if (hadFocus) rows[selected].focus({ preventScroll: true });
}
export function renderEvidence(element, state, onChange) {
  const { study, layer, arm, pop, visit, gene, pathway: pathwayId } = state;
  const key = JSON.stringify([layer, arm, pop, visit, gene, pathwayId]);
  const previous = cache.get(element);
  if (previous?.study === study && previous.key === key) return;
  if (!study?.groups) return;
  const active = element.contains(document.activeElement) ? document.activeElement?.dataset.evSelect : null;
  const selectedMember = element.querySelector('[data-ev-member][aria-pressed="true"]');
  const lastGene = selectedMember?.firstElementChild.textContent;
  const hadFocus = !!selectedMember && selectedMember === document.activeElement;
  const listScroll = element.querySelector('.ev-member-list')?.scrollTop || 0;
  const lastPair =
    Number(element.querySelector('[data-ev-slopes]')?.getAttribute('aria-valuenow') || 0) - 1;
  const samePopulation = previous?.study === study && previous.pop === pop && previous.arm === arm;
  const keepMember =
    previous?.study === study &&
    previous.pathwayId === pathwayId &&
    previous.pop === pop &&
    previous.arm === arm;
  cache.set(element, { study, key, pathwayId, pop, arm });
  element.dataset.evVisit = visit;
  const groups = visits.map((v) =>
    study.groups.find((g) => g.arm === arm && g.visit === v && g.population === pop)
  );
  const kind = layer === 'pathway' ? 'pathway' : 'gene';
  const options = study.features.filter((f) => f.kind === kind);
  const feature = options.find((f) => f.id === (kind === 'gene' ? gene : pathwayId)) || options[0];
  let content = '';
  if (layer === 'expression') content = expression(state, groups, feature, options);
  else if (layer === 'pathway') content = pathway(state, groups, feature, options);
  else content = fractions(state, groups);
  memberViews.delete(element);
  element.classList.add('ev-panel');
  element.innerHTML = content;
  element.querySelectorAll('[data-ev-select]').forEach((select) => {
    select.addEventListener('change', () => onChange({ [select.dataset.evSelect]: select.value }));
    if (select.dataset.evSelect === active) select.focus({ preventScroll: true });
  });
  if (layer === 'composition') bindPairs(element, groups, study, samePopulation ? lastPair : -1);
  if (layer === 'expression' && feature) bindHistogram(element, groups, feature, study);
  if (layer === 'pathway') {
    bindMembers(element, state.member || (keepMember ? lastGene : null), keepMember && hadFocus, onChange);
    const list = element.querySelector('.ev-member-list');
    if (keepMember && list) list.scrollTop = listScroll;
  }
}
