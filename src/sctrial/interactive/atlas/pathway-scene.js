// The two marks at each gene's angle are observed means for the two visits.
const svgNS = 'http://www.w3.org/2000/svg';
const tau = Math.PI * 2;
const visits = ['pre', 'post'];

const domainCache = new WeakMap();

function element(name, className, parent, text) {
  const el = document.createElement(name);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  parent.append(el);
  return el;
}

function shape(name, parent, attributes) {
  const el = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
  parent.append(el);
  return el;
}

function limitFor(study, feature) {
  if (!domainCache.has(study)) domainCache.set(study, new Map());
  const saved = domainCache.get(study);
  if (!saved.has(feature.id)) {
    let limit = 0;
    for (const group of study.groups) {
      for (const value of group.signals?.[feature.id]?.memberMeans || []) {
        if (Number.isFinite(value)) limit = Math.max(value, limit);
      }
    }
    const step = limit > 2 ? 0.5 : limit > 1 ? 0.25 : 0.1;
    saved.set(feature.id, Math.max(step, Math.ceil(limit / step) * step));
  }
  return saved.get(feature.id);
}

function compactNumber(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value === 0) return '0';
  return Number(value.toPrecision(4)).toString();
}

function position(cx, cy, radius, angle) {
  return { x: cx + Math.sin(angle) * radius, y: cy - Math.cos(angle) * radius };
}

function stroke(parent, a, b, className) {
  return shape('line', parent, { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: className });
}

function circle(parent, p, r, className) {
  return shape('circle', parent, { cx: p.x, cy: p.y, r, class: className });
}

export function createPathwayScene(container, { onInspect } = {}) {
  const root = element('section', 'pw-scene', container);
  root.setAttribute('aria-label', 'Pathway member expression');
  const svg = shape('svg', root, { class: 'pw-dial', 'aria-hidden': 'true' });
  const header = element('div', 'pw-heading', root);
  element('span', 'pw-kicker', header, 'Pathway member expression');
  const heading = element('div', 'pw-title', header);
  const count = element('span', 'pw-count', header);
  const legend = element('div', 'pw-legend', root);
  const preLabel = element('span', 'pw-pre-key', legend);
  const postLabel = element('span', 'pw-post-key', legend);
  const center = element('div', 'pw-center', root);
  const gene = element('strong', 'pw-gene', center);
  const measurement = element('div', 'pw-values', center);
  const preValue = element('span', 'pw-pre-value', measurement);
  const postValue = element('span', 'pw-post-value', measurement);
  const unit = element('span', 'pw-unit', center);
  const empty = element('div', 'pw-empty', root);
  empty.hidden = true;
  const ruler = element('div', 'pw-scale', root);
  const toolbar = element('div', 'pw-picker', root);
  const previous = element('button', 'pw-prev', toolbar, '‹');
  previous.type = 'button';
  previous.setAttribute('aria-label', 'Previous pathway member');
  const menu = element('select', 'pw-menu', toolbar);
  menu.setAttribute('aria-label', 'Inspect a pathway member');
  const next = element('button', 'pw-next', toolbar, '›');
  next.type = 'button';
  next.setAttribute('aria-label', 'Next pathway member');
  const instruction = element('span', 'pw-help', toolbar, 'Explore members');
  instruction.setAttribute('aria-hidden', 'true');

  let state,
    model,
    selected,
    geometry,
    spokes = [],
    glyph,
    active,
    idKey = '';
  let pointerId = null;
  const abort = new AbortController();
  const options = { signal: abort.signal };
  let timer, lastProgress, lastCompare;
  const resize = new ResizeObserver(() => render());
  resize.observe(root);

  function sector(parent, cx, cy, r0, r1, angle, halfWidth, name) {
    if (r1 <= r0) return;
    const corners = [
      [r0, -1],
      [r1, -1],
      [r1, 1],
      [r0, 1]
    ].map(([r, sign]) => position(cx, cy, r, angle + sign * halfWidth));
    shape('polygon', parent, {
      points: corners.map((p) => p.x.toFixed(2) + ',' + p.y.toFixed(2)).join(' '),
      class: name
    });
  }

  function select(idx, notify) {
    if (!model?.entries.length) return;
    const i = (idx + model.entries.length) % model.entries.length;
    if (selected === i && active === i) return;
    selected = i;
    const entry = model.entries[i];
    menu.value = entry.gene;
    gene.textContent = entry.gene;
    gene.style.setProperty(
      '--pw-gene-size',
      entry.gene.length > 9 ? '16px' : entry.gene.length > 6 ? '20px' : '26px'
    );
    preValue.textContent = compactNumber(entry.pre);
    postValue.textContent = compactNumber(entry.post);
    for (const [j, value] of [entry.pre, entry.post].entries()) {
      const el = j ? postValue : preValue;
      const label = j ? postLabel.textContent : preLabel.textContent;
      el.title = label + ' ' + (Number.isFinite(value) ? String(value) + ' ' + model.unit : 'unavailable');
      el.setAttribute('aria-label', el.title);
    }
    center.setAttribute('aria-label', [entry.gene, preValue.title, postValue.title].join(', '));
    if (spokes[active]) spokes[active].classList.remove('pw-selected');
    if (spokes[i]) spokes[i].classList.add('pw-selected');
    active = i;
    if (glyph && geometry) {
      const { cx, cy, inner, outer } = geometry;
      const angle = (tau * i) / model.entries.length;
      glyph.setAttribute(
        'transform',
        'translate(' + cx + ' ' + cy + ') rotate(' + (angle * 180) / Math.PI + ')'
      );
      glyph.replaceChildren();
      sector(glyph, 0, 0, inner, outer, 0, (tau / model.entries.length) * 0.45, 'pw-cursor-band');
      const start = { x: 0, y: -inner + 2 },
        end = { x: 0, y: -outer - 10 };
      stroke(glyph, start, end, 'pw-cursor-ray');
      shape('path', glyph, {
        d: 'M-4 ' + (-outer - 11) + ' L0 ' + (-outer - 4) + ' L4 ' + (-outer - 11) + ' Z',
        class: 'pw-cursor-cap'
      });
      for (const [j, visit] of visits.entries()) {
        if (!Number.isFinite(entry[visit])) continue;
        const radius = inner + ((outer - inner) * entry[visit]) / model.max;
        const p = position(0, 0, radius, geometry.separation * (j ? 1 : -1));
        circle(glyph, p, 3.1, 'pw-cursor-dot pw-cursor-' + visit);
      }
    }
    if (notify && onInspect)
      onInspect({ gene: entry.gene, pre: entry.pre, post: entry.post, pathway: model.feature.id });
  }

  function setMix() {
    if (!state) return;
    const progress = Math.max(
      0,
      Math.min(1, Number.isFinite(state.progress) ? state.progress : state.visit === 'post' ? 1 : 0)
    );
    if (progress === lastProgress && state.compare === lastCompare) return;
    root.style.setProperty(
      '--pw-pre-alpha',
      state.compare ? 0.98 : (0.4 + 0.58 * (1 - progress)).toFixed(3)
    );
    root.style.setProperty('--pw-post-alpha', state.compare ? 0.98 : (0.4 + 0.58 * progress).toFixed(3));
    root.dataset.pwVisit = state.visit;
    lastProgress = progress;
    lastCompare = state.compare;
  }

  function render() {
    if (!model || !root.clientWidth || !root.clientHeight) return;
    const w = root.clientWidth,
      h = root.clientHeight;
    const wide = w >= 610;
    root.classList.toggle('pw-wide', wide);
    // Leave the corners available for the surrounding app's view controls.
    const cx = wide ? w * 0.52 : w / 2;
    const top = wide ? 39 : 78,
      bottom = wide ? 43 : 48;
    const cy = (h + top - bottom) / 2;
    const outer = Math.max(
      18,
      Math.min(w * (wide ? 0.34 : 0.393), (h - top - bottom) / 2 - (wide ? 14 : 8))
    );
    const inner = Math.max(33, Math.min(wide ? 83 : 63, outer * 0.47));
    const separation = (tau / model.entries.length) * 0.16;
    geometry = { cx, cy, inner, outer, separation };
    center.style.left = cx + 'px';
    center.style.top = cy + 'px';
    center.style.width = Math.max(57, inner * 1.85) + 'px';
    center.style.setProperty('--pw-inner', inner + 'px');
    center.classList.toggle('pw-small', inner < 57);
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    svg.replaceChildren();
    const scene = shape('g', svg, {});
    const guides = shape('g', scene, { class: 'pw-guides' });
    for (let j = 0; j <= 4; j++) {
      const radius = inner + ((outer - inner) * j) / 4;
      circle(guides, { x: cx, y: cy }, radius, j === 0 ? 'pw-zero' : j === 4 ? 'pw-bound' : 'pw-guide');
    }
    const activeGroup = shape('g', scene, { class: 'pw-spokes' });
    const labelGroup = shape('g', scene, { class: 'pw-labels' });
    const entries = model.entries;
    const n = entries.length;
    const labelCount = wide ? 12 : 8;
    const outside = new Set(
      Array.from({ length: labelCount }, (_, j) => Math.round(((j + 0.5) * n) / labelCount) % n)
    );
    // All gene positions stay fixed in alphabetical order for both visits.
    spokes = [];
    for (let i = 0; i < n; i++) {
      const a = (tau * i) / n;
      const major = outside.has(i);
      stroke(
        guides,
        position(cx, cy, outer + 2, a),
        position(cx, cy, outer + (major ? 8 : 4), a),
        major ? 'pw-major-tick' : 'pw-tick'
      );
      const item = shape('g', activeGroup, { 'data-pw-member': entries[i].gene });
      spokes.push(item);
      for (const [j, visit] of visits.entries()) {
        const value = entries[i][visit];
        if (!Number.isFinite(value)) continue;
        const angle = a + separation * (j ? 1 : -1);
        const end = position(cx, cy, inner + ((outer - inner) * value) / model.max, angle);
        const start = position(cx, cy, inner, angle);
        stroke(item, start, end, 'pw-spoke pw-' + visit);
        sector(
          item,
          cx,
          cy,
          inner,
          inner + ((outer - inner) * value) / model.max,
          angle,
          separation * 0.8,
          'pw-fill pw-' + visit
        );
        const size = wide ? 1.75 : 1.35;
        if (visit === 'pre') {
          shape('rect', item, {
            x: end.x - size,
            y: end.y - size,
            width: 2 * size,
            height: 2 * size,
            class: 'pw-mark pw-pre'
          });
        } else {
          circle(item, end, size, 'pw-mark pw-post');
        }
      }
      if (major) {
        const { x, y } = position(cx, cy, outer + 11, a);
        const direction = Math.sin(a);
        const anchor = Math.abs(direction) < 0.22 ? 'middle' : direction > 0 ? 'start' : 'end';
        const label = shape('text', labelGroup, {
          x,
          y: y + 3,
          'text-anchor': anchor,
          class: 'pw-member-label'
        });
        label.textContent = entries[i].gene;
      }
    }
    glyph = shape('g', scene, { class: 'pw-cursor' });
    active = undefined;
    select(selected ?? 0, false);
    ruler.textContent = 'Radial scale  0 – ' + compactNumber(model.max) + ' ' + model.unit;
    ruler.title = 'Linear scale shared by all populations, treatment arms and visits for this pathway.';
  }

  function setState(nextState) {
    if (!nextState?.study || !nextState.pathway) return;
    const previousStudy = state?.study;
    state = nextState;
    const key = [state.arm, state.pop, state.pathway].join('|');
    if (key === idKey && previousStudy === state.study) {
      setMix();
      return;
    }
    idKey = key;
    const previousGene = model?.entries[selected]?.gene;
    const feature = state.study.features.find((f) => f.id === state.pathway && f.kind === 'pathway');
    if (!feature?.members?.length) {
      model = null;
      svg.replaceChildren();
      header.hidden = center.hidden = toolbar.hidden = legend.hidden = ruler.hidden = true;
      empty.textContent = 'No pathway members are available for this study.';
      empty.hidden = false;
      return;
    }
    const values = visits.map((visit) => {
      const g = state.study.groups.find(
        (g) => g.arm === state.arm && g.population === state.pop && g.visit === visit
      );
      return g?.cells ? g.signals?.[feature.id]?.memberMeans : null;
    });
    const entries = feature.members.map((gene, index) => ({
      gene,
      pre: values[0]?.[index],
      post: values[1]?.[index]
    }));
    entries.sort((a, b) => (a.gene < b.gene ? -1 : a.gene > b.gene ? 1 : 0));
    model = {
      feature,
      entries,
      max: limitFor(state.study, feature),
      unit: feature.memberUnit || feature.unit?.replace(/^mean member /i, '') || 'expression'
    };
    unit.textContent = model.unit;
    heading.textContent = feature.label;
    count.textContent = entries.length + ' measured genes';
    for (const [i, el] of [preLabel, postLabel].entries())
      el.textContent = state.study.visits.find((v) => v.id === visits[i])?.label || visits[i];
    const hasValues = entries.some((e) => Number.isFinite(e.pre) || Number.isFinite(e.post));
    header.hidden = legend.hidden = false;
    empty.hidden = hasValues;
    if (!hasValues) {
      svg.replaceChildren();
      svg.style.display = 'none';
      center.hidden = toolbar.hidden = ruler.hidden = true;
      empty.textContent = 'No member expression is available for this population at either visit.';
      return;
    }
    svg.style.display = '';
    center.hidden = toolbar.hidden = ruler.hidden = false;
    const preferred = [state.member, previousGene];
    selected = preferred.map((g) => entries.findIndex((e) => e.gene === g)).find((i) => i >= 0) ?? 0;
    menu.replaceChildren(...entries.map((entry) => new Option(entry.gene, entry.gene)));
    unit.title = 'Mean expression across all captured cells of the selected population';
    setMix();
    render();
    if (!state.reduced) {
      root.classList.remove('pw-reveal');
      void root.offsetWidth;
      root.classList.add('pw-reveal');
      clearTimeout(timer);
      timer = setTimeout(() => root.classList.remove('pw-reveal'), 480);
    }
  }

  function fromPointer(event) {
    if (!model || !geometry) return false;
    const rect = svg.getBoundingClientRect();
    const dx = event.clientX - rect.left - geometry.cx,
      dy = event.clientY - rect.top - geometry.cy;
    const radius = Math.hypot(dx, dy);
    if (radius < geometry.inner - 5 || radius > geometry.outer + 24) return false;
    const phase = (Math.atan2(dx, -dy) + tau) % tau;
    select(Math.round((phase * model.entries.length) / tau) % model.entries.length, true);
    return true;
  }
  svg.addEventListener(
    'pointerdown',
    (e) => {
      if (fromPointer(e)) {
        pointerId = e.pointerId;
        svg.setPointerCapture(pointerId);
      }
    },
    options
  );
  svg.addEventListener(
    'pointermove',
    (e) => {
      if (e.pointerType === 'mouse' || pointerId === e.pointerId) fromPointer(e);
    },
    options
  );
  svg.addEventListener(
    'lostpointercapture',
    () => {
      pointerId = null;
    },
    options
  );
  menu.addEventListener(
    'change',
    () =>
      select(
        model.entries.findIndex((e) => e.gene === menu.value),
        true
      ),
    options
  );
  menu.addEventListener(
    'keydown',
    (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        select(selected + (e.key === 'ArrowLeft' ? -1 : 1), true);
      }
    },
    options
  );
  previous.addEventListener('click', () => select(selected - 1, true), options);
  next.addEventListener('click', () => select(selected + 1, true), options);
  function selectGene(symbol) {
    const idx = model?.entries.findIndex((e) => e.gene === symbol) ?? -1;
    if (idx < 0) return false;
    select(idx, false);
    return true;
  }
  return {
    setState,
    selectGene,
    destroy() {
      clearTimeout(timer);
      resize.disconnect();
      abort.abort();
      root.remove();
    }
  };
}
