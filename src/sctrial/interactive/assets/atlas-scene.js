(() => {
  'use strict';

  const circle = Math.PI * 2;
  const hues = [171, 273, 38, 213, 324, 90, 19];
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const number = new Intl.NumberFormat(undefined, {
    maximumSignificantDigits: 4,
  });
  const count = new Intl.NumberFormat();
  const materialCache = new Map();

  function pigment(name, index) {
    if (/\bCD8\b|cytotoxic/i.test(name)) return 171;
    if (/\bB[ -]?(?:cell|lymph)/i.test(name)) return 273;
    if (/\bCD4\b/i.test(name)) return 38;
    if (/\bNK\b|natural killer/i.test(name)) return 213;
    if (/monocyt|myeloid|macrophag/i.test(name)) return 14;
    return hues[index % hues.length];
  }

  function hash(text) {
    let h = 2166136261;
    for (const c of text) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
    return h >>> 0;
  }

  function random(seed) {
    return () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
  }

  function layer(name, hue, variant) {
    const key = JSON.stringify([name, hue, variant]);
    if (materialCache.has(key)) return materialCache.get(key);
    const texture = document.createElement('canvas');
    texture.width = texture.height = 96;
    const ctx = texture.getContext('2d');
    ctx.filter = 'saturate(.75)';
    window.SctrialCellArt.paint(ctx, {
      x: 48,
      y: 48,
      radius: 43,
      kind: name,
      hue,
      seed: hash(name),
      rotation: [0.63, 2.35, 4.53][variant],
    });
    if (materialCache.size >= 48) materialCache.delete(materialCache.keys().next().value);
    materialCache.set(key, texture);
    return texture;
  }

  const tintCache = new Map();
  function tintLayer(name, hue, variant, rgb) {
    const key = JSON.stringify([name, hue, variant, rgb]);
    if (tintCache.has(key)) return tintCache.get(key);
    // Use one optical material for quantitation so identity pigment cannot change a value's brightness.
    const base = layer(name, 166, variant);
    const texture = document.createElement('canvas');
    texture.width = base.width;
    texture.height = base.height;
    const ctx = texture.getContext('2d');
    ctx.drawImage(base, 0, 0);
    const image = ctx.getImageData(0, 0, texture.width, texture.height);
    const channels = rgb.split(',').map(Number);
    const ramps = channels.map((channel) =>
      Array.from({ length: 256 }, (_, k) => {
        const lum = Math.pow(k / 255, 0.79);
        const peak = channel + (255 - channel) * 0.64;
        return lum <= 0.53
          ? (channel * lum) / 0.53
          : channel + ((peak - channel) * (lum - 0.53)) / 0.47;
      }),
    );
    for (let i = 0; i < image.data.length; i += 4) {
      if (!image.data[i + 3]) continue;
      const lum = Math.round(
        0.23 * image.data[i] + 0.61 * image.data[i + 1] + 0.16 * image.data[i + 2],
      );
      for (let channel = 0; channel < 3; channel++) image.data[i + channel] = ramps[channel][lum];
    }
    ctx.putImageData(image, 0, 0);
    if (tintCache.size >= 48) tintCache.delete(tintCache.keys().next().value);
    tintCache.set(key, texture);
    return texture;
  }

  const percent = new Intl.NumberFormat(undefined, {
    style: 'percent',
    maximumSignificantDigits: 3,
  });
  const maxMarks = 1250;

  // A shear curves the island's centerline without changing its area. Stretch and
  // rotation also have determinant one, leaving observed fraction as the sole area input.
  function cloudPoint(model, angle, radial) {
    const radius = radial * model.outline(angle);
    const u = radius * Math.cos(angle) * model.stretch;
    const v = (radius * Math.sin(angle)) / model.stretch + model.bend * (u * u - 0.32);
    return { x: u * model.cos - v * model.sin, y: u * model.sin + v * model.cos };
  }

  function colony(name, index) {
    const seed = hash(name),
      rnd = random(seed);
    const phases = Array.from({ length: 4 }, () => rnd() * circle);
    const amplitudes = [0.18, 0.13, 0.075, 0.045];
    const normal = Math.sqrt(1 + amplitudes.reduce((sum, a) => sum + a * a, 0) / 2);
    const angle = (rnd() - 0.5) * 1.7;
    const model = {
      name,
      hue: pigment(name, index),
      rnd,
      phase: phases[0],
      cells: [],
      path: new Path2D(),
      stretch: 1.18 + rnd() * 0.33,
      bend: (rnd() > 0.5 ? 1 : -1) * (0.14 + rnd() * 0.18),
      cos: Math.cos(angle),
      sin: Math.sin(angle),
      sectors: [],
      rim: [],
      bounds: { left: 0, right: 0, top: 0, bottom: 0 },
      buckets: new Map(),
      outline: (theta) =>
        (1 +
          amplitudes.reduce(
            (sum, amplitude, i) => sum + amplitude * Math.sin((i + 2) * theta + phases[i]),
            0,
          )) /
        normal,
    };
    // Polar sectors have unequal areas in a lobed outline; sample in area coordinates.
    const area = [0],
      steps = 192;
    for (let j = 0; j < steps; j++) {
      const theta = (circle * j) / steps;
      const point = cloudPoint(model, theta, 1);
      model.rim.push(point);
      const b = model.bounds;
      b.left = Math.min(b.left, point.x);
      b.right = Math.max(b.right, point.x);
      b.top = Math.min(b.top, point.y);
      b.bottom = Math.max(b.bottom, point.y);
      if (j) model.path.lineTo(point.x, point.y);
      else model.path.moveTo(point.x, point.y);
      area.push(area[j] + model.outline(theta + circle / (steps * 2)) ** 2);
    }
    model.path.closePath();
    let j = 0;
    for (let i = 0; i <= 512; i++) {
      const target = (area[steps] * i) / 512;
      while (j < steps - 1 && area[j + 1] < target) j++;
      model.sectors.push((circle * (j + (target - area[j]) / (area[j + 1] - area[j]))) / steps);
    }
    return model;
  }

  // Progressive best-candidate sampling keeps stable cells evenly scattered without rows,
  // spiral spokes or a reset as the visit changes. Only nearby points enter each search.
  function addMarks(model, needed) {
    const { cells, buckets, rnd } = model;
    function bucket(x, y) {
      return Math.floor(x * 5) * 128 + Math.floor(y * 5);
    }
    while (cells.length < needed) {
      let best,
        clearance = -1;
      const stratum = Math.floor(rnd() * 12);
      for (let j = 0; j < 7; j++) {
        const sector = rnd() * 512,
          k = Math.floor(sector);
        const theta = model.sectors[k] + (model.sectors[k + 1] - model.sectors[k]) * (sector - k);
        // A diffuse rim leaves the population contour legible at low cell fractions.
        const radial = Math.sqrt((stratum + rnd()) / 12);
        const point = cloudPoint(model, theta, radial * (0.66 + 0.34 * radial ** 5));
        const key = bucket(point.x, point.y);
        let d = 0.18 ** 2;
        for (const offset of [-129, -128, -127, -1, 0, 1, 127, 128, 129]) {
          for (const prior of buckets.get(key + offset) || []) {
            const dx = prior.x - point.x,
              dy = prior.y - point.y;
            d = Math.min(d, dx * dx + dy * dy);
          }
        }
        if (d > clearance) {
          best = point;
          clearance = d;
        }
      }
      const cell = {
        ...best,
        size: 0.74 + 0.43 * rnd(),
        phase: circle * rnd(),
        rotation: circle * rnd(),
        variant: Math.floor(rnd() * 3),
      };
      const key = bucket(cell.x, cell.y);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(cell);
      cells.push(cell);
    }
  }

  function molecularField(group, cellMode, color, fallbackDomain) {
    const bins = group.cellBins;
    // CSV rows carry participant summaries. Individual cell colors there retain
    // the population identity. Only a measured single-cell histogram drives signal colors.
    if (!cellMode) return { key: 'type', bands: [{ end: 1, shade: null }], images: new Map() };
    const missing = color(null, fallbackDomain);
    if (!bins || !bins.counts || !bins.domain || !group.count)
      return { key: 'missing', bands: [{ end: 1, shade: missing }], images: new Map() };
    const { counts, domain } = bins;
    const [lo, hi] = domain;
    const size = counts.length;
    const boundary = (i) => lo * (1 - i / size) + hi * (i / size);
    // The histogram uses right-open bins, except the final bin. Compare to the
    // same floating-point edges as the importer, including exact zero on an edge.
    let zeroBin = 0;
    if (lo !== hi) while (zeroBin + 1 < size && boundary(zeroBin + 1) <= 0) zeroBin++;
    let total = 0;
    const bands = [];
    function add(n, shade) {
      if (n <= 0) return;
      total += n;
      bands.push({ end: total / group.count, shade });
    }
    for (let i = 0; i < size; i++) {
      if (!counts[i]) continue;
      const zero = i === zeroBin ? Math.min(counts[i], bins.zero || 0) : 0;
      const t = (i + 0.5) / size;
      // A bin crossing the origin cannot identify the sign of every value inside it.
      const crossesZero = boundary(i) < 0 && boundary(i + 1) > 0;
      const midpoint = lo * (1 - t) + hi * t;
      add(zero, color(0, domain));
      const shade = crossesZero
        ? color(boundary(i), domain)
            .split(',')
            .map((v, j) =>
              Math.round((Number(v) + Number(color(boundary(i + 1), domain).split(',')[j])) / 2),
            )
            .join(',')
        : color(midpoint, domain);
      add(counts[i] - zero, shade);
    }
    add(group.count - total, missing);
    const key = JSON.stringify(bands);
    return { key, bands, images: new Map() };
  }

  function prepareShades(field, n, phase) {
    if (field.shades?.length === n) return field.shades;
    if (field.bands.length === 1) return (field.shades = Array(n).fill(field.bands[0].shade));
    // Stratified quantiles preserve the observed frequencies. Scrambling their order
    // avoids inventing a spatial expression gradient in a schematic colony.
    const order = Array.from({ length: n }, (_, i) => i);
    // An independent shuffle matters: using a second spiral sequence can align
    // color with angle and create a wedge of high values without a spatial measurement.
    const shuffle = random(Math.floor((phase / circle) * 0xffffffff) ^ 0x5c7a390b);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(shuffle() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const shades = Array(n);
    let band = 0;
    order.forEach((index, j) => {
      const quantile = (j + 0.5) / n;
      while (band + 1 < field.bands.length && quantile >= field.bands[band].end) band++;
      shades[index] = field.bands[band].shade;
    });
    return (field.shades = shades);
  }

  function material(model, field, shade, variant) {
    let variants = field.images.get(shade);
    if (!variants) field.images.set(shade, (variants = []));
    return (variants[variant] ||= shade
      ? tintLayer(model.name, model.hue, variant, shade)
      : layer(model.name, model.hue, variant));
  }

  const style = `
    .sc-atlas-world{position:absolute;inset:0;overflow:hidden;isolation:isolate;background:radial-gradient(ellipse at 48% 47%,#101a20 0%,#0b1217 57%,#0a1014 100%);color:#e7f1f0;user-select:none}
    .sc-atlas-world::before{content:"";position:absolute;inset:0;background-image:radial-gradient(circle,rgba(152,180,195,.20) .65px,transparent .9px);background-size:26px 26px;pointer-events:none}
    .sc-atlas-world canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:grab;outline-offset:-4px}
    .sc-atlas-world.is-dragging canvas{cursor:grabbing}
    .sc-atlas-tags{position:absolute;inset:0;pointer-events:none}
    .sc-atlas-world .sc-atlas-tag{position:absolute;left:0;top:0;display:grid;grid-template-columns:4px minmax(0,1fr);align-items:start;gap:6px;pointer-events:auto;white-space:nowrap;appearance:none;background:transparent;border:0;border-radius:3px;padding:4px 5px;color:#c1cfce;text-shadow:0 1px 5px #080c11,0 0 9px #080c11;text-align:left;cursor:pointer;font:inherit;box-sizing:border-box}
    .sc-atlas-world .sc-atlas-tag:hover,.sc-atlas-world .sc-atlas-tag:focus-visible{background:#17232af5;outline:1px solid #57646e;outline-offset:1px}
    .sc-atlas-tag i{margin-top:4px;height:8px;border-radius:1px;background:var(--pop-pigment)}
    .sc-atlas-tag>span{min-width:0;display:grid;gap:3px}
    .sc-atlas-tag strong,.sc-atlas-tag small{display:block;overflow:hidden;text-overflow:ellipsis}
    .sc-atlas-tag strong{font-family:inherit;font-weight:600;font-size:12px;line-height:1.18;letter-spacing:-.008em;color:#e3ecea}
    .sc-atlas-world .sc-atlas-tag[aria-pressed="true"] strong{color:#ffdeaa}
    .sc-atlas-tag small{font:500 10px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-.03em;color:#a5b4bc;font-variant-numeric:tabular-nums}
    .sc-atlas-tag .at-signal-mini[hidden]{display:none}
    .sc-atlas-pages{position:absolute;top:8px;right:9px;display:flex;align-items:center;background:#131c22ed;border:1px solid #34424a;border-radius:4px;font:11px/1.2 system-ui;padding:2px 4px;gap:6px}
    .sc-atlas-world .sc-atlas-tag[hidden],.sc-atlas-pages[hidden]{display:none}.sc-atlas-pages button{color:#d5e3df;border:0;background:transparent;padding:3px 8px;cursor:pointer;font:16px/1 system-ui}.sc-atlas-pages button:disabled{opacity:.35;cursor:default}
    @media(max-width:570px){.sc-atlas-world .sc-atlas-tag{padding:4px 2px;gap:4px}.sc-atlas-tag strong{font-size:11px}.sc-atlas-tag small{font-size:9px}}
  `;

  // Anchor families are shared by every visit. Fractions change footprints while the
  // population keeps its identity and neighborhood. Positions have no biological units.
  const anchors = [
    [],
    [[0.49, 0.5]],
    [
      [0.3, 0.57],
      [0.7, 0.37],
    ],
    [
      [0.27, 0.45],
      [0.73, 0.3],
      [0.64, 0.73],
    ],
    [
      [0.26, 0.36],
      [0.72, 0.25],
      [0.7, 0.71],
      [0.28, 0.74],
    ],
    [
      [0.22, 0.43],
      [0.53, 0.22],
      [0.78, 0.46],
      [0.59, 0.75],
      [0.26, 0.77],
    ],
    [
      [0.18, 0.32],
      [0.5, 0.24],
      [0.78, 0.36],
      [0.75, 0.74],
      [0.44, 0.72],
      [0.18, 0.74],
    ],
  ];

  function pack(groups, fullPage, width, height, paged, models) {
    if (!groups.length || width < 1 || height < 1) return [];
    const margin = width < 500 ? 7 : 14;
    const top = paged ? 34 : margin;
    const innerW = Math.max(1, width - 2 * margin),
      innerH = Math.max(1, height - top - margin);
    const portrait = innerH > 1.12 * innerW;
    const wide = innerW / innerH > 2;
    const minor = Math.min(innerW, innerH),
      major = Math.max(innerW, innerH);
    const n = fullPage;
    const targetUnit =
      n === 1
        ? Math.min(innerW * 0.3, innerH * 0.38)
        : Math.min(minor * (n > 3 ? 0.35 : 0.52), major * (n > 3 ? 0.234 : 0.267));
    const unit = paged
      ? Math.min(targetUnit, minor / 4.6)
      : groups.reduce((available, g) => {
          const b = models.get(g.name).bounds;
          return Math.min(available, innerW / (b.right - b.left), innerH / (b.bottom - b.top));
        }, targetUnit);
    const budget = Math.min(maxMarks, Math.max(1, Math.round((unit / 4.65) ** 2)));
    const labelWidth = Math.max(52, Math.min(164, innerW / (innerW < 500 ? 2.8 : 3.3)));
    const points = groups.map((g, i) => {
      const model = models.get(g.name);
      let [a, b] = anchors[groups.length][i];
      if (portrait) [a, b] = [b, a];
      if (wide && groups.length === 3) {
        a = [0.18, 0.52, 0.83][i];
        b = [0.58, 0.31, 0.62][i];
      }
      const r = unit * Math.sqrt(g.fraction),
        bounds = model.bounds;
      return {
        name: g.name,
        r,
        marks: Math.min(g.count, budget * g.fraction),
        labelWidth,
        x: margin + a * innerW,
        y: top + b * innerH,
        bounds,
      };
    });
    // Move only when occupied extents collide. Small populations retain their anchors and
    // the radius scale stays fixed across the full range from 0% to 100%.
    function constrain(p) {
      const b = p.bounds;
      p.x = clamp(p.x, margin - b.left * p.r, width - margin - b.right * p.r);
      p.y = clamp(p.y, top - b.top * p.r, height - margin - b.bottom * p.r);
    }
    for (let step = 0; step < 14; step++) {
      points.forEach(constrain);
      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        for (let j = i + 1; j < points.length; j++) {
          const b = points[j],
            dx = b.x - a.x,
            dy = b.y - a.y,
            distance = Math.hypot(dx, dy) || 1;
          const vx = dx / distance,
            vy = dy / distance;
          // Directional projections are tighter than bounding boxes for tilted populations.
          const reach = (point, vx, vy) =>
            Math.max(...models.get(point.name).rim.map((q) => q.x * vx + q.y * vy)) * point.r;
          const separation = reach(a, vx, vy) + reach(b, -vx, -vy) + minor * 0.025;
          const move = (separation - distance) * 0.4;
          if (move <= 0) continue;
          a.x -= move * vx;
          a.y -= move * vy;
          b.x += move * vx;
          b.y += move * vy;
        }
      }
    }
    points.forEach(constrain);
    return points;
  }

  function element(tag, cls, value) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function mount(host, initial) {
    const world = element('div', 'sc-atlas-world');
    const sheet = element('style');
    sheet.textContent = style;
    const canvas = element('canvas');
    canvas.setAttribute(
      'aria-label',
      'Schematic cell populations sized by fraction of observed cells. Tab to select a population, drag to pan, or use plus and minus to zoom.',
    );
    canvas.tabIndex = 0;
    const tags = element('div', 'sc-atlas-tags');
    const pages = element('div', 'sc-atlas-pages');
    const prev = element('button', '', '‹'),
      next = element('button', '', '›'),
      label = element('span');
    prev.type = next.type = 'button';
    prev.setAttribute('aria-label', 'Previous populations');
    next.setAttribute('aria-label', 'Next populations');
    pages.append(prev, label, next);
    world.append(sheet, canvas, tags, pages);
    host.append(world);
    const ctx = canvas.getContext('2d');
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    let opt = initial,
      groups = [],
      visible = [],
      models = new Map(),
      tagNodes = new Map(),
      hits = [];
    let w = 0,
      h = 0,
      density = 1,
      page = 0,
      capacity = 6,
      frame = 0,
      zoom = 1,
      targetZoom = 1,
      px = 0,
      py = 0;
    let time = 0,
      last = 0,
      inViewport = true,
      destroyed = false,
      dirty = true;
    const fingers = new Map();
    let focus = null,
      pinch = null,
      drag = null;
    const compactMotion = () => opt.reducedMotion ?? media.matches;

    function wake() {
      dirty = true;
      if (!destroyed && !document.hidden && inViewport && !frame)
        frame = requestAnimationFrame(draw);
    }

    function reset() {
      zoom = targetZoom = 1;
      focus = null;
      px = py = 0;
      wake();
    }
    function focusAt(point) {
      const sx = point.x - w / 2,
        sy = point.y - h / 2;
      return { sx, sy, wx: (sx - px) / zoom, wy: (sy - py) / zoom };
    }
    function zoomBy(factor, point = { x: w / 2, y: h / 2 }) {
      if (Number.isFinite(factor) && factor > 0) {
        focus = focusAt(point);
        targetZoom = clamp(targetZoom * factor, 0.65, 2.8);
        wake();
      }
    }
    function select(name) {
      if (typeof opt.onSelect === 'function') opt.onSelect(name);
    }

    function renderTags() {
      const names = new Set(visible.map((g) => g.name));
      for (const [name, button] of tagNodes)
        if (!names.has(name)) {
          button.remove();
          tagNodes.delete(name);
        }
      for (const g of visible) {
        let btn = tagNodes.get(g.name);
        if (!btn) {
          btn = element('button', 'sc-atlas-tag');
          btn.type = 'button';
          const wrap = element('span');
          wrap.append(
            element('strong', '', g.name),
            element('small'),
            element('span', 'at-signal-mini'),
          );
          btn.append(element('i'), wrap);
          btn.addEventListener('click', () => select(g.name));
          tags.append(btn);
          tagNodes.set(g.name, btn);
        }
        const m = models.get(g.name);
        const measured = opt.measurement && Number.isFinite(g.value);
        const readout = opt.measurement
          ? measured
            ? 'Mean ' + number.format(g.value) + ' · n=' + count.format(g.n)
            : 'Unmeasured · n=0'
          : percent.format(g.fraction) +
            ' · ' +
            count.format(g.count) +
            (g.count === 1 ? ' cell' : ' cells');
        btn.dataset.population = g.name;
        btn.dataset.count = String(g.count);
        btn.dataset.fraction = String(g.fraction);
        btn.dataset.n = String(opt.measurement ? g.n : g.participants);
        if (measured) btn.dataset.value = String(g.value);
        else delete btn.dataset.value;
        btn.querySelector('small').textContent = readout;
        const mini = btn.querySelector('.at-signal-mini');
        mini.hidden = !measured;
        if (measured) window.SctrialAtlasSignal?.mini(mini, g, opt.domain);
        const total = count.format(g.totalCells);
        const full = [
          g.name,
          count.format(g.count) +
            ' of ' +
            total +
            ' observed cells (' +
            percent.format(g.fraction) +
            ')',
          opt.featureLabel,
          m.cellMode && g.count
            ? count.format(g.cellBins?.measured ?? 0) +
              ' cells with measured signal. Colors summarize binned values, positions are schematic'
            : null,
          measured
            ? 'Participant mean ' +
              number.format(g.value) +
              ' across ' +
              count.format(g.n) +
              (g.n === 1 ? ' measured participant' : ' measured participants')
            : opt.measurement
              ? 'Unmeasured'
              : null,
          measured && Number.isFinite(g.q1) && Number.isFinite(g.q3)
            ? 'Middle half ' +
              number.format(g.q1) +
              ' to ' +
              number.format(g.q3) +
              ' on the shared feature scale'
            : null,
          opt.visit,
        ]
          .filter(Boolean)
          .join(' · ');
        btn.title = full;
        btn.setAttribute('aria-label', full);
        btn.setAttribute('aria-pressed', String(g.name === opt.selected || Boolean(g.selected)));
        btn.style.setProperty(
          '--pop-pigment',
          measured
            ? 'rgb(' + m.shade + ')'
            : opt.measurement
              ? '#829499'
              : 'hsl(' + m.hue + ' 62% 68%)',
        );
      }
      pages.hidden = groups.length <= capacity;
      prev.disabled = page === 0;
      next.disabled = (page + 1) * capacity >= groups.length;
      label.textContent =
        page * capacity +
        1 +
        '–' +
        Math.min(groups.length, (page + 1) * capacity) +
        ' / ' +
        groups.length;
    }
    const fields = ['x', 'y', 'r', 'marks', 'labelX', 'labelY', 'leaderX', 'leaderY'];
    function position(model, now) {
      const dest = model.target;
      if (!model.from || compactMotion()) return dest;
      const t = clamp((now - model.layoutStart) / 480, 0, 1);
      if (t === 1) return dest;
      const eased = t * t * (3 - 2 * t);
      const blended = {};
      for (const key of fields)
        blended[key] = model.from[key] + (dest[key] - model.from[key]) * eased;
      return blended;
    }
    function layout() {
      const now = performance.now();
      const fullPage = Math.min(groups.length, capacity);
      const packed = pack(visible, fullPage, w, h, groups.length > capacity, models);
      const placed = [];
      for (const dest of packed) {
        const model = models.get(dest.name),
          btn = tagNodes.get(dest.name);
        btn.style.maxWidth = dest.labelWidth + 'px';
        btn.dataset.markerCount = String(dest.marks);
        btn.dataset.areaRadius = String(dest.r);
        btn.hidden = false;
        model.tagWidth = btn.offsetWidth;
        model.tagHeight = btn.offsetHeight;
      }
      for (const dest of packed) {
        const model = models.get(dest.name),
          tw = model.tagWidth,
          th = model.tagHeight;
        // Attach labels to the rim, favoring space outside the colony and keeping
        // neighboring annotations and plotted populations readable.
        const origin = Math.atan2((dest.y - h * 0.5) / h, (dest.x - w * 0.5) / w);
        const lowY = groups.length > capacity ? 31 : 3;
        const overlap = (a, b) =>
          Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l)) *
          Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));
        let preferred,
          cost = Infinity;
        for (const shift of [0, 0.48, -0.48, 0.96, -0.96, 1.45, -1.45, 2.25, -2.25, Math.PI]) {
          const angle = origin + shift,
            vx = Math.cos(angle),
            vy = Math.sin(angle);
          const edge = model.rim.reduce((best, q) =>
            q.x * vx + q.y * vy > best.x * vx + best.y * vy ? q : best,
          );
          const leadX = dest.x + edge.x * dest.r,
            leadY = dest.y + edge.y * dest.r;
          const gap = w < 560 ? 8 : 16;
          const tx = leadX + vx * (tw / 2 + gap),
            ty = leadY + vy * (th / 2 + gap);
          const cx = clamp(tx, tw / 2 + 3, w - tw / 2 - 3),
            cy = clamp(ty, lowY + th / 2, h - th / 2 - 3);
          const box = {
            l: cx - tw / 2 - 3,
            r: cx + tw / 2 + 3,
            t: cy - th / 2 - 2,
            b: cy + th / 2 + 2,
          };
          let score = Math.abs(shift) * 2 + Math.hypot(cx - tx, cy - ty) * 1.2;
          for (const other of packed) {
            const b = other.bounds;
            const area = {
              l: other.x + b.left * other.r,
              r: other.x + b.right * other.r,
              t: other.y + b.top * other.r,
              b: other.y + b.bottom * other.r,
            };
            score += overlap(box, area) * 0.7;
          }
          for (const other of placed) score += overlap(box, other) * 12;
          if (score < cost) {
            cost = score;
            preferred = { labelX: cx, labelY: cy, leaderX: leadX, leaderY: leadY, box };
          }
        }
        Object.assign(dest, preferred);
        placed.push(preferred.box);
        if (!model.target || fields.some((k) => dest[k] !== model.target[k])) {
          model.from = model.target ? position(model, now) : null;
          model.target = dest;
          model.layoutStart = now;
        }
      }
    }
    function showPage(i) {
      page = clamp(i, 0, Math.max(0, Math.ceil(groups.length / capacity) - 1));
      visible = groups.slice(page * capacity, (page + 1) * capacity);
      const nextModels = new Map();
      const color = opt.signalColor || window.SctrialAtlasSignal?.color;
      const cellMode =
        opt.measurement && (Boolean(opt.cellDomain) || groups.some((g) => g.cellBins));
      visible.forEach((g, i) => {
        const model = models.get(g.name) || colony(g.name, page * capacity + i);
        model.shade =
          opt.measurement && Number.isFinite(g.value) ? color(g.value, opt.domain) : null;
        model.cellMode = cellMode;
        const field = molecularField(g, cellMode, color, opt.cellDomain);
        if (model.field?.key !== field.key) {
          model.priorField = model.field;
          model.changeStart = performance.now();
          model.field = field;
        }
        nextModels.set(g.name, model);
      });
      models = nextModels;
      renderTags();
      layout();
      wake();
    }
    prev.onclick = () => {
      reset();
      showPage(page - 1);
    };
    next.onclick = () => {
      reset();
      showPage(page + 1);
    };

    function update(options) {
      const lastSelected = opt && opt.selected;
      opt = options;
      const supplied = options.groups || [];
      const typed = supplied.filter((g) => !g.isPooled);
      const eligible = typed.length ? typed : supplied;
      const total = eligible.reduce((sum, g) => sum + g.count, 0);
      groups = eligible.map((g) => ({
        ...g,
        totalCells: g.totalCells ?? total,
        fraction: g.count > 0 ? clamp(g.fraction ?? g.count / total, 0, 1) : 0,
      }));
      const selected = groups.findIndex((g) => g.name === options.selected);
      showPage(
        selected >= 0 && (options.selected !== lastSelected || !visible.length)
          ? Math.floor(selected / capacity)
          : page,
      );
    }
    function resize() {
      w = world.clientWidth;
      h = world.clientHeight;
      density = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(w * density));
      canvas.height = Math.max(1, Math.round(h * density));
      const old = capacity;
      capacity = h > w * 0.88 && (w < 345 || h < 420) ? 4 : 6;
      if (w < 450 && h < 260) capacity = 2;
      if (w < 310 && w < h * 0.64) capacity = 2;
      if (h < 176 && w >= 450) capacity = Math.min(capacity, 3);
      if (w < 280 || h < 135) capacity = 1;
      if (old !== capacity) {
        const idx = groups.findIndex((g) => g.name === opt.selected);
        showPage(idx < 0 ? 0 : Math.floor(idx / capacity));
      } else {
        layout();
        wake();
      }
    }

    function draw(stamp) {
      frame = 0;
      if (destroyed || !inViewport || document.hidden || w < 1 || h < 1) return;
      const reduce = compactMotion(),
        elapsed = last ? clamp((stamp - last) / 1000, 0, 0.075) : 0;
      const animatingZoom = Math.abs(targetZoom - zoom) > 0.001;
      if (!dirty && !reduce && stamp - last < 32 && !animatingZoom) {
        frame = requestAnimationFrame(draw);
        return;
      }
      time += reduce ? 0 : elapsed;
      last = stamp;
      dirty = false;
      zoom =
        reduce || !animatingZoom
          ? targetZoom
          : zoom + (targetZoom - zoom) * (1 - Math.exp(-13 * Math.max(elapsed, 0.016)));
      if (focus) {
        px = focus.sx - focus.wx * zoom;
        py = focus.sy - focus.wy * zoom;
        if (!pinch && zoom === targetZoom) focus = null;
      }
      px = clamp(px, -w * zoom * 0.43, w * zoom * 0.43);
      py = clamp(py, -h * zoom * 0.43, h * zoom * 0.43);
      ctx.setTransform(density, 0, 0, density, 0, 0);
      ctx.clearRect(0, 0, w, h);
      hits = [];
      visible.forEach((g) => {
        const model = models.get(g.name);
        if (!model.target) return;
        const p = position(model, stamp);
        const x = (p.x - w / 2) * zoom + w / 2 + px;
        const y = (p.y - h / 2) * zoom + h / 2 + py;
        const r = p.r * zoom;
        const chosen = g.name === opt.selected || g.selected;
        if (r > 0) hits.push({ x, y, r, model, name: g.name });
        const btn = tagNodes.get(g.name);
        const bx = clamp(
          (p.labelX - w / 2) * zoom + w / 2 + px,
          model.tagWidth / 2 + 3,
          w - model.tagWidth / 2 - 3,
        );
        const by = clamp(
          (p.labelY - h / 2) * zoom + h / 2 + py,
          model.tagHeight / 2 + 2,
          h - model.tagHeight / 2 - 2,
        );
        btn.style.transform =
          'translate(' + Math.round(bx) + 'px,' + Math.round(by) + 'px) translate(-50%,-50%)';
        const bound = model.bounds;
        btn.hidden =
          x + r * bound.right < 0 ||
          x + r * bound.left > w ||
          y + r * bound.bottom < 0 ||
          y + r * bound.top > h;
        if (btn.hidden || !(r > 0) || !(p.marks > 0)) return;
        const leadX = (p.leaderX - w / 2) * zoom + w / 2 + px,
          leadY = (p.leaderY - h / 2) * zoom + h / 2 + py;
        const endX = clamp(leadX, bx - model.tagWidth / 2, bx + model.tagWidth / 2);
        const endY = clamp(leadY, by - model.tagHeight / 2, by + model.tagHeight / 2);
        if (Math.hypot(leadX - endX, leadY - endY) > 5 && w > 450 && h > 180) {
          ctx.beginPath();
          ctx.moveTo(leadX, leadY);
          ctx.lineTo(endX, endY);
          ctx.lineWidth = 0.75;
          ctx.strokeStyle = chosen ? 'rgba(237,194,135,.65)' : 'rgba(159,179,188,.44)';
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(leadX, leadY, 1.4, 0, circle);
          ctx.fillStyle = ctx.strokeStyle;
          ctx.fill();
        }
        const tint = 'hsl(' + model.hue + ' 66% 63%)';
        const mix =
          reduce || !model.priorField ? 1 : clamp((stamp - model.changeStart) / 340, 0, 1);
        if (mix === 1) model.priorField = null;
        const n = Math.ceil(p.marks);
        const targetCount = Math.max(n, Math.ceil(model.target.marks));
        const shades = prepareShades(model.field, targetCount, model.phase);
        const oldShades = model.priorField
          ? prepareShades(model.priorField, targetCount, model.phase)
          : null;
        ctx.save();
        ctx.translate(x, y);
        // The filled boundary carries the fraction even for subpixel rare populations.
        ctx.save();
        ctx.scale(r, r);
        ctx.fillStyle = tint;
        ctx.globalAlpha = chosen ? 0.028 : 0.016;
        ctx.fill(model.path);
        ctx.strokeStyle = tint;
        ctx.lineWidth = Math.min(0.11, 0.8 / r);
        ctx.globalAlpha = chosen ? 0.19 : 0.065;
        ctx.stroke(model.path);
        ctx.restore();
        ctx.save();
        // Clip the cell and its edge lighting to the same measured-area footprint.
        ctx.scale(r, r);
        ctx.clip(model.path);
        ctx.scale(1 / r, 1 / r);
        addMarks(model, n);
        const spacing = r / Math.sqrt(Math.max(1, p.marks));
        const markR = Math.min(5.3 * zoom, spacing * 0.68);
        // Leave space for a complete cell membrane along the edge of the colony.
        const inset = Math.max(0, 1 - (markR / r) * 1.24);
        for (let i = 0; i < n; i++) {
          const cell = model.cells[i];
          const phase = cell.phase + time * 0.32;
          const xx = cell.x * r * inset + Math.sin(phase) * markR * 0.12;
          const yy = cell.y * r * inset + Math.cos(phase * 0.9) * markR * 0.12;
          const partial = Math.min(1, p.marks - i);
          const extent = markR * cell.size * (n === 1 ? 1 : Math.sqrt(partial)) * 1.12;
          ctx.save();
          ctx.translate(xx, yy);
          ctx.rotate(cell.rotation + Math.sin(phase) * 0.08);
          if (oldShades)
            ctx.drawImage(
              material(model, model.priorField, oldShades[i], cell.variant),
              -extent,
              -extent,
              extent * 2,
              extent * 2,
            );
          ctx.globalAlpha = mix;
          ctx.drawImage(
            material(model, model.field, shades[i], cell.variant),
            -extent,
            -extent,
            extent * 2,
            extent * 2,
          );
          ctx.restore();
        }
        ctx.restore();
        ctx.restore();
      });
      if (!reduce || Math.abs(zoom - targetZoom) > 0.001) frame = requestAnimationFrame(draw);
    }

    function point(e) {
      const b = canvas.getBoundingClientRect();
      return { x: e.clientX - b.left, y: e.clientY - b.top };
    }
    function pick(p) {
      let best = null,
        distance = Infinity;
      for (const hit of hits) {
        const m = hit.model,
          x = (p.x - hit.x) / hit.r,
          y = (p.y - hit.y) / hit.r;
        const u = x * m.cos + y * m.sin,
          v = -x * m.sin + y * m.cos - m.bend * (u * u - 0.32);
        const dx = u / m.stretch,
          dy = v * m.stretch;
        const d = Math.hypot(dx, dy) / m.outline(Math.atan2(dy, dx));
        if (d < distance && d <= 1.02) {
          distance = d;
          best = hit;
        }
      }
      return best;
    }
    function distance() {
      const [a, b] = [...fingers.values()];
      return Math.max(5, Math.hypot(a.x - b.x, a.y - b.y));
    }
    canvas.onpointerdown = (e) => {
      if (e.button !== 0) return;
      const p = point(e);
      fingers.set(e.pointerId, p);
      canvas.setPointerCapture(e.pointerId);
      if (fingers.size === 1) {
        focus = null;
        targetZoom = zoom;
        drag = { start: p, px, py, moved: false };
      } else if (fingers.size === 2) {
        const [a, b] = fingers.values();
        focus = focusAt({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
        pinch = { distance: distance(), zoom };
        drag = null;
      }
    };
    canvas.onpointermove = (e) => {
      const p = point(e);
      if (!fingers.has(e.pointerId)) {
        canvas.style.cursor = pick(p) ? 'pointer' : 'grab';
        return;
      }
      fingers.set(e.pointerId, p);
      if (pinch && fingers.size === 2) {
        const [a, b] = fingers.values();
        focus.sx = (a.x + b.x - w) / 2;
        focus.sy = (a.y + b.y - h) / 2;
        targetZoom = clamp((pinch.zoom * distance()) / pinch.distance, 0.65, 2.8);
        wake();
        return;
      }
      if (!drag) return;
      const dx = p.x - drag.start.x,
        dy = p.y - drag.start.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (drag.moved) {
        world.classList.add('is-dragging');
        px = drag.px + dx;
        py = drag.py + dy;
        wake();
      }
    };
    function pointerEnd(e) {
      const tracked = fingers.delete(e.pointerId);
      if (!tracked) return;
      if (e.type === 'pointerup' && drag && !drag.moved) {
        const hit = pick(point(e));
        if (hit) select(hit.name);
      }
      drag = null;
      pinch = null;
      world.classList.remove('is-dragging');
    }
    canvas.onpointerup = canvas.onpointercancel = pointerEnd;
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        zoomBy(
          Math.exp(clamp(-e.deltaY * (e.deltaMode === 1 ? 0.024 : 0.0017), -0.5, 0.5)),
          point(e),
        );
      },
      { passive: false },
    );
    canvas.ondblclick = reset;
    canvas.onkeydown = (e) => {
      if (e.key === '+' || e.key === '=') zoomBy(1.2);
      else if (e.key === '-') zoomBy(1 / 1.2);
      else if (e.key === 'Home') reset();
      else if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        if (e.key === 'ArrowLeft') px += 28;
        if (e.key === 'ArrowRight') px -= 28;
        if (e.key === 'ArrowUp') py += 28;
        if (e.key === 'ArrowDown') py -= 28;
        wake();
      } else return;
      e.preventDefault();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(world);
    const visibilityObserver = new IntersectionObserver((entries) => {
      inViewport = entries[0].isIntersecting;
      last = 0;
      if (inViewport) wake();
    });
    visibilityObserver.observe(world);
    const onVisibility = () => {
      last = 0;
      wake();
    };
    document.addEventListener('visibilitychange', onVisibility);
    media.addEventListener('change', onVisibility);
    resize();
    update(initial);
    return {
      update,
      reset,
      zoomBy,
      destroy() {
        destroyed = true;
        cancelAnimationFrame(frame);
        resizeObserver.disconnect();
        visibilityObserver.disconnect();
        document.removeEventListener('visibilitychange', onVisibility);
        media.removeEventListener('change', onVisibility);
        world.remove();
      },
    };
  }
  window.SctrialAtlasScene = Object.freeze({ mount, pigment });
})();
