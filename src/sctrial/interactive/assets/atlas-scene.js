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
    texture.width = texture.height = 192;
    window.SctrialCellArt.paint(texture.getContext('2d'), {
      x: 96,
      y: 96,
      radius: 87,
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
    const base = layer(name, hue, variant);
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

  // Fixed illustrative geometry. A population receives one color for its aggregate measurement.
  function colony(name, index) {
    const seed = hash(name);
    const rnd = random(seed);
    const turn = rnd() * circle,
      cells = [];
    for (let i = 0; i < 20; i++) {
      const ring = i === 0 ? 0 : i < 7 ? 1 : i < 18 ? 2 : 3;
      const pos = ring === 1 ? i - 1 : ring === 2 ? i - 7 : i - 18;
      const n = [1, 6, 11, 2][ring];
      const a = turn + circle * (pos / n) + 0.19 * ring + rnd() * 0.16;
      const d = [0, 0.36, 0.71, 1.02][ring] * (0.9 + 0.19 * rnd());
      const x = Math.cos(a) * d * 1.05 + Math.sin(2 * a) * 0.045;
      const y = Math.sin(a) * d * 0.76;
      cells.push({
        x,
        y,
        r: (ring === 3 ? 0.085 : ring === 2 ? 0.147 : 0.198) * (0.83 + rnd() * 0.54),
        phase: circle * rnd(),
        rotation: rnd() * 1.9 - 0.95,
        variant: i % 3,
        stretch: 0.87 + 0.22 * rnd(),
        depth: y * 0.7 + rnd() * 0.36,
      });
    }
    return {
      name,
      hue: pigment(name, index),
      seed,
      cells: cells.sort((a, b) => a.depth - b.depth),
    };
  }

  function ratioColor(level) {
    const f = clamp(Number.isFinite(level) ? level : 0.5, 0, 1);
    return [50, 107, 114].map((v, i) => Math.round(v + ([255, 203, 119][i] - v) * f)).join(',');
  }

  const style = `
    .sc-atlas-world{position:absolute;inset:0;overflow:hidden;isolation:isolate;background:radial-gradient(ellipse at 38% 40%,#12282b 0%,#0b191e 55%,#091217 100%);color:#e7f4f1;user-select:none}
    .sc-atlas-world::before{content:"";position:absolute;inset:0;background-image:radial-gradient(circle,rgba(128,170,167,.19) .7px,transparent 1px);background-size:22px 22px;pointer-events:none}
    .sc-atlas-world canvas{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:grab;outline-offset:-4px}
    .sc-atlas-world.is-dragging canvas{cursor:grabbing}
    .sc-atlas-tags{position:absolute;inset:0;pointer-events:none}
    .sc-atlas-world .sc-atlas-tag{position:absolute;left:0;top:0;display:flex;align-items:flex-start;gap:7px;pointer-events:auto;white-space:nowrap;appearance:none;max-width:min(240px,39%);background:rgba(10,24,29,.84);border:1px solid rgba(129,160,159,.27);border-radius:7px;padding:7px 10px;color:#e6efef;box-shadow:0 2px 12px #06101544;line-height:1.18;text-align:left;cursor:pointer;font:inherit}
    .sc-atlas-world .sc-atlas-tag[aria-pressed="true"]{background:#172d31;border-color:#a6c6bd99}
    .sc-atlas-world .sc-atlas-tag:hover,.sc-atlas-world button:focus-visible{border-color:#a9dfcf;outline:2px solid #b8e0d6;outline-offset:2px}
    .sc-atlas-tag i{margin-top:4px;flex:0 0 6px;height:6px;border-radius:50%;background:var(--pop-pigment)}
    .sc-atlas-tag>span{min-width:0;display:grid;gap:3px}
    .sc-atlas-tag strong,.sc-atlas-tag small{display:block;overflow:hidden;text-overflow:ellipsis}
    .sc-atlas-tag strong{font-size:13px;font-weight:620;letter-spacing:-.015em;color:#eff8f7}
    .sc-atlas-tag small{font:500 10px/1.2 system-ui,sans-serif;color:#b9cece;font-variant-numeric:tabular-nums}
    .sc-atlas-pages{position:absolute;top:8px;right:10px;display:flex;align-items:center;background:#101c23ec;border:1px solid #52686b;border-radius:8px;font:11px/1.2 system-ui;padding:1px 3px;gap:6px}
    .sc-atlas-world .sc-atlas-tag[hidden],.sc-atlas-pages[hidden]{display:none}.sc-atlas-pages button{color:#d5e3df;border:0;background:transparent;padding:5px 8px;cursor:pointer;font:16px/1 system-ui}.sc-atlas-pages button:disabled{opacity:.35;cursor:default}
    @media(max-width:570px){.sc-atlas-world .sc-atlas-tag{padding:5px 7px;gap:5px;max-width:44%}.sc-atlas-tag strong{font-size:11px}.sc-atlas-tag small{font-size:9px}}
  `;

  function placement(n, w, h, index) {
    const portrait = h > w * 0.88;
    const cap = portrait ? 4 : 6;
    const flat = h < 235;
    let coords;
    if (n === 1) coords = [[0.5, 0.48]];
    else if (n === 2)
      coords = portrait
        ? [
            [0.33, 0.27],
            [0.66, 0.68],
          ]
        : [
            [0.31, 0.46],
            [0.7, 0.51],
          ];
    else if (n === 3)
      coords = portrait
        ? [
            [0.3, 0.2],
            [0.7, 0.49],
            [0.32, 0.77],
          ]
        : [
            [0.215, 0.5],
            [0.52, 0.32],
            [0.8, 0.62],
          ];
    else if (n === 4)
      coords = [
        [0.27, 0.24],
        [0.73, 0.27],
        [0.26, 0.7],
        [0.73, 0.72],
      ];
    else coords = Array.from({ length: n }, (_, i) => [((i % 3) + 0.5) / 3, i < 3 ? 0.27 : 0.72]);
    const tp = n > cap || flat ? 9 : 14;
    const labelSpace = flat ? 29 : 44;
    const used = Math.max(70, h - tp - labelSpace);
    const [a, b] = coords[index];
    const x = w * a,
      y = tp + used * b;
    let r = Math.min(
      w *
        (n === 1
          ? 0.39
          : n === 2
            ? 0.218
            : n === 3
              ? portrait
                ? 0.255
                : 0.167
              : portrait
                ? 0.21
                : 0.149),
      used *
        (n === 1
          ? 0.51
          : n === 2
            ? portrait
              ? 0.285
              : 0.51
            : n === 3
              ? portrait
                ? 0.208
                : 0.42
              : 0.254),
      209,
    );
    r = Math.min(r, x / 1.2, (w - x) / 1.2, (h - y - labelSpace) / 0.91);
    return { x, y, r: Math.max(12, r) };
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
      'Schematic population clusters. Use Tab to select a population, or drag to pan. Use plus and minus to zoom.',
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
    let pinch = null,
      drag = null;
    const compactMotion = () => opt.reducedMotion ?? media.matches;

    function wake() {
      dirty = true;
      if (!destroyed && !document.hidden && inViewport && !frame)
        frame = requestAnimationFrame(draw);
    }

    function reset() {
      zoom = targetZoom = 1;
      px = py = 0;
      wake();
    }
    function zoomBy(factor) {
      if (Number.isFinite(factor) && factor > 0) {
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
          wrap.append(element('strong', '', g.name), element('small'));
          btn.append(element('i'), wrap);
          btn.addEventListener('click', () => select(g.name));
          tags.append(btn);
          tagNodes.set(g.name, btn);
        }
        const m = models.get(g.name);
        const amount = Number.isFinite(g.value) ? number.format(g.value) : 'Unmeasured';
        const detail = opt.measurement ? amount : count.format(g.count || 0) + ' cells';
        const nMeasured = opt.measurement ? g.n : g.participants;
        btn.dataset.population = g.name;
        btn.dataset.n = String(nMeasured ?? 0);
        if (Number.isFinite(g.value)) btn.dataset.value = String(g.value);
        else delete btn.dataset.value;
        const readout =
          opt.measurement && Number.isFinite(g.value)
            ? 'Mean ' +
              detail +
              (Number.isFinite(nMeasured) ? ' · n=' + count.format(nMeasured) : '')
            : detail;
        btn.querySelector('small').textContent =
          opt.measurement && !Number.isFinite(g.value) ? readout + ' · n=0' : readout;
        const full =
          g.name +
          ' · ' +
          readout +
          (opt.measurement && opt.featureLabel ? ' · ' + opt.featureLabel : '') +
          (opt.visit ? ' · ' + opt.visit : '');
        btn.title = full;
        btn.setAttribute('aria-label', full);
        btn.setAttribute('aria-pressed', String(g.name === opt.selected || Boolean(g.selected)));
        btn.style.setProperty('--pop-pigment', 'hsl(' + m.hue + ' 62% 68%)');
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
    function showPage(i) {
      page = clamp(i, 0, Math.max(0, Math.ceil(groups.length / capacity) - 1));
      visible = groups.slice(page * capacity, (page + 1) * capacity);
      renderTags();
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
      groups = typed.length ? typed : supplied;
      const nextModels = new Map();
      groups.forEach((g, i) => {
        const model = models.get(g.name) || colony(g.name, i);
        const shade = options.measurement
          ? Number.isFinite(g.value)
            ? ratioColor(g.level)
            : '99,113,118'
          : null;
        if (model.shade !== shade) {
          model.priorShade = model.shade;
          model.changeStart = performance.now();
          model.shade = shade;
        }
        nextModels.set(g.name, model);
      });
      models = nextModels;
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
      capacity = h > w * 0.88 ? 4 : 6;
      if (w < 450 && h < 260) capacity = 2;
      if (w < 280 || h < 135) capacity = 1;
      if (old !== capacity) {
        const idx = groups.findIndex((g) => g.name === opt.selected);
        showPage(idx < 0 ? 0 : Math.floor(idx / capacity));
      } else wake();
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
      px = clamp(px, -w * zoom * 0.43, w * zoom * 0.43);
      py = clamp(py, -h * zoom * 0.43, h * zoom * 0.43);
      ctx.setTransform(density, 0, 0, density, 0, 0);
      ctx.clearRect(0, 0, w, h);
      hits = [];
      visible.forEach((g, i) => {
        const p = placement(visible.length, w, h, i),
          model = models.get(g.name);
        const x = (p.x - w / 2) * zoom + w / 2 + px,
          y = (p.y - h / 2) * zoom + h / 2 + py,
          r = p.r * zoom;
        const chosen = g.name === opt.selected || g.selected;
        hits.push({ x, y, r, name: g.name });
        const btn = tagNodes.get(g.name);
        const bx = clamp(x, btn.offsetWidth / 2 + 3, w - btn.offsetWidth / 2 - 3);
        btn.style.transform =
          'translate(' +
          Math.round(bx) +
          'px,' +
          Math.round(clamp(y + r * 0.98 + 3, 3, h - btn.offsetHeight - 3)) +
          'px) translateX(-50%)';
        btn.hidden = x + r < 0 || x - r > w || y + r < 0 || y - r > h;
        if (btn.hidden) return;
        const measured = Number.isFinite(g.value);
        const ratio = measured ? ratioColor(g.level) : '99,113,118';
        const aura = ctx.createRadialGradient(x, y, r * 0.04, x, y, r * 1.55);
        if (opt.measurement) {
          aura.addColorStop(0, 'rgba(' + ratio + ',.44)');
          aura.addColorStop(0.48, 'rgba(' + ratio + ',.22)');
          aura.addColorStop(0.72, 'rgba(' + ratio + ',.09)');
          aura.addColorStop(1, 'rgba(' + ratio + ',0)');
        } else {
          aura.addColorStop(0, 'hsla(' + model.hue + ',63%,58%,.17)');
          aura.addColorStop(0.64, 'hsla(' + model.hue + ',58%,42%,.075)');
          aura.addColorStop(1, 'hsla(' + model.hue + ',50%,38%,0)');
        }
        ctx.fillStyle = aura;
        ctx.fillRect(x - r * 1.55, y - r * 1.55, 3.1 * r, 3.1 * r);
        const observed = Number(g.count) > 0;
        if (observed) {
          for (const c of model.cells) {
            const phase = c.phase + time * 0.24;
            const xx = x + r * (c.x + 0.017 * Math.sin(phase)),
              yy = y + r * (c.y + 0.016 * Math.cos(phase * 1.1));
            const radius = c.r * r * (1 + 0.014 * Math.sin(phase));
            ctx.save();
            ctx.translate(xx, yy);
            ctx.rotate(c.rotation + 0.09 * Math.sin(phase));
            ctx.scale(c.stretch, 1 / c.stretch);
            const imageFor = (shade) =>
              shade
                ? tintLayer(model.name, model.hue, c.variant, shade)
                : layer(model.name, model.hue, c.variant);
            const mix =
              reduce || model.priorShade === undefined
                ? 1
                : clamp((stamp - model.changeStart) / 300, 0, 1);
            const extent = radius * 1.09;
            if (mix < 1)
              ctx.drawImage(imageFor(model.priorShade), -extent, -extent, 2 * extent, 2 * extent);
            ctx.globalAlpha = mix;
            ctx.drawImage(imageFor(model.shade), -extent, -extent, 2 * extent, 2 * extent);
            ctx.restore();
          }
        }
        const ring = opt.measurement
          ? measured
            ? 'rgb(' + ratioColor(g.level) + ')'
            : '#809094'
          : 'hsl(' + model.hue + ' 50% 65%)';
        ctx.save();
        ctx.translate(x, y);
        ctx.beginPath();
        ctx.ellipse(0, 0, r * 1.15, r * 0.94, 0, 0, circle);
        ctx.setLineDash((opt.measurement && !measured) || !observed ? [3, 5] : []);
        ctx.strokeStyle = ring;
        ctx.lineWidth = opt.measurement ? 2.6 : 1;
        ctx.globalAlpha = opt.measurement ? 0.94 : chosen ? 0.51 : 0.18;
        ctx.stroke();
        if (chosen) {
          ctx.globalAlpha = 0.91;
          ctx.setLineDash([]);
          ctx.strokeStyle = '#d7eeda';
          ctx.lineWidth = 1.6;
          for (const start of [0.09, Math.PI + 0.09]) {
            ctx.beginPath();
            ctx.ellipse(0, 0, r * 1.19, r * 0.98, 0, start, start + 0.2);
            ctx.stroke();
          }
        }
        ctx.restore();
      });
      if (!reduce || Math.abs(zoom - targetZoom) > 0.001) frame = requestAnimationFrame(draw);
    }

    function point(e) {
      const b = canvas.getBoundingClientRect();
      return { x: e.clientX - b.left, y: e.clientY - b.top };
    }
    function pick(p) {
      return hits
        .map((s) => ({
          s,
          d: Math.hypot((p.x - s.x) / 1.15, (p.y - s.y) / 0.94) / s.r,
        }))
        .filter((a) => a.d <= 1.03)
        .sort((a, b) => a.d - b.d)[0]?.s;
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
      if (fingers.size === 1) drag = { start: p, px, py, moved: false };
      else if (fingers.size === 2) {
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
        zoomBy(Math.exp(clamp(-e.deltaY * (e.deltaMode === 1 ? 0.024 : 0.0017), -0.5, 0.5)));
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
  window.SctrialAtlasScene = Object.freeze({ mount });
})();
