(() => {
  'use strict';
  const fmt = new Intl.NumberFormat('en-US');
  const palettes = [169, 274, 38, 207, 330, 115, 20];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function hash(label) {
    let n = 2166136261;
    for (const c of String(label)) n = Math.imul(n ^ c.charCodeAt(0), 16777619);
    return n >>> 0;
  }
  function hueFor(label, index) {
    if (/\bCD8\b|cytotoxic/i.test(label)) return 171;
    if (/\bB[ -]?(?:cell|lymph)/i.test(label)) return 273;
    if (/\bCD4\b/i.test(label)) return 38;
    return palettes[index % palettes.length];
  }
  function statsFor(config) {
    const totals = new Map(
      config.populations.map((name) => [name, { cells: 0, people: new Set() }]),
    );
    for (const row of config.observations) {
      const total = totals.get(row.population);
      if (total && row.cells > 0) {
        total.cells += row.cells;
        total.people.add(row.id);
      }
    }
    return totals;
  }
  let current;
  function build(host, config) {
    let options = config,
      selected = config.selected;
    const stats = statsFor(config);
    const models = config.populations.filter((name) => name !== config.allLabel);
    if (!models.length) models.push(...config.populations);
    const species = models.map((name, i) => ({
      name,
      seed: hash(name),
      hue: hueFor(name, i),
      i,
      rotation: 0,
    }));
    const scene = el('section', 'cell-scene');
    scene.setAttribute('aria-label', 'Cell population explorer');
    const stage = el('div', 'cell-world');
    const canvas = el('canvas', 'cell-field');
    canvas.tabIndex = 0;
    canvas.setAttribute(
      'aria-label',
      'Illustrative cell models. Drag to rotate, use arrow keys to rotate, and plus or minus to zoom.',
    );
    const ctx = canvas.getContext('2d', { alpha: true });
    const header = el('header', 'cell-heading');
    header.append(el('h1', '', 'Cell populations'));
    const summary = el('p', 'cell-summary');
    header.append(summary);
    const labelLayer = el('div', 'cell-annotations');
    const caption = el('span', 'cell-schematic', 'Illustrative cell models');
    const dock = el('div', 'cell-dock');
    const picker = el('label', 'cell-picker');
    const popIcon = el('span', 'cell-pop-icon');
    popIcon.setAttribute('aria-hidden', 'true');
    const select = el('select');
    select.setAttribute('aria-label', 'Cell population');
    for (const name of options.populations) select.add(new Option(name, name));
    picker.append(popIcon, select, el('i', 'cell-chevron'));
    const analyze = el('button', 'cell-analyze', 'Analyze population');
    analyze.type = 'button';
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    arrow.setAttribute('viewBox', '0 0 16 16');
    arrow.setAttribute('width', '14');
    arrow.setAttribute('height', '14');
    arrow.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M3 8h9m-4-4 4 4-4 4');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '1.6');
    arrow.append(path);
    analyze.append(arrow);
    const reset = el('button', 'cell-reset', 'Reset view');
    reset.type = 'button';
    reset.hidden = true;
    dock.append(picker, analyze);
    const help = el('div', 'cell-gestures', 'Drag to rotate · Scroll to zoom');
    const bottom = el('div', 'cell-bottom');
    bottom.append(help, dock);
    stage.append(canvas, labelLayer);
    scene.append(stage, header, caption, reset, bottom);
    host.replaceChildren(scene);

    let width = 0,
      height = 0,
      scale = 1,
      zoom = 1,
      targetZoom = 1;
    let raf = 0,
      alive = true,
      last = 0,
      clock = 0,
      drawAgain = true;
    let drag = null,
      hover = null,
      shown = [],
      anchorSet = new Map(),
      keyboardAngle = 0;
    const pointers = new Map();
    let pinchDistance = 0;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const geometries = new Map();
    const out = new ResizeObserver(resize);
    out.observe(stage);
    const visibility = () => {
      if (!document.hidden) queue();
      else {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    };
    document.addEventListener('visibilitychange', visibility);
    function queue() {
      drawAgain = true;
      if (alive && !raf && !document.hidden) raf = requestAnimationFrame(draw);
    }
    function resize() {
      const box = stage.getBoundingClientRect();
      if (!box.width || !box.height) return;
      width = box.width;
      height = box.height;
      scale = Math.min(devicePixelRatio || 1, 1.65);
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      updateShown();
      queue();
    }
    function updateShown() {
      const capacity = width < 630 && height < 480 ? 2 : 3;
      const index = models.indexOf(selected);
      const first =
        index < 0 || species.length <= capacity ? 0 : Math.floor(index / capacity) * capacity;
      const next = species.slice(first, first + capacity);
      if (next.length === shown.length && next.every((s, i) => s === shown[i])) return;
      shown = next;
      caption.textContent =
        shown.length < species.length
          ? 'Illustrative cell models · ' +
            (first + 1) +
            '–' +
            (first + shown.length) +
            ' of ' +
            species.length
          : 'Illustrative cell models';
      anchorSet = new Map();
      labelLayer.replaceChildren();
      for (const s of shown) {
        const button = el('button', 'cell-tag');
        button.type = 'button';
        button.setAttribute('aria-label', 'Select ' + s.name);
        button.style.setProperty('--marker-hue', String(s.hue));
        button.append(el('i', 'cell-tag-dot'), el('span', '', s.name));
        button.onclick = () => selectPop(s.name);
        labelLayer.append(button);
        anchorSet.set(s.name, button);
      }
    }
    function selectPop(name) {
      if (!options.populations.includes(name)) return;
      if (name !== selected) {
        selected = name;
        refresh();
        options.onSelect(name);
      }
    }
    function refresh() {
      updateShown();
      select.value = selected;
      const n = stats.get(selected);
      summary.textContent = n
        ? fmt.format(n.cells) + ' cells  ·  ' + fmt.format(n.people.size) + ' participants'
        : '';
      for (const [label, but] of anchorSet)
        but.setAttribute('aria-pressed', String(label === selected));
      scene.dataset.selected = String(shown.some((s) => s.name === selected));
      queue();
    }
    function targetFor(s, i) {
      const compact = width < 630;
      const n = shown.length;
      const free = Math.max(120, height - (compact ? 175 : 140));
      const top = compact ? 92 : 60;
      const active = s.name === selected;
      const hasSelection = shown.some((o) => o.name === selected) && n > 1;
      let x, y, r;
      if (n === 1) {
        x = width * 0.5;
        y = top + free * 0.49;
        r = Math.min(width * (compact ? 0.28 : 0.24), free * 0.36, 223);
      } else if (compact && n === 3) {
        x = width * [0.27, 0.72, 0.43][i];
        y = top + free * [0.28, 0.48, 0.78][i];
        r = Math.min(width * 0.18, free * 0.155, 85);
      } else if (compact) {
        x = width * (i === 0 ? 0.33 : 0.69);
        y = top + free * (i === 0 ? 0.37 : 0.68);
        r = Math.min(width * 0.2, free * 0.205);
      } else if (n === 2) {
        x = width * (i === 0 ? 0.36 : 0.69);
        y = top + free * (i === 0 ? 0.58 : 0.39);
        r = Math.min(
          width * (i === 0 ? 0.158 : 0.142),
          free * (i === 0 ? 0.32 : 0.285),
          i === 0 ? 215 : 188,
        );
      } else {
        x = width * [0.23, 0.52, 0.79][i];
        y = top + free * [0.56, 0.36, 0.64][i];
        r = Math.min(width * 0.11, free * 0.235, 150);
      }
      if (hasSelection) {
        r *= active ? 1.16 : 0.83;
        y += active ? -4 : 8;
      }
      return { x, y, r: r * zoom, alpha: hasSelection && !active ? 0.72 : 1 };
    }
    function draw(time) {
      raf = 0;
      if (!alive || !width || !height || !ctx) return;
      const dt = last ? Math.min((time - last) / 1000, 0.07) : 0.016;
      last = time;
      clock += dt;
      const ease = motion.matches ? 1 : 1 - Math.exp(-dt * 10);
      zoom += (targetZoom - zoom) * ease;
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.clearRect(0, 0, width, height);
      // These low-contrast specks belong to the optical background, not to the study sample.
      {
        for (let i = 0; i < 70; i++) {
          const fx = ((i * 73.3147) % 97) / 97,
            fy = ((i * 41.777 + 9) % 101) / 101;
          const t = motion.matches ? 0 : clock;
          ctx.beginPath();
          ctx.fillStyle = i % 4 === 0 ? 'rgba(172,220,206,.24)' : 'rgba(150,192,205,.13)';
          ctx.arc(
            fx * width + Math.sin(i * 8.2 + t * 0.13) * 3,
            fy * height + Math.cos(i + t * 0.11) * 2,
            i % 8 === 0 ? 1.3 : 0.67,
            0,
            Math.PI * 2,
          );
          ctx.fill();
        }
      }
      let moving = Math.abs(zoom - targetZoom) > 0.002;
      shown.forEach((s, i) => {
        const tar = targetFor(s, i);
        let g = geometries.get(s.name);
        if (!g) {
          g = { ...tar };
          geometries.set(s.name, g);
        }
        for (const key of ['x', 'y', 'r', 'alpha']) {
          moving ||= Math.abs(tar[key] - g[key]) > 0.08;
          g[key] += (tar[key] - g[key]) * ease;
        }
        const t = motion.matches ? 0 : clock;
        const x = g.x + Math.sin(t * 0.25 + s.i * 9) * Math.min(3, g.r * 0.04);
        const y = g.y + Math.sin(t * 0.42 + s.i * 5) * Math.min(4, g.r * 0.035);
        g.screenX = x;
        g.screenY = y;
        ctx.save();
        ctx.globalAlpha = g.alpha;
        const aura = ctx.createRadialGradient(x, y, g.r * 0.55, x, y, g.r * 1.5);
        aura.addColorStop(0, 'hsla(' + s.hue + ',62%,51%,.13)');
        aura.addColorStop(1, 'hsla(' + s.hue + ',55%,45%,0)');
        ctx.fillStyle = aura;
        ctx.fillRect(x - g.r * 1.6, y - g.r * 1.6, g.r * 3.2, g.r * 3.2);
        window.SctrialCellArt.paint(ctx, {
          x,
          y,
          radius: g.r,
          kind: s.name,
          hue: s.hue,
          rotation:
            s.rotation +
            keyboardAngle +
            (motion.matches ? 0 : Math.sin(clock * 0.115 + s.i) * 0.12),
          seed: s.seed,
          detail: drag || moving ? 0.6 : 0.9,
        });
        ctx.restore();
        const but = anchorSet.get(s.name);
        if (but) {
          const half = (but.offsetWidth || 100) / 2;
          const bx = clamp(x, half + 9, width - half - 9);
          const by = clamp(y + g.r * 1.04 + 10, 82, height - (width < 630 ? 96 : 78));
          but.style.transform =
            'translate(' + Math.round(bx) + 'px,' + Math.round(by) + 'px) translateX(-50%)';
          but.classList.toggle('is-hovered', hover === s.name);
        }
      });
      if (moving || !motion.matches || drawAgain) {
        drawAgain = false;
        raf = requestAnimationFrame(draw);
      }
    }
    const point = (e) => {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    };
    function hit(p) {
      for (const s of shown) {
        const g = geometries.get(s.name);
        if (g && Math.hypot(p.x - g.screenX, p.y - g.screenY) <= g.r * 1.1) return s;
      }
      return null;
    }
    canvas.addEventListener('pointerdown', (e) => {
      const p = point(e);
      pointers.set(e.pointerId, p);
      canvas.setPointerCapture(e.pointerId);
      if (pointers.size === 2) {
        const [a, b] = pointers.values();
        pinchDistance = Math.hypot(a.x - b.x, a.y - b.y);
        if (drag) drag.moved = true;
        return;
      }
      const s = hit(p) || shown.find((a) => a.name === selected) || shown[0];
      drag = { id: e.pointerId, start: p, last: p, s, moved: false };
      canvas.classList.add('is-dragging');
    });
    canvas.addEventListener('pointermove', (e) => {
      const p = point(e);
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
      if (pointers.size === 2) {
        const [a, b] = pointers.values();
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDistance > 0) targetZoom = clamp((targetZoom * dist) / pinchDistance, 0.7, 1.45);
        pinchDistance = dist;
        reset.hidden = false;
        queue();
        return;
      }
      if (drag && drag.id === e.pointerId) {
        if (Math.hypot(p.x - drag.start.x, p.y - drag.start.y) > 4) drag.moved = true;
        if (drag.moved && drag.s) {
          drag.s.rotation += (p.x - drag.last.x) * 0.012 + (p.y - drag.last.y) * 0.002;
          reset.hidden = false;
          queue();
        }
        drag.last = p;
      } else {
        const under = hit(p);
        hover = under?.name || null;
        canvas.classList.toggle('cell-under-pointer', !!under);
        queue();
      }
    });
    function release(e, activate) {
      const p = point(e);
      if (activate && drag?.id === e.pointerId && !drag.moved) {
        const s = hit(p);
        if (s) selectPop(s.name);
      }
      pointers.delete(e.pointerId);
      drag = null;
      pinchDistance = 0;
      canvas.classList.remove('is-dragging');
    }
    canvas.addEventListener('pointerup', (e) => release(e, true));
    canvas.addEventListener('pointercancel', (e) => release(e, false));
    canvas.addEventListener('pointerleave', () => {
      hover = null;
      queue();
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        targetZoom = clamp(targetZoom * Math.exp(-e.deltaY * 0.0014), 0.7, 1.45);
        reset.hidden = false;
        queue();
      },
      { passive: false },
    );
    canvas.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        keyboardAngle += e.key === 'ArrowLeft' ? -0.22 : 0.22;
      } else if (e.key === '+' || e.key === '=') targetZoom = clamp(targetZoom + 0.1, 0.7, 1.45);
      else if (e.key === '-') targetZoom = clamp(targetZoom - 0.1, 0.7, 1.45);
      else return;
      e.preventDefault();
      reset.hidden = false;
      queue();
    });
    reset.onclick = () => {
      targetZoom = 1;
      keyboardAngle = 0;
      for (const s of species) s.rotation = 0;
      reset.hidden = true;
      queue();
    };
    select.onchange = () => selectPop(select.value);
    analyze.onclick = () => options.onAnalyze(selected);
    if (!ctx) {
      help.textContent = 'Select a population to inspect its measurements.';
    }
    refresh();
    resize();
    return {
      host,
      config,
      update(next) {
        options = next;
        selected = next.selected;
        refresh();
      },
      stop() {
        alive = false;
        out.disconnect();
        cancelAnimationFrame(raf);
        document.removeEventListener('visibilitychange', visibility);
      },
    };
  }
  window.SctrialCells = {
    render(host, config) {
      if (!host) return;
      if (current?.host === host && current.config.observations === config.observations) {
        current.update(config);
        return;
      }
      current?.stop();
      current = build(host, config);
    },
    dispose(host) {
      if (!host || current?.host === host) {
        current?.stop();
        current = null;
      }
    },
  };
})();
