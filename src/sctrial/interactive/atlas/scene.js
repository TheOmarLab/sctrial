// The display uses every eligible observation at the two recorded coordinates.
// Visits are independent point layers. Mixing changes opacity only.
const ramp = ['#29394b', '#694d91', '#cb648e', '#fb9b64', '#fff0ad'];
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const rampRGB = ramp.map(rgb);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const signalPalette = Array.from({ length: 256 }, (_, i) => {
  const t = (i * 4) / 255,
    j = Math.min(3, Math.floor(t));
  return rampRGB[j].map((v, k) => v + (t - j) * (rampRGB[j + 1][k] - v));
});
const vertex = `
attribute vec2 a_point;
attribute vec3 a_color, a_previousColor;
attribute float a_selected, a_previousSelected;
uniform float u_mix;
uniform vec2 u_origin, u_view;
uniform float u_scale, u_size, u_emphasis;
uniform mediump float u_halo;
varying mediump vec3 v_color;
varying mediump float v_selected;
void main() {
  vec2 p = u_origin + vec2(a_point.x,-a_point.y) * u_scale;
  gl_Position=vec4(p/u_view*vec2(2.,-2.)+vec2(-1.,1.),0.,1.);
  v_selected=mix(1.,mix(a_previousSelected,a_selected,u_mix),u_emphasis);
  gl_PointSize=u_size*mix(.76,1.,v_selected)*mix(1.,3.6,u_halo);
  v_color=mix(a_previousColor,a_color,u_mix);
}`;
const fragment = `
precision mediump float;
varying mediump vec3 v_color;
varying mediump float v_selected;
uniform float u_alpha, u_halo, u_mask;
uniform vec3 u_lens;
void main(){
  if(u_mask!=0.) {
    float inside = step(distance(gl_FragCoord.xy,u_lens.xy),u_lens.z);
    if((u_mask>0. && inside<.5)||(u_mask<0. && inside>.5)) discard;
  }
  vec2 q=gl_PointCoord*2.-1.; float r2=dot(q,q);
  if(r2>1.) discard;
  vec3 c=v_color;
  float alpha;
  if(u_halo>.5) {
    alpha=exp(-r2*4.8)*.095;
  } else {
    alpha=(1.-smoothstep(.25,1.,r2))*.85;
    float shine=max(0.,1.-length(q+vec2(.23,.28)))*.24;
    c=c*(1.-r2*.16)+c*shine;
  }
  alpha*=u_alpha*mix(.56,1.,v_selected);
  gl_FragColor=vec4(c,clamp(alpha,0.,1.));
}`;

function gpuRenderer(canvas) {
  const gl = canvas.getContext('webgl', {
    alpha: true,
    antialias: false,
    premultipliedAlpha: true,
    powerPreference: 'high-performance'
  });
  if (!gl) return null;
  const shaders = [
    [gl.VERTEX_SHADER, vertex],
    [gl.FRAGMENT_SHADER, fragment]
  ].map(([type, source]) => {
    const obj = gl.createShader(type);
    gl.shaderSource(obj, source);
    gl.compileShader(obj);
    return obj;
  });
  const program = gl.createProgram();
  shaders.forEach((s) => gl.attachShader(program, s));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
  gl.useProgram(program);
  const uniforms = Object.fromEntries(
    ['origin', 'view', 'scale', 'size', 'alpha', 'halo', 'emphasis', 'mask', 'lens', 'mix'].map((k) => [
      k,
      gl.getUniformLocation(program, 'u_' + k)
    ])
  );
  const attributes = ['point', 'color', 'selected', 'previousColor', 'previousSelected'].map((k) =>
    gl.getAttribLocation(program, 'a_' + k)
  );
  attributes.forEach((k) => gl.enableVertexAttribArray(k));
  const buffers = [gl.createBuffer(), gl.createBuffer()];
  const oldBuffers = [gl.createBuffer(), gl.createBuffer()];
  const counts = [0, 0];
  const maxPointSize = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  return {
    upload(layers, previous = layers) {
      layers.forEach((values, i) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers[i]);
        gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, oldBuffers[i]);
        gl.bufferData(gl.ARRAY_BUFFER, previous[i], gl.STATIC_DRAW);
        counts[i] = values.length / 6;
      });
    },
    draw(frame) {
      const { w, h, ratio, x, y, scale, size, passes, emphasis } = frame;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.origin, x, y);
      gl.uniform2f(uniforms.view, w, h);
      gl.uniform1f(uniforms.scale, scale);
      gl.uniform1f(uniforms.size, Math.min(size * ratio, maxPointSize / 3.6));
      gl.uniform1f(uniforms.emphasis, emphasis);
      gl.uniform1f(uniforms.mix, frame.mix);
      if (frame.lens)
        gl.uniform3f(uniforms.lens, frame.lens.x * ratio, (h - frame.lens.y) * ratio, frame.lens.r * ratio);
      for (const { visit, alpha, mask = 0 } of passes) {
        if (alpha < 0.001) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, buffers[visit]);
        gl.vertexAttribPointer(attributes[0], 2, gl.FLOAT, false, 24, 0);
        gl.vertexAttribPointer(attributes[1], 3, gl.FLOAT, false, 24, 8);
        gl.vertexAttribPointer(attributes[2], 1, gl.FLOAT, false, 24, 20);
        gl.bindBuffer(gl.ARRAY_BUFFER, oldBuffers[visit]);
        gl.vertexAttribPointer(attributes[3], 3, gl.FLOAT, false, 24, 8);
        gl.vertexAttribPointer(attributes[4], 1, gl.FLOAT, false, 24, 20);
        gl.uniform1f(uniforms.alpha, alpha);
        gl.uniform1f(uniforms.mask, mask);
        for (const halo of [1, 0]) {
          gl.uniform1f(uniforms.halo, halo);
          gl.drawArrays(gl.POINTS, 0, counts[visit]);
        }
      }
    },
    destroy() {
      buffers.concat(oldBuffers).forEach((b) => gl.deleteBuffer(b));
      shaders.forEach((s) => gl.deleteShader(s));
      gl.deleteProgram(program);
    }
  };
}

// Use the same complete vertex layers when WebGL is unavailable. Cache each
// observed visit independently so timeline scrubbing only composites two images.
function softwareRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  let layers = [],
    prior = null,
    cacheKey = '';
  const cache = new Map();
  return {
    upload(values, previous) {
      // Snapshot the frame already visible, including any interrupted reveal.
      // The cached endpoint is redrawn at full resolution beneath it.
      if (previous && canvas.width > 1 && canvas.height > 1) {
        prior = document.createElement('canvas');
        prior.width = canvas.width;
        prior.height = canvas.height;
        prior.getContext('2d').drawImage(canvas, 0, 0);
      } else prior = null;
      layers = values;
      cache.clear();
    },
    draw(frame) {
      const { w, h, ratio, x, y, scale, size, passes, emphasis } = frame;
      const key = [w, h, ratio, x, y, scale, size, emphasis].join('/');
      if (key !== cacheKey) {
        cacheKey = key;
        cache.clear();
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const { visit, alpha, mask = 0 } of passes) {
        if (alpha < 0.001 || !layers[visit]) continue;
        const id = visit;
        let surface = cache.get(id);
        if (!surface) {
          surface = document.createElement('canvas');
          surface.width = canvas.width;
          surface.height = canvas.height;
          const g = surface.getContext('2d');
          g.scale(ratio, ratio);
          const v = layers[visit],
            paints = new Map();
          for (let i = 0; i < v.length; i += 6) {
            const px = x + v[i] * scale,
              py = y - v[i + 1] * scale;
            if (px < -size || px > w + size || py < -size || py > h + size) continue;
            const a = !emphasis || v[i + 5] ? 0.83 : 0.45;
            const r = Math.round(v[i + 2] * 255),
              green = Math.round(v[i + 3] * 255),
              b = Math.round(v[i + 4] * 255);
            const color = (r << 16) | (green << 8) | b;
            if (!paints.has(color)) paints.set(color, 'rgb(' + r + ',' + green + ',' + b + ')');
            g.fillStyle = paints.get(color);
            g.globalAlpha = a;
            const s = size * (!emphasis || v[i + 5] ? 1 : 0.76) * 0.4;
            g.beginPath();
            g.arc(px, py, s, 0, Math.PI * 2);
            g.fill();
          }
          cache.set(id, surface);
        }
        ctx.save();
        if (mask) {
          const lens = frame.lens;
          ctx.beginPath();
          if (mask < 0) ctx.rect(0, 0, canvas.width, canvas.height);
          ctx.arc(lens.x * ratio, lens.y * ratio, lens.r * ratio, 0, Math.PI * 2);
          ctx.clip('evenodd');
        }
        ctx.globalAlpha = alpha;
        ctx.drawImage(surface, 0, 0);
        ctx.restore();
      }
      if (prior && frame.mix < 1) {
        ctx.globalAlpha = 1 - frame.mix;
        ctx.drawImage(prior, 0, 0);
      }
      if (frame.mix >= 1) prior = null;
      ctx.globalAlpha = 1;
    },
    destroy() {
      cache.clear();
    }
  };
}

export function createScene(
  canvas,
  { onSelect = () => {}, onInspect = () => {}, onCamera = () => {} } = {}
) {
  const parent = canvas.parentElement;
  if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
  const overlay = document.createElement('canvas');
  overlay.setAttribute('aria-hidden', 'true');
  Object.assign(overlay.style, { position: 'absolute', pointerEvents: 'none', display: 'block' });
  canvas.insertAdjacentElement('afterend', overlay);
  const hud = overlay.getContext('2d');
  let surface = canvas,
    renderer;
  function useSoftware() {
    // A canvas that has acquired a WebGL context cannot acquire a 2D context.
    if (canvas.getContext('webgl') !== null) {
      surface = document.createElement('canvas');
      Object.assign(surface.style, { position: 'absolute', pointerEvents: 'none', display: 'block' });
      canvas.insertAdjacentElement('afterend', surface);
    }
    renderer = softwareRenderer(surface);
  }
  try {
    renderer = gpuRenderer(canvas);
  } catch {
    renderer = null;
  }
  if (!renderer) useSoftware();
  const abort = new AbortController();
  const listen = (el, event, fn, options = {}) =>
    el.addEventListener(event, fn, { ...options, signal: abort.signal });
  Object.assign(canvas.style, { touchAction: 'none', cursor: 'grab', display: 'block' });
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  let state = null,
    layout = null,
    signature = '',
    raw = null,
    pointsRef = null,
    metaRef = null;
  let sx, sy, xDomain, yDomain, xy, pops, arms, visits, view, signalOffset;
  let layers = [],
    previousLayers = null,
    indexes = [],
    grid = new Map(),
    anchors = [],
    hitLabels = [],
    summaries = new Map();
  let colorStart = -1;
  const colorDuration = 330;
  function colorMix(now) {
    if (colorStart < 0 || state?.reduced) return 1;
    const t = clamp((now - colorStart) / colorDuration, 0, 1);
    return t * t * (3 - 2 * t);
  }
  const guards = Array.from(
    parent.querySelectorAll('.field-label,.camera-tools,.field-legend,.focus-tool')
  );
  let reserved = [];
  let width = 1,
    height = 1,
    ratio = 1,
    zoom = 1,
    panX = 0,
    panY = 0,
    fit = 1,
    frameId = 0;
  let hover = null,
    pulseStart = -1,
    disposed = false,
    dragging = false,
    pinch = null,
    flight = null;
  const pointers = new Map();
  let lensUnit = null,
    lensDrag = null;
  function lensCircle() {
    const r = Math.min(clamp(Math.min(width, height) * 0.193, 55, 105), Math.min(width, height) * 0.39);
    const p = lensUnit || { x: 0.5, y: 0.49 };
    return {
      r,
      x: clamp(p.x * width, r + 4, width - r - 4),
      y: clamp(p.y * height, r + 4, height - r - 4)
    };
  }
  function moveLens(x, y) {
    lensUnit = { x: clamp(x / width, 0, 1), y: clamp(y / height, 0, 1) };
    invalidate();
  }
  function projectX(wx) {
    return width / 2 + panX + wx * fit * zoom;
  }
  function projectY(wy) {
    return height / 2 + panY - wy * fit * zoom;
  }
  function invalidate() {
    if (!frameId && !disposed) frameId = requestAnimationFrame(render);
  }
  function measure() {
    const rect = canvas.getBoundingClientRect(),
      parentRect = parent.getBoundingClientRect();
    const w = Math.max(1, rect.width),
      h = Math.max(1, rect.height),
      d = Math.min(2.5, window.devicePixelRatio || 1);
    for (const c of surface === canvas ? [overlay] : [surface, overlay])
      Object.assign(c.style, {
        width: w + 'px',
        height: h + 'px',
        left: rect.left - parentRect.left - parent.clientLeft + parent.scrollLeft + 'px',
        top: rect.top - parentRect.top - parent.clientTop + parent.scrollTop + 'px'
      });
    const resized = w !== width || h !== height || ratio !== d || canvas.width !== Math.round(w * d);
    if (resized) {
      if (width > 1 && height > 1) {
        panX = (panX * w) / width;
        panY = (panY * h) / height;
      }
      width = w;
      height = h;
      ratio = d;
      for (const c of new Set([canvas, overlay, surface])) {
        c.width = Math.round(w * d);
        c.height = Math.round(h * d);
      }
      if (sx) fit = 0.88 * Math.min(width / sx, height / sy);
    }
    reserved = guards
      .filter((el) => el.getClientRects().length)
      .map((el) => {
        const b = el.getBoundingClientRect();
        return { x: b.left - rect.left - 3, y: b.top - rect.top - 3, w: b.width + 6, h: b.height + 6 };
      });
    if (resized && state?.points && raw) {
      // A resize clears the browser's drawing buffers. Replace them before
      // this layout pass is presented, including when an inspector grows.
      cancelAnimationFrame(frameId);
      frameId = 0;
      render(performance.now());
    } else invalidate();
  }
  const watcher = new ResizeObserver(measure);
  watcher.observe(canvas);
  guards.forEach((el) => watcher.observe(el));
  measure();
  function decode() {
    layout = state.study.map;
    view = ArrayBuffer.isView(state.points)
      ? new DataView(state.points.buffer, state.points.byteOffset, state.points.byteLength)
      : new DataView(state.points);
    const n = layout.rows;
    if (n * layout.stride > view.byteLength)
      throw new Error('The cell map is shorter than its declared row count.');
    xDomain = layout.xDomain;
    yDomain = layout.yDomain;
    const dx = xDomain[1] - xDomain[0],
      dy = yDomain[1] - yDomain[0],
      span = Math.max(dx, dy) || 1;
    sx = dx / span || 1;
    sy = dy / span || 1;
    fit = 0.88 * Math.min(width / sx, height / sy);
    const bytes = layout.categoryBytes || 1;
    const indexType = { 1: Uint8Array, 2: Uint16Array, 4: Uint32Array }[bytes];
    if (!indexType) throw Error('Unsupported category index width.');
    const readIndex = bytes === 1 ? 'getUint8' : bytes === 2 ? 'getUint16' : 'getUint32';
    const visitOffset = 4 + 2 * bytes;
    signalOffset = visitOffset + 1;
    xy = new Float32Array(n * 2);
    pops = new indexType(n);
    arms = new indexType(n);
    visits = new Uint8Array(n);
    for (let row = 0; row < n; row++) {
      const i = row * layout.stride;
      xy[2 * row] = dx ? (view.getUint16(i, true) / 65535 - 0.5) * sx : 0;
      xy[2 * row + 1] = dy ? (view.getUint16(i + 2, true) / 65535 - 0.5) * sy : 0;
      pops[row] = view[readIndex](i + 4, true);
      arms[row] = view[readIndex](i + 4 + bytes, true);
      visits[row] = view.getUint8(i + visitOffset);
    }
    summaries = new Map(state.study.groups.map((g) => [[g.arm, g.visit, g.population].join('|'), g]));
    raw = state.points;
  }
  function buildIndex() {
    const arm = state.study.arms.findIndex((a) => a.id === state.arm);
    indexes = [[], []];
    grid = new Map();
    const bins = state.study.populations.map(() => new Map());
    for (let r = 0; r < layout.rows; r++) {
      if (arm > 0 && arms[r] !== arm) continue;
      const v = visits[r],
        p = pops[r];
      if (v > 1 || p >= bins.length) continue;
      indexes[v].push(r);
      const x = xy[r * 2],
        y = xy[r * 2 + 1],
        gx = Math.floor((x + 0.5) * 72),
        gy = Math.floor((y + 0.5) * 72);
      const k = gx + gy * 128;
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(r);
      const k2 = (gx >> 1) + (gy >> 1) * 64;
      const m = bins[p];
      let b = m.get(k2);
      if (!b) {
        b = [0, 0, 0];
        m.set(k2, b);
      }
      b[0]++;
      b[1] += x;
      b[2] += y;
    }
    anchors = bins.map((bins, p) => {
      let best = null,
        local = -1;
      for (const [key, b] of bins) {
        let score = b[0];
        for (const offset of [-65, -64, -63, -1, 1, 63, 64, 65])
          score += (bins.get(key + offset)?.[0] || 0) * 0.3;
        if (score > local) {
          local = score;
          best = b;
        }
      }
      return best ? { p, x: best[1] / best[0], y: best[2] / best[0] } : null;
    });
  }
  function featureId() {
    return state.layer === 'pathway' ? state.pathway : state.gene;
  }
  function upload(animate) {
    const now = performance.now(),
      mix = colorMix(now);
    let before = animate ? layers : null;
    if (before && previousLayers && mix < 1) {
      before = layers.map((v, idx) => {
        const from = previousLayers[idx],
          visible = new Float32Array(v);
        for (let i = 0; i < v.length; i += 6)
          for (let j = 2; j < 6; j++) visible[i + j] = from[i + j] + (v[i + j] - from[i + j]) * mix;
        return visible;
      });
    }
    const palette = state.study.populations.map((p) => rgb(p.color || '#89c9d4'));
    const selected = state.study.populations.findIndex((p) => p.id === state.pop);
    const f = state.layer === 'composition' ? -1 : layout.featureIds.indexOf(featureId());
    layers = indexes.map((ids) => {
      const out = new Float32Array(ids.length * 6);
      ids.forEach((r, j) => {
        const i = j * 6,
          p = pops[r],
          focus = p === selected;
        let color = palette[p];
        if (f >= 0)
          color =
            focus || selected < 0
              ? signalPalette[view.getUint8(r * layout.stride + signalOffset + f)]
              : [0.19, 0.27, 0.32];
        out[i] = xy[2 * r];
        out[i + 1] = xy[2 * r + 1];
        out[i + 2] = color[0];
        out[i + 3] = color[1];
        out[i + 4] = color[2];
        out[i + 5] = focus ? 1 : 0;
      });
      return out;
    });
    colorStart = before ? now : -1;
    previousLayers = before;
    renderer.upload(layers, before || undefined);
  }
  function setState(next) {
    const old = state;
    state = next ? { ...next } : null;
    if (!state?.study?.map || !state.points) {
      invalidate();
      return;
    }
    const fresh = pointsRef !== state.points || metaRef !== state.study;
    if (fresh) {
      zoom = 1;
      panX = 0;
      panY = 0;
      lensUnit = null;
      flight = null;
      previousLayers = null;
      pulseStart = -1;
      decode();
      pointsRef = state.points;
      metaRef = state.study;
    }
    if (fresh || old?.arm !== state.arm) buildIndex();
    if (state.compare && !old?.compare) {
      clearHover();
      if (!lensUnit) {
        const index = state.study.populations.findIndex((p) => p.id === state.pop),
          a = anchors[index];
        if (a) moveLens(projectX(a.x), projectY(a.y));
      }
    }
    const sig = [state.arm, state.pop, state.gene, state.pathway, state.layer].join('|');
    if (fresh || sig !== signature) {
      upload(!fresh && old?.arm === state.arm && !state.reduced);
      signature = sig;
    }
    if (state.reduced) flight = null;
    if (
      state.reduced ||
      old?.visit !== state.visit ||
      old?.arm !== state.arm ||
      old?.compare !== state.compare
    ) {
      colorStart = -1;
      previousLayers = null;
    }
    if (old && old.pop !== state.pop) pulseStart = state.reduced ? -1 : performance.now();
    if (
      fresh ||
      old?.arm !== state.arm ||
      old?.visit !== state.visit ||
      old?.pop !== state.pop ||
      old?.layer !== state.layer ||
      old?.gene !== state.gene ||
      old?.pathway !== state.pathway ||
      old?.compare !== state.compare
    ) {
      clearHover();
      const population =
        state.study.populations.find((p) => p.id === state.pop)?.label || 'all populations';
      const timestep = state.study.visits.find((v) => v.id === state.visit)?.label || '';
      canvas.setAttribute(
        'aria-label',
        'Cell coordinates in ' +
          state.embeddingLabel +
          ', ' +
          timestep +
          ', ' +
          population +
          (state.compare
            ? '. Visit lens active. Drag the lens or use arrow keys to move it. Tap elsewhere to position it.'
            : '. Drag or use arrow keys to pan.') +
          ' Plus and minus zoom, Home resets, square brackets select populations and Enter centers the selection.'
      );
    }
    invalidate();
  }
  function group(p, visit = state.visit) {
    return summaries.get([state.arm, visit, state.study.populations[p].id].join('|'));
  }
  function alphaMix() {
    const post = state.visit === state.study.visits[1]?.id ? 1 : 0;
    const t = Number.isFinite(state.progress) ? clamp(state.progress, 0, 1) : post;
    if (state.compare)
      return [
        { visit: post, alpha: 1, mask: -1 },
        { visit: 1 - post, alpha: 1, mask: 1 }
      ];
    const passes = [];
    if (t < 1) passes.push({ visit: 0, alpha: 1 - t });
    if (t > 0) passes.push({ visit: 1, alpha: t });
    return passes;
  }
  function drawAxes(g) {
    if (width < 230 || height < 180) return;
    const dx = (xDomain[1] - xDomain[0]) / (sx * fit * zoom),
      desired = Math.max(24, Math.min(52, width * 0.1)) * dx;
    if (!(desired > 0)) return;
    const power = Math.pow(10, Math.floor(Math.log10(desired)));
    const unit = (desired / power >= 5 ? 5 : desired / power >= 2 ? 2 : 1) * power,
      len = unit / dx;
    const y = height - 13;
    const x = [(width - len) / 2, width - len - 70, 15].find(
      (x) =>
        x >= 8 &&
        !reserved.some(
          (b) => x < b.x + b.w + 3 && x + Math.max(42, len) > b.x - 3 && y - 20 < b.y + b.h && y > b.y
        )
    );
    if (x === undefined) return;
    g.strokeStyle = 'rgba(137,178,190,.55)';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(x, y - 4);
    g.lineTo(x, y);
    g.lineTo(x + len, y);
    g.lineTo(x + len, y - 4);
    g.stroke();
    g.fillStyle = '#9db1bc';
    g.font = '9px ui-monospace, monospace';
    g.fillText(
      String(Number(unit.toPrecision(3))) + ' ' + (layout.coordinateUnit || state.embeddingLabel),
      x,
      y - 8
    );
  }
  function drawLabels(g, now) {
    hitLabels = [];
    const selected = state.study.populations.findIndex((p) => p.id === state.pop);
    const ranked = anchors
      .filter(Boolean)
      .filter((a) => (group(a.p)?.cells || 0) > 0)
      .sort(
        (a, b) =>
          (b.p === selected) - (a.p === selected) || (group(b.p)?.cells || 0) - (group(a.p)?.cells || 0)
      );
    const occupied = reserved.slice();
    const small = width < 570,
      limit = small ? 3 : width < 800 ? 5 : 7;
    for (const a of ranked) {
      if (hitLabels.length >= limit) break;
      const selectedHere = a.p === selected,
        pop = state.study.populations[a.p],
        m = group(a.p);
      const px = projectX(a.x),
        py = projectY(a.y);
      if (px < 3 || px > width - 3 || py < 3 || py > height - 3) continue;
      const color = pop.color || '#89c9d4';
      const signal = m?.signals?.[featureId()];
      const value =
        state.layer === 'composition'
          ? (m.fraction * 100).toFixed(1) + '%'
          : selectedHere && Number.isFinite(signal?.mean)
            ? 'mean ' + Intl.NumberFormat('en', { maximumSignificantDigits: 3 }).format(signal.mean)
            : '';
      const label = pop.label;
      const fontSize = small ? 10.5 : 11.5;
      g.font =
        (selectedHere ? '600 ' : '500 ') +
        fontSize +
        'px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';
      const nameW = g.measureText(label).width;
      g.font = '9px ui-monospace,monospace';
      const amountW = value ? g.measureText(value).width + 8 : 0;
      const bw = 15 + nameW + amountW,
        bh = small ? 20 : 22;
      const offsets = [
        [13, -bh - 12],
        [-bw - 13, -bh - 10],
        [14, 9],
        [-bw - 14, 9],
        [6, -bh - 30],
        [-bw + 4, 26],
        [-bw / 2, 31],
        [-bw / 2, -53]
      ];
      // Place labels in the side gutters of the fitted map when width permits.
      const pad = (width - fit * zoom * sx) / 2;
      if (pad > bw + 9) {
        const useLeft = px < width * 0.5;
        const rail = useLeft ? Math.max(12, pad - bw - 13) : Math.min(width - bw - 12, width - pad + 13);
        offsets.unshift(...[0, -26, 26, -52, 52].map((dy) => [rail - px, -bh / 2 + dy]));
      }
      const lens = state.compare ? lensCircle() : null;
      let candidate;
      for (const [ox, oy] of offsets) {
        const x = clamp(px + ox, 3, width - bw - 3),
          y = clamp(py + oy, 3, height - bh - 4);
        if (lens) {
          const nx = clamp(lens.x, x, x + bw),
            ny = clamp(lens.y, y, y + bh);
          if (Math.hypot(nx - lens.x, ny - lens.y) < lens.r + 7) continue;
        }
        if (
          !occupied.some(
            (b) => x < b.x + b.w + 5 && x + bw > b.x - 5 && y < b.y + b.h + 5 && y + bh > b.y - 5
          )
        ) {
          candidate = { x, y };
          break;
        }
      }
      if (!candidate) continue;
      const { x, y } = candidate;
      occupied.push({ x, y, w: bw, h: bh });
      hitLabels.push({ x, y, w: bw, h: bh, p: a.p, anchor: a });
      const nearX = clamp(px, x + 3, x + bw - 3),
        nearY = clamp(py, y + 3, y + bh - 3),
        edgeX = px < x ? x : px > x + bw ? x + bw : nearX;
      g.strokeStyle = selectedHere ? color : 'rgba(164,186,195,.65)';
      g.lineWidth = selectedHere ? 1.1 : 0.7;
      // Inside the lens the alternate visit is the only visible measurement.
      // Omit the outer-visit leader and marker when their anchor is covered.
      const anchorCovered = lens && Math.hypot(lens.x - px, lens.y - py) < lens.r + 4;
      if (!anchorCovered) {
        g.save();
        if (lens) {
          g.beginPath();
          g.rect(0, 0, width, height);
          g.arc(lens.x, lens.y, lens.r + 2, 0, 2 * Math.PI);
          g.clip('evenodd');
        }
        g.globalAlpha = selectedHere ? 0.72 : 0.52;
        g.beginPath();
        g.moveTo(px, py);
        if (Math.abs(edgeX - px) > 52) g.lineTo(edgeX + (px < edgeX ? -10 : 10), nearY);
        g.lineTo(edgeX, nearY);
        g.stroke();
        g.restore();
      }
      g.fillStyle = selectedHere ? 'rgba(6,19,28,.94)' : 'rgba(5,17,26,.80)';
      g.beginPath();
      g.roundRect(x + 0.5, y + 0.5, bw, bh, 3);
      g.fill();
      if (selectedHere) {
        g.strokeStyle = color;
        g.globalAlpha = 0.65;
        g.stroke();
        g.globalAlpha = 1;
      }
      g.fillStyle = color;
      g.fillRect(x + 4, y + bh * 0.28, 2, bh * 0.44);
      g.font =
        (selectedHere ? '600 ' : '500 ') +
        fontSize +
        'px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';
      g.fillStyle = selectedHere ? '#f3f7f6' : '#d6e4e8';
      g.textBaseline = 'middle';
      g.fillText(label, x + 9, y + bh / 2 + 0.5);
      if (value) {
        g.fillStyle = selectedHere ? '#c9d9e1' : '#a1b3bf';
        g.font = '9px ui-monospace,monospace';
        g.fillText(value, x + nameW + 16, y + bh / 2 + 0.5);
      }
      g.textBaseline = 'alphabetic';
      if (anchorCovered) continue;
      if (selectedHere) {
        const age = now - pulseStart,
          ar = 7 + (!state.reduced && pulseStart >= 0 && age < 380 ? 9 * (1 - age / 380) : 0);
        g.strokeStyle = color;
        g.lineWidth = 1.2;
        for (let i = 0; i < 4; i++) {
          g.beginPath();
          g.arc(px, py, ar, (i * Math.PI) / 2 + 0.18, (i * Math.PI) / 2 + 0.9);
          g.stroke();
        }
      } else {
        g.fillStyle = color;
        g.beginPath();
        g.arc(px, py, 1.9, 0, 2 * Math.PI);
        g.fill();
      }
    }
  }
  function drawLens(g, lens) {
    const { x, y, r } = lens;
    g.save();
    g.lineWidth = 4;
    g.strokeStyle = 'rgba(3,15,23,.84)';
    g.beginPath();
    g.arc(x, y, r + 1, 0, Math.PI * 2);
    g.stroke();
    g.lineWidth = 1.5;
    g.strokeStyle = '#9af0de';
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.stroke();
    g.strokeStyle = 'rgba(187,244,235,.64)';
    g.lineWidth = 1;
    for (let i = 0; i < 12; i++) {
      const a = (i * Math.PI) / 6,
        dx = Math.cos(a),
        dy = Math.sin(a),
        l = i % 3 === 0 ? 7 : 3;
      g.beginPath();
      g.moveTo(x + (r + 2) * dx, y + (r + 2) * dy);
      g.lineTo(x + (r + 2 + l) * dx, y + (r + 2 + l) * dy);
      g.stroke();
    }
    const outside = state.visit === state.study.visits[1].id ? 1 : 0;
    const label = 'LENS  /  ' + state.study.visits[1 - outside].label.toUpperCase();
    g.font = '600 10px ui-monospace,monospace';
    const lw = g.measureText(label).width + 16,
      lh = 20;
    const lx = clamp(x - lw / 2, 3, width - lw - 3);
    const choices = [y + r - lh / 2, y - r - lh / 2, y + r - lh - 16].map((pos) =>
      clamp(pos, 3, height - lh - 4)
    );
    const ly =
      choices.find(
        (pos) =>
          !reserved.some(
            (box) =>
              lx < box.x + box.w + 3 &&
              lx + lw > box.x - 3 &&
              pos < box.y + box.h + 3 &&
              pos + lh > box.y - 3
          )
      ) ?? choices[0];
    g.fillStyle = '#0b252b';
    g.strokeStyle = '#8ddccc';
    g.beginPath();
    g.roundRect(lx, ly, lw, lh, 4);
    g.fill();
    g.stroke();
    g.fillStyle = '#e0fff5';
    g.textBaseline = 'middle';
    g.fillText(label, lx + 8, ly + lh / 2 + 0.5);
    g.restore();
  }
  function render(now) {
    frameId = 0;
    hud.setTransform(ratio, 0, 0, ratio, 0, 0);
    hud.clearRect(0, 0, width, height);
    if (!state?.points || !raw) return;
    if (flight) {
      const t = clamp((now - flight.start) / 350, 0, 1),
        k = 1 - Math.pow(1 - t, 3);
      zoom = flight.z0 + (flight.z1 - flight.z0) * k;
      panX = flight.x0 + (flight.x1 - flight.x0) * k;
      panY = flight.y0 + (flight.y1 - flight.y0) * k;
      onCamera({ zoom, panX, panY });
      if (t >= 1) flight = null;
    }
    const size = clamp(2.75 * Math.pow(Math.min(width, height) / 520, 0.2) * Math.pow(zoom, 0.55), 1.9, 10);
    const mix = colorMix(now);
    if (mix >= 1) {
      colorStart = -1;
      previousLayers = null;
    }
    const lens = state.compare ? lensCircle() : null;
    renderer.draw({
      lens,
      mix,
      w: width,
      h: height,
      ratio,
      x: projectX(0),
      y: projectY(0),
      scale: fit * zoom,
      size,
      passes: alphaMix(),
      emphasis: state.layer === 'composition' ? 0.35 : 1
    });
    drawAxes(hud);
    drawLabels(hud, now);
    if (lens) drawLens(hud, lens);
    if (hover !== null && !state.compare) {
      const x = projectX(xy[hover * 2]),
        y = projectY(xy[hover * 2 + 1]);
      hud.strokeStyle = '#f7f5d9';
      hud.lineWidth = 1;
      hud.beginPath();
      hud.arc(x, y, 6.5, 0, 2 * Math.PI);
      hud.stroke();
      hud.beginPath();
      hud.moveTo(x + 9, y);
      hud.lineTo(x + 14, y);
      hud.moveTo(x - 9, y);
      hud.lineTo(x - 14, y);
      hud.stroke();
    }
    if (flight || mix < 1 || (!state.reduced && pulseStart >= 0 && now - pulseStart < 380)) invalidate();
  }
  function pointEvent(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  function hitAt(pos) {
    const label = hitLabels.find(
      (b) => pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h
    );
    if (label) return { p: label.p, row: null };
    const scale = fit * zoom,
      x = (pos.x - projectX(0)) / scale,
      y = (projectY(0) - pos.y) / scale;
    const near = 9 / scale,
      cellVisit = state.visit === state.study.visits[1]?.id ? 1 : 0;
    let dist = near * near,
      best = null;
    for (
      let gy = Math.max(0, Math.floor((y + 0.5 - near) * 72));
      gy <= Math.min(72, Math.floor((y + 0.5 + near) * 72));
      gy++
    ) {
      for (
        let gx = Math.max(0, Math.floor((x + 0.5 - near) * 72));
        gx <= Math.min(72, Math.floor((x + 0.5 + near) * 72));
        gx++
      ) {
        const rows = grid.get(gx + gy * 128);
        if (!rows) continue;
        for (const r of rows) {
          if (visits[r] !== cellVisit) continue;
          const dx = xy[r * 2] - x,
            dy = xy[r * 2 + 1] - y,
            d = dx * dx + dy * dy;
          if (d < dist) {
            dist = d;
            best = r;
          }
        }
      }
    }
    return best !== null ? { p: pops[best], row: best } : null;
  }
  function clearHover() {
    if (hover !== null) hover = null;
    onInspect(null);
  }
  function inspect(pos, hit) {
    if (!hit) {
      clearHover();
      invalidate();
      return;
    }
    const row = hit.row,
      p = state.study.populations[hit.p],
      f = layout.featureIds.indexOf(featureId()),
      feature = state.study.features.find((s) => s.id === featureId());
    const data = {
      label: p.label,
      population: p.id,
      x: pos.x,
      y: pos.y,
      visit: state.visit,
      arm: state.arm,
      row,
      kind: row === null ? 'population' : 'cell'
    };
    if (row !== null) {
      data.umap = [
        xDomain[0] + (xy[row * 2] / sx + 0.5) * (xDomain[1] - xDomain[0]),
        yDomain[0] + (xy[row * 2 + 1] / sy + 0.5) * (yDomain[1] - yDomain[0])
      ];
      data.arm = state.study.arms[arms[row]]?.id;
      if (state.layer !== 'composition' && f >= 0 && feature?.domain) {
        const q = view.getUint8(row * layout.stride + signalOffset + f);
        data.feature = feature.id;
        data.value = feature.domain[0] + (q / 255) * (feature.domain[1] - feature.domain[0]);
        data.quantized = true;
      }
    }
    hover = row;
    onInspect(data);
    invalidate();
  }
  function notifyCamera() {
    flight = null;
    colorStart = -1;
    previousLayers = null;
    onCamera({ zoom, panX, panY });
    invalidate();
  }
  function zoomBy(factor, at = { x: width / 2, y: height / 2 }) {
    clearHover();
    const z = clamp(zoom * factor, 0.65, 10),
      k = z / zoom;
    panX = (panX + width / 2 - at.x) * k + at.x - width / 2;
    panY = (panY + height / 2 - at.y) * k + at.y - height / 2;
    zoom = z;
    notifyCamera();
  }
  function reset() {
    zoom = 1;
    panX = 0;
    panY = 0;
    clearHover();
    notifyCamera();
  }
  listen(
    canvas,
    'wheel',
    (e) => {
      e.preventDefault();
      zoomBy(Math.exp(-clamp(e.deltaY * (e.deltaMode === 1 ? 14 : 1), -160, 160) * 0.002), pointEvent(e));
    },
    { passive: false }
  );
  function centerOn(wx, wy, targetZoom = zoom * 1.65) {
    clearHover();
    const z = clamp(targetZoom, 1.65, 10);
    const x = -wx * fit * z,
      y = wy * fit * z;
    if (state.reduced) {
      panX = x;
      panY = y;
      zoom = z;
      notifyCamera();
      return;
    }
    flight = { x0: panX, y0: panY, z0: zoom, x1: x, y1: y, z1: z, start: performance.now() };
    invalidate();
  }
  function focusPopulation() {
    if (!state?.points) return;
    const a = anchors[state.study.populations.findIndex((p) => p.id === state.pop)];
    if (a) centerOn(a.x, a.y, 2.1);
  }
  listen(canvas, 'dblclick', (e) => {
    if (!state?.points) return;
    e.preventDefault();
    const at = pointEvent(e);
    centerOn((at.x - projectX(0)) / (fit * zoom), (projectY(0) - at.y) / (fit * zoom));
  });
  listen(canvas, 'pointerdown', (e) => {
    if (e.button !== 0) return;
    flight = null;
    const p = pointEvent(e);
    pointers.set(e.pointerId, { ...p, ox: p.x, oy: p.y });
    canvas.setPointerCapture(e.pointerId);
    if (state?.compare && pointers.size === 1) {
      const l = lensCircle();
      if (Math.hypot(l.x - p.x, l.y - p.y) <= l.r + 10)
        lensDrag = { id: e.pointerId, ox: p.x - l.x, oy: p.y - l.y };
    }
    if (pointers.size === 2) {
      lensDrag = null;
      const [a, b] = pointers.values();
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      dragging = true;
    }
  });
  listen(canvas, 'pointermove', (e) => {
    if (!state?.points) return;
    const p = pointEvent(e),
      prev = pointers.get(e.pointerId);
    if (prev) {
      const next = { ...prev, ...p };
      pointers.set(e.pointerId, next);
      if (pointers.size === 2 && pinch) {
        const [a, b] = pointers.values(),
          x = (a.x + b.x) / 2,
          y = (a.y + b.y) / 2,
          d = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
        panX += x - pinch.x;
        panY += y - pinch.y;
        zoomBy(d / Math.max(1, pinch.d), { x, y });
        pinch = { d, x, y };
        return;
      }
      if (Math.hypot(p.x - prev.ox, p.y - prev.oy) > 3) dragging = true;
      if (lensDrag?.id === e.pointerId) {
        if (dragging) {
          moveLens(p.x - lensDrag.ox, p.y - lensDrag.oy);
          canvas.style.cursor = 'grabbing';
        }
        return;
      }
      if (dragging) {
        panX += p.x - prev.x;
        panY += p.y - prev.y;
        clearHover();
        canvas.style.cursor = 'grabbing';
        notifyCamera();
      }
    } else if (e.pointerType !== 'touch') {
      if (state.compare) {
        const l = lensCircle();
        canvas.style.cursor = Math.hypot(l.x - p.x, l.y - p.y) < l.r ? 'move' : 'crosshair';
        return;
      }
      const hit = hitAt(p);
      canvas.style.cursor = hit ? 'crosshair' : 'grab';
      inspect(p, hit);
    }
  });
  function endPointer(e) {
    const p = pointEvent(e),
      wasDown = pointers.has(e.pointerId);
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (e.type === 'pointerup' && wasDown && !dragging && state?.points) {
      if (state.compare) moveLens(p.x, p.y);
      else {
        const hit = hitAt(p);
        if (hit) {
          onSelect(state.study.populations[hit.p].id);
          inspect(p, hit);
        }
      }
    }
    if (lensDrag?.id === e.pointerId) lensDrag = null;
    if (!pointers.size) {
      dragging = false;
      canvas.style.cursor = 'grab';
    }
  }
  listen(canvas, 'pointerup', endPointer);
  listen(canvas, 'pointercancel', endPointer);
  listen(canvas, 'pointerleave', () => {
    if (!pointers.size) {
      clearHover();
      invalidate();
    }
  });
  listen(canvas, 'keydown', (e) => {
    if (!state?.points) return;
    const d = 40;
    if (e.key === '+' || e.key === '=') zoomBy(1.3);
    else if (e.key === '-') zoomBy(1 / 1.3);
    else if (e.key === 'Home' || e.key === '0') reset();
    else if (e.key === 'Enter') focusPopulation();
    else if (e.key === '[' || e.key === ']') {
      const ps = state.study.populations,
        i = ps.findIndex((p) => p.id === state.pop);
      onSelect(ps[(i + (e.key === ']' ? 1 : -1) + ps.length) % ps.length].id);
    } else if (e.key.startsWith('Arrow')) {
      if (state.compare) {
        const l = lensCircle(),
          step = e.shiftKey ? 32 : 14;
        moveLens(
          l.x + (e.key === 'ArrowRight' ? step : e.key === 'ArrowLeft' ? -step : 0),
          l.y + (e.key === 'ArrowDown' ? step : e.key === 'ArrowUp' ? -step : 0)
        );
      } else {
        panX += e.key === 'ArrowLeft' ? d : e.key === 'ArrowRight' ? -d : 0;
        panY += e.key === 'ArrowUp' ? d : e.key === 'ArrowDown' ? -d : 0;
        clearHover();
        notifyCamera();
      }
    } else return;
    e.preventDefault();
    e.stopPropagation();
  });
  listen(canvas, 'webglcontextlost', (e) => {
    e.preventDefault();
    if (surface !== canvas) return;
    surface = document.createElement('canvas');
    Object.assign(surface.style, { position: 'absolute', pointerEvents: 'none', display: 'block' });
    canvas.insertAdjacentElement('afterend', surface);
    renderer = softwareRenderer(surface);
    surface.width = canvas.width;
    surface.height = canvas.height;
    renderer.upload(layers);
    measure();
  });
  return {
    setState,
    zoomBy,
    reset,
    focusPopulation,
    destroy() {
      disposed = true;
      abort.abort();
      watcher.disconnect();
      cancelAnimationFrame(frameId);
      renderer.destroy();
      overlay.remove();
      if (surface !== canvas) surface.remove();
    }
  };
}
