(() => {
  'use strict';

  // Procedural illustrations. Geometry and pigments are independent of study observations.
  const TAU = 2 * Math.PI;
  const PAD = 1.09;
  const MAP_W = 384;
  const MAP_H = 192;
  const FRAME_COUNT = 96;
  const MAX_FRAME_PIXELS = 5_400_000;
  const tissues = new Map();
  const colors = new Map();
  const frames = new Map();
  const projectionGrids = new Map();
  let cachedPixels = 0;

  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const smooth = (x) => x * x * (3 - 2 * x);

  function hash(name, seed) {
    let h = (Number(seed) | 0) ^ 2166136261;
    for (let i = 0; i < name.length; i++) h = Math.imul(h ^ name.charCodeAt(i), 16777619);
    return h >>> 0;
  }

  function random(seed) {
    let state = seed;
    return () => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return state / 4294967296;
    };
  }

  // Seeded lattice noise in object coordinates. Sampling a sphere in 3D avoids polar seams.
  function noiseField(next) {
    const side = 32;
    const grid = Uint8Array.from({ length: side ** 3 }, () => Math.floor(next() * 256));
    return (x, y, z) => {
      const ix = Math.floor(x),
        iy = Math.floor(y),
        iz = Math.floor(z);
      const fx = smooth(x - ix),
        fy = smooth(y - iy),
        fz = smooth(z - iz);
      const x0 = ix & 31,
        x1 = (ix + 1) & 31;
      const y0 = (iy & 31) * 32,
        y1 = ((iy + 1) & 31) * 32;
      const z0 = (iz & 31) * 1024,
        z1 = ((iz + 1) & 31) * 1024;
      const a = grid[x0 + y0 + z0] * (1 - fx) + grid[x1 + y0 + z0] * fx;
      const b = grid[x0 + y1 + z0] * (1 - fx) + grid[x1 + y1 + z0] * fx;
      const c = grid[x0 + y0 + z1] * (1 - fx) + grid[x1 + y0 + z1] * fx;
      const d = grid[x0 + y1 + z1] * (1 - fx) + grid[x1 + y1 + z1] * fx;
      return ((a * (1 - fy) + b * fy) * (1 - fz) + (c * (1 - fy) + d * fy) * fz) / 255;
    };
  }

  function surface(kind, seed) {
    const next = random(hash(kind, seed));
    const noise = noiseField(next);
    const bCell = /(^|[^a-z])b(?:[ -]?(?:lymph|cell)|$)/i.test(kind);
    const myeloid = /monocyt|myeloid|macrophag/i.test(kind);
    const nk = /\bNK\b|natural killer/i.test(kind);
    const frequency = bCell ? 3.9 : myeloid ? 3.5 : 4.6 + next() * 0.4;
    const count = MAP_W * MAP_H;
    const height = new Float32Array(count);
    const normals = new Int8Array(count * 3);
    const grain = new Uint8Array(count);
    const pigment = new Uint8Array(count);
    const chromatin = new Uint8Array(count);
    const center = [
      next() * 0.07 - (bCell ? 0.07 : 0.18),
      next() * 0.09 + 0.015,
      0.025 + next() * 0.11,
    ];
    const phi = Array.from({ length: MAP_W }, (_, i) => ((i + 0.5) / MAP_W) * TAU - Math.PI);
    const sin = phi.map(Math.sin),
      cos = phi.map(Math.cos);
    const granules = [];
    for (let n = 0; n < (nk ? 18 : myeloid ? 55 : 37); n++) {
      const y = next() * 2 - 1,
        p = next() * TAU,
        r = 0.68 + next() * 0.17;
      const xy = Math.sqrt(1 - y * y);
      granules.push([
        r * xy * Math.sin(p),
        r * y,
        r * xy * Math.cos(p),
        nk ? 0.032 + next() * 0.039 : 0.013 + next() ** 2 * 0.029,
      ]);
    }

    for (let j = 0; j < MAP_H; j++) {
      const lat = ((j + 0.5) / MAP_H - 0.5) * Math.PI;
      const cy = Math.cos(lat),
        y = Math.sin(lat);
      for (let i = 0; i < MAP_W; i++) {
        const k = j * MAP_W + i,
          x = cy * sin[i],
          z = cy * cos[i];
        const wx = noise(2.3 * x + 9, 2.3 * y, 2.3 * z) - 0.5;
        const wy = noise(2 * x, 2 * y + 13, 2 * z) - 0.5;
        const wz = noise(2.2 * x, 2.2 * y, 2.2 * z + 18) - 0.5;
        const u = x * frequency + wx * 2.1,
          v = y * frequency + wy * 2.1,
          w = z * frequency + wz * 2.1;
        const coarse = noise(u, v, w);
        const warp = noise(u * 1.9 + 8, v * 1.9, w * 1.9 + 6);
        const field = coarse + 0.13 * (warp - 0.5);
        const foldDistance = Math.abs(field - 0.49);
        const ridge = Math.exp(-115 * foldDistance * foldDistance);
        const ripple = noise(x * 22 + 12 * wx, y * 18 + 1, z * 21 + 4 * wy);
        const fine = noise(x * 45 + 3, y * 42, z * 44);
        const offset = ripple - 0.52;
        const filament = Math.exp(-88 * offset * offset);
        const papilla = smooth(clamp((ripple - 0.42) * 2.9));
        height[k] =
          0.08 * wx +
          0.054 * wy +
          0.038 * (coarse - 0.5) +
          (bCell ? 0.039 : 0.084) * ridge * (0.32 + 0.68 * warp) +
          (bCell ? 0.042 * papilla + 0.008 * filament : 0.016 * filament) +
          0.009 * (fine - 0.5);
        grain[k] = (fine * 0.34 + ripple * 0.18 + warp * 0.29 + ridge * 0.19) * 255;
        chromatin[k] =
          255 *
          (0.4 * noise(5.8 * x + 7, 5.8 * y + 9, 5.8 * z) +
            0.38 * noise(11.7 * x, 11.7 * y + 2, 11.7 * z + 7) +
            0.22 * fine);
        pigment[k] =
          clamp(0.12 + 0.46 * coarse + 0.29 * ripple + 0.15 * noise(10 * x, 10 * y, 10 * z)) * 255;
      }
    }

    // Differentiate the displacement on its spherical tangent plane.
    for (let j = 0; j < MAP_H; j++) {
      const lat = ((j + 0.5) / MAP_H - 0.5) * Math.PI;
      const c = Math.cos(lat),
        s = Math.sin(lat);
      const northRow = Math.min(j + 1, MAP_H - 1) * MAP_W;
      const southRow = Math.max(j - 1, 0) * MAP_W;
      for (let i = 0; i < MAP_W; i++) {
        const k = j * MAP_W + i;
        const du =
          ((height[j * MAP_W + ((i + 1) % MAP_W)] - height[j * MAP_W + ((i + MAP_W - 1) % MAP_W)]) *
            MAP_W) /
          (2 * TAU * Math.max(0.07, c));
        const dv = ((height[northRow + i] - height[southRow + i]) * MAP_H) / (2 * Math.PI);
        const nx = c * sin[i] - du * cos[i] + dv * s * sin[i];
        const ny = s - dv * c;
        const nz = c * cos[i] + du * sin[i] + dv * s * cos[i];
        const scale = 127 / Math.sqrt(nx * nx + ny * ny + nz * nz);
        normals[3 * k] = Math.round(nx * scale);
        normals[3 * k + 1] = Math.round(ny * scale);
        normals[3 * k + 2] = Math.round(nz * scale);
      }
    }
    return {
      height,
      normals,
      grain,
      pigment,
      chromatin,
      center,
      granules,
      myeloid,
      nucleus: bCell ? 0.68 : myeloid ? 0.66 : nk ? 0.54 : 0.63,
    };
  }

  function hsl(h, saturation, lightness) {
    h = (((h % 360) + 360) % 360) / 360;
    const a = saturation * Math.min(lightness, 1 - lightness);
    return [0, 8, 4].map((v) => {
      const k = (v + h * 12) % 12;
      return 255 * (lightness - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)));
    });
  }

  function palette(hue) {
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      const deep = hue > 120 && hue < 225 ? 24 : hue < 70 ? -18 : -10;
      const shift = hue > 225 && hue < 310 ? -20 : hue < 70 ? 12 : -7;
      const h = hue + deep * (1 - t) ** 2 + shift * t * t;
      const c = hsl(h, 0.81 - 0.22 * smooth(t), 0.034 + 0.84 * t ** 1.14);
      out.set(c, 3 * i);
    }
    return out;
  }

  function bitmap(size) {
    if (typeof OffscreenCanvas === 'function') {
      try {
        const canvas = new OffscreenCanvas(size, size);
        const ctx = canvas.getContext('2d');
        if (ctx) return { canvas, ctx };
      } catch {
        // Some browsers expose the constructor while disabling its 2D context.
      }
    }
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    return { canvas, ctx: canvas.getContext('2d') };
  }

  function mapPoint(x, y, z) {
    const lon = Math.atan2(x, z) / TAU + 0.5;
    const lat = Math.asin(clamp(y, -1, 1)) / Math.PI + 0.5;
    const i = Math.floor(lon * MAP_W) % MAP_W;
    return Math.min(MAP_H - 1, Math.floor(lat * MAP_H)) * MAP_W + i;
  }

  function projectionGrid(size) {
    if (projectionGrids.has(size)) return projectionGrids.get(size);
    const radial = new Float32Array(size * size);
    const azimuth = new Uint16Array(size * size);
    const coordinate = Float32Array.from(
      { length: size },
      (_, i) => ((i + 0.5 - size / 2) * 2 * PAD) / size,
    );
    for (let j = 0; j < size; j++) {
      const y = -coordinate[j];
      for (let i = 0; i < size; i++) {
        const x = coordinate[i],
          pixel = j * size + i;
        radial[pixel] = Math.sqrt(x * x + y * y);
        const angle = Math.atan2(y, x) / TAU;
        azimuth[pixel] = Math.round(((angle + 1) % 1) * 65535);
      }
    }
    const grid = { radial, azimuth, coordinate };
    if (projectionGrids.size >= 3) projectionGrids.delete(projectionGrids.keys().next().value);
    projectionGrids.set(size, grid);
    return grid;
  }

  function project(tissue, hue, size, angle) {
    const { canvas, ctx } = bitmap(size);
    if (!ctx) return null;
    const img = ctx.createImageData(size, size),
      pixels = img.data;
    const { height, normals, grain, pigment, chromatin, center, granules, nucleus } = tissue;
    const grid = projectionGrid(size);
    const ca = Math.cos(angle),
      sa = Math.sin(angle);
    const scale = size / (2 * PAD),
      middle = size / 2;
    const ramp = colors.get(hue) || palette(hue);
    colors.set(hue, ramp);
    const innerHue = (hue + (hue < 90 ? 350 : hue < 215 ? 13 : 347)) % 360;
    const innerRamp = colors.get(innerHue) || palette(innerHue);
    colors.set(innerHue, innerRamp);
    const ncx = center[0] * ca + center[2] * sa,
      ncy = center[1],
      ncz = center[2] * ca - center[0] * sa;

    // Sample the displaced silhouette, including relief just in front of the limb.
    const outline = new Float32Array(384);
    for (let i = 0; i < outline.length; i++) {
      const dir = (i / outline.length) * TAU,
        px = Math.cos(dir),
        py = Math.sin(dir);
      let rim = 0.9;
      for (const depth of [0, 0.1, 0.2, 0.28]) {
        const lateral = Math.sqrt(1 - depth * depth);
        const ox = px * lateral * ca - depth * sa,
          oz = px * lateral * sa + depth * ca;
        const k = mapPoint(ox, py * lateral, oz);
        rim = Math.max(rim, (0.975 + height[k]) * lateral);
      }
      outline[i] = rim;
    }

    for (let j = 0; j < size; j++) {
      const y = -grid.coordinate[j];
      for (let i = 0; i < size; i++) {
        const offset = j * size + i;
        const radial = grid.radial[offset];
        if (radial > PAD) continue;
        const x = grid.coordinate[i];
        const arc = grid.azimuth[offset] * (outline.length / 65535);
        const az = Math.floor(arc),
          mix = arc - az;
        const radius =
          outline[az % outline.length] * (1 - mix) + outline[(az + 1) % outline.length] * mix;
        const alpha = clamp((radius - radial) * scale + 0.5);
        if (alpha <= 0) continue;
        const radial3 = Math.min(1, radial * radial * radial);
        const radiusMix = 0.993 + (radius - 0.993) * smooth(radial3);
        const sx = x / radiusMix,
          sy = y / radiusMix,
          sz = Math.sqrt(Math.max(0, 1 - sx * sx - sy * sy));
        // Interpolate the membrane normals before lighting, including across the texture seam.
        const longitude = ((Math.atan2(sx, sz) - angle) / TAU + 1.5) * MAP_W - 0.5;
        const latitude = clamp(
          (Math.asin(clamp(sy, -1, 1)) / Math.PI + 0.5) * MAP_H - 0.5,
          0,
          MAP_H - 1.001,
        );
        const ix = Math.floor(longitude),
          iy = Math.floor(latitude),
          fx = longitude - ix,
          fy = latitude - iy;
        const row = iy * MAP_W,
          left = (ix + MAP_W) % MAP_W,
          right = (left + 1) % MAP_W;
        const k = row + left,
          k1 = row + right,
          k2 = k + MAP_W,
          k3 = k1 + MAP_W;
        const a = (1 - fx) * (1 - fy),
          b = fx * (1 - fy),
          c = (1 - fx) * fy,
          d = fx * fy;
        const ux =
          a * normals[3 * k] + b * normals[3 * k1] + c * normals[3 * k2] + d * normals[3 * k3];
        const uy =
          a * normals[3 * k + 1] +
          b * normals[3 * k1 + 1] +
          c * normals[3 * k2 + 1] +
          d * normals[3 * k3 + 1];
        const uz =
          a * normals[3 * k + 2] +
          b * normals[3 * k1 + 2] +
          c * normals[3 * k2 + 2] +
          d * normals[3 * k3 + 2];
        const norm = 1 / Math.sqrt(ux * ux + uy * uy + uz * uz);
        const nx = (ux * ca + uz * sa) * norm,
          ny = uy * norm,
          nz = (uz * ca - ux * sa) * norm;
        const diffuse = Math.max(0, -0.59 * nx + 0.66 * ny + 0.46 * nz);
        const largeLight = clamp(0.44 - 0.5 * sx + 0.48 * sy + 0.13 * sz);
        const spec = Math.max(0, -0.32 * nx + 0.38 * ny + 0.868 * nz);
        const grit = (a * grain[k] + b * grain[k1] + c * grain[k2] + d * grain[k3]) / 255;
        const dye = (a * pigment[k] + b * pigment[k1] + c * pigment[k2] + d * pigment[k3]) / 255;
        const fresnel = (1 - sz) ** 2;
        const rim = clamp((0.73 * sy - 0.68 * sx - 0.34) / 0.66);
        const soft = 0.13 + 0.21 * largeLight + 0.24 * diffuse + 0.085 * grit + 0.09 * dye;
        const spec2 = spec * spec,
          spec4 = spec2 * spec2,
          spec8 = spec4 * spec4,
          spec16 = spec8 * spec8;
        const gloss = 0.095 * spec8 + 0.13 * spec16 * spec16 * spec4 * spec2;
        const distanceToEdge = radius - radial;
        const thinRim = clamp(1 - distanceToEdge * 55);
        const brightness =
          soft + gloss + fresnel * (0.065 + 0.41 * rim) + thinRim * thinRim * (0.055 + 0.17 * rim);
        const p = 3 * Math.round(clamp(brightness) * 255);
        let red = ramp[p],
          green = ramp[p + 1],
          blue = ramp[p + 2];
        // View the denser nuclear material through an optically thinner part of the cortex.
        const xx = (x - ncx) / nucleus,
          yy = (y - ncy) / (nucleus * 0.92);
        const notch = tissue.myeloid
          ? 1.1 * Math.exp(-5 * (xx - 0.72) * (xx - 0.72) - 6 * yy * yy)
          : 0;
        const nd =
          xx * xx * (1 + 0.12 * yy) + yy * yy + 0.22 * (dye - 0.5) + 0.11 * xx * yy + notch;
        if (nd < 1.16) {
          const inside = Math.min(1, Math.max(0, 1 - nd));
          const front = Math.sqrt(inside);
          const depth = Math.max(0, sz - ncz - front * nucleus);
          const u = clamp(xx, -1, 1),
            v = clamp(yy, -1, 1);
          const nk = mapPoint(u * ca - front * sa, v, u * sa + front * ca);
          const condensed = chromatin[nk] / 255;
          const clump = smooth(clamp((condensed - 0.32) * 2.7));
          const window = smooth(clamp((1.16 - nd) * 3.5));
          const cortex =
            window *
            (0.79 + 0.15 * (1 - grit)) *
            (1 - 0.25 * smooth(clamp((diffuse - 0.26) * 1.8))) *
            clamp(1.32 - 0.62 * depth);
          const innerLight = clamp(0.11 - 0.48 * u + 0.47 * v + 0.24 * front);
          const value = 0.16 + 0.19 * innerLight + 0.115 * (1 - clump) + 0.07 * grit + 0.08 * front;
          const q = 3 * Math.round(clamp(value) * 255);
          const envelope = Math.exp(-120 * (nd - 0.98) * (nd - 0.98)) * (0.08 + 0.15 * largeLight);
          red = red * (1 - cortex) + innerRamp[q] * cortex;
          green = green * (1 - cortex) + innerRamp[q + 1] * cortex;
          blue = blue * (1 - cortex) + innerRamp[q + 2] * cortex;
          red += (ramp[Math.min(255 * 3, p + 66)] - red) * envelope;
          green += (ramp[Math.min(255 * 3, p + 66) + 1] - green) * envelope;
          blue += (ramp[Math.min(255 * 3, p + 66) + 2] - blue) * envelope;
        }
        const kpx = 4 * offset;
        const white = clamp(
          0.22 * spec16 * spec8 * spec2 * spec + 0.23 * fresnel * (0.2 + 0.8 * rim),
        );
        pixels[kpx] = red + (242 - red) * white;
        pixels[kpx + 1] = green + (249 - green) * white;
        pixels[kpx + 2] = blue + (255 - blue) * white;
        pixels[kpx + 3] = Math.round(255 * alpha);
      }
    }
    ctx.putImageData(img, 0, 0);

    // Small, submerged organelles rotate with the material. The overlying cortex mutes them.
    ctx.save();
    ctx.globalCompositeOperation = 'source-atop';
    const [gr, gg, gb] = hsl(hue + 15, 0.7, 0.28).map(Math.round);
    for (const [gx, gy, gz, gradius] of granules) {
      const z = gz * ca - gx * sa;
      if (z < 0.08) continue;
      const x = (gx * ca + gz * sa) * scale + middle,
        y = middle - gy * scale,
        r = gradius * scale;
      const overlay = ctx.createRadialGradient(x - r * 0.26, y - r * 0.35, r * 0.04, x, y, r);
      overlay.addColorStop(0, 'rgba(218,248,239,' + (0.26 + 0.32 * z) + ')');
      overlay.addColorStop(
        0.37,
        'rgba(' + gr + ',' + gg + ',' + gb + ',' + (0.32 + 0.14 * z) + ')',
      );
      overlay.addColorStop(0.74, 'rgba(9,15,30,' + (0.27 + 0.08 * z) + ')');
      overlay.addColorStop(1, 'rgba(9,15,30,0)');
      ctx.fillStyle = overlay;
      ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    }
    ctx.restore();
    return canvas;
  }

  function paint(ctx, { x, y, radius, kind = '', hue = 173, rotation = 0, seed = 1, detail = 1 }) {
    if (
      !ctx ||
      !Number.isFinite(radius) ||
      radius <= 0 ||
      !Number.isFinite(x) ||
      !Number.isFinite(y)
    )
      return;
    const transform = ctx.getTransform?.();
    const density = transform ? Math.hypot(transform.a, transform.b) : 1;
    // Larger portraits already cover the baked membrane map at native sampling density.
    const sampling = clamp(density, 0.7, 1.08 + 0.48 * clamp((160 - radius) / 100));
    const target = radius * 2 * PAD * sampling * clamp(detail, 0.5, 1.35);
    const size =
      [64, 96, 128, 160, 208, 256, 320, 384, 464, 544, 640].find((n) => n >= target) || 640;
    const frame =
      ((Math.round(((Number(rotation) || 0) * FRAME_COUNT) / TAU) % FRAME_COUNT) + FRAME_COUNT) %
      FRAME_COUNT;
    const tissueKey = JSON.stringify([String(kind), Number(seed) | 0]);
    hue = ((Math.round(Number(hue) || 0) % 360) + 360) % 360;
    const key = tissueKey + '|' + hue + '|' + size + '|' + frame;
    let canvas = frames.get(key);
    if (canvas) {
      frames.delete(key);
      frames.set(key, canvas);
    } else {
      let tissue = tissues.get(tissueKey);
      if (!tissue) {
        if (tissues.size >= 8) tissues.delete(tissues.keys().next().value);
        tissue = surface(String(kind), Number(seed) | 0);
        tissues.set(tissueKey, tissue);
      }
      canvas = project(tissue, hue, size, (frame / FRAME_COUNT) * TAU);
      if (!canvas) return;
      cachedPixels += size * size;
      frames.set(key, canvas);
      while (cachedPixels > MAX_FRAME_PIXELS && frames.size > 1) {
        const oldest = frames.keys().next().value;
        const dropped = frames.get(oldest);
        cachedPixels -= dropped.width * dropped.height;
        frames.delete(oldest);
      }
    }
    const extent = radius * PAD;
    ctx.drawImage(canvas, x - extent, y - extent, 2 * extent, 2 * extent);
  }

  window.SctrialCellArt = Object.freeze({ paint });
})();
