import h5wasm from './vendor/hdf5_hl.js';

let mountSerial = 0;
let library;
function hdf5Library() {
  if (!library) {
    library = h5wasm.ready
      .then(async (module) => {
        module.activate_throwing_error_handler();
        const response = await fetch(new URL('./vendor/libH5Zlzf.wasm', import.meta.url));
        if (!response.ok) throw Error('The local HDF5 compression library could not load.');
        const dir = module.get_plugin_search_paths()[0];
        module.FS.mkdirTree(dir);
        module.FS.writeFile(dir + '/libH5Zlzf.so', new Uint8Array(await response.arrayBuffer()));
        return module;
      })
      .catch((error) => {
        library = null;
        throw error;
      });
  }
  return library;
}
const numeric = (dataset) => dataset?.type === 'Dataset' && [0, 1].includes(dataset.metadata.type);
const attribute = (node, key) => node?.attrs[key]?.value;
const optional = (group, name) => (group?.keys().includes(name) ? group.get(name) : null);
const dimension = (value) => {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n < 0) throw Error('Invalid H5AD dimension or sparse matrix index.');
  return n;
};

// A local CLI session supplies an authenticated, loopback-only range endpoint.
// A dropped File uses WORKERFS directly. Neither path materializes the whole file.
function rangeReader(source) {
  const endpoint = new URL(source.url, self.location.href);
  if (
    endpoint.origin !== self.location.origin ||
    endpoint.pathname !== '/local/h5ad' ||
    endpoint.search ||
    endpoint.hash ||
    !source.session
  )
    throw Error('Invalid local H5AD endpoint.');
  const blockSize = 1024 * 1024,
    cache = new Map();
  const size = dimension(source.size);
  function block(start) {
    let bytes = cache.get(start);
    if (bytes) cache.delete(start);
    else {
      const last = Math.min(size, start + blockSize) - 1;
      const xhr = new XMLHttpRequest();
      xhr.open('GET', endpoint.href, false);
      xhr.responseType = 'arraybuffer';
      xhr.setRequestHeader('Range', 'bytes=' + start + '-' + last);
      xhr.setRequestHeader('X-sctrial-session', source.session);
      xhr.send();
      if (
        xhr.status !== 206 ||
        xhr.getResponseHeader('Content-Range') !== 'bytes ' + start + '-' + last + '/' + size ||
        xhr.response?.byteLength !== last - start + 1
      ) {
        throw Error(
          'The local launcher could not read this H5AD byte range. Reopen the study in its launcher.'
        );
      }
      bytes = new Uint8Array(xhr.response);
    }
    cache.set(start, bytes);
    if (cache.size > 24) cache.delete(cache.keys().next().value);
    return bytes;
  }
  return (_stream, buffer, offset, length, position) => {
    const available = Math.min(length, Math.max(0, size - position));
    for (let done = 0; done < available; ) {
      const pos = position + done,
        start = Math.floor(pos / blockSize) * blockSize;
      const bytes = block(start),
        within = pos - start;
      const count = Math.min(available - done, bytes.length - within);
      buffer.set(bytes.subarray(within, within + count), offset + done);
      done += count;
    }
    return available;
  };
}

function vectorLength(node) {
  if (!node) return null;
  if (node.type === 'Dataset') return node.shape?.length === 1 ? node.shape[0] : null;
  const encoding = attribute(node, 'encoding-type');
  if (encoding === 'categorical') return vectorLength(optional(node, 'codes'));
  if (encoding?.startsWith('nullable-')) return vectorLength(optional(node, 'values'));
  return null;
}

function describeColumn(node) {
  const encoding = attribute(node, 'encoding-type');
  if (encoding === 'categorical') {
    const count = vectorLength(node.get('categories'));
    return {
      kind: 'categorical',
      ...(count !== null && count <= 256 ? { levels: columnValues(node.get('categories')) } : {})
    };
  }
  if (encoding?.startsWith('nullable-')) return describeColumn(node.get('values'));
  if (node.metadata?.type === 3) return { kind: 'string' };
  if (numeric(node)) return { kind: 'numeric' };
  return { kind: 'other' };
}

function textValue(value) {
  if (value === null || value === undefined || (typeof value === 'number' && !Number.isFinite(value)))
    return null;
  return String(value);
}

function columnValues(node, legacyCategories) {
  const encoding = attribute(node, 'encoding-type');
  if (encoding === 'categorical' || legacyCategories) {
    const cats = columnValues(legacyCategories || node.get('categories'));
    const codes = (legacyCategories ? node : node.get('codes')).value;
    return Array.from(codes, (code) => {
      if (Number(code) === -1) return null;
      const index = dimension(code);
      if (index >= cats.length) throw Error('Categorical annotation has a code outside its categories.');
      return cats[index];
    });
  }
  if (encoding?.startsWith('nullable-')) {
    const items = columnValues(node.get('values')),
      mask = node.get('mask').value;
    if (items.length !== mask.length)
      throw Error('The nullable annotation has an invalid missing-value mask.');
    return items.map((value, i) => (mask[i] ? null : value));
  }
  if (node.type !== 'Dataset' || node.shape?.length !== 1)
    throw Error('Expected a one-dimensional H5AD annotation.');
  // Read strings in bounded batches: variable-length HDF5 strings need native
  // memory for each value in addition to the JS strings returned to the caller.
  const length = node.shape[0],
    result = new Array(length);
  for (let i = 0; i < length; i += 16384) {
    const data = node.slice([[i, Math.min(i + 16384, length)]]);
    for (let j = 0; j < data.length; j++) result[i + j] = textValue(data[j]);
  }
  return result;
}

function frameIndex(group) {
  const indexName = attribute(group, '_index') ?? '_index';
  return optional(group, indexName) || optional(group, 'index');
}

function describeFrame(frame, length) {
  const legacyGroup = optional(frame, '__categories');
  const names = frame.keys().filter((k) => k !== '__categories');
  return names.flatMap((name) => {
    const node = frame.get(name);
    if (vectorLength(node) !== length) return [];
    const legacy = optional(legacyGroup, name);
    const desc = legacy ? { kind: 'categorical' } : describeColumn(node);
    return [{ name, ...desc }];
  });
}

function matrixLayout(node) {
  if (node?.type === 'Dataset' && numeric(node) && node.shape?.length === 2) {
    return { node, encoding: 'dense', shape: node.shape.map(dimension) };
  }
  if (node?.type !== 'Group') return null;
  const fmt = attribute(node, 'encoding-type') || attribute(node, 'h5sparse_format');
  const encoding = { csr: 'csr', csc: 'csc', csr_matrix: 'csr', csc_matrix: 'csc' }[fmt];
  if (!encoding) return null;
  const shape = Array.from(attribute(node, 'shape') ?? attribute(node, 'h5sparse_shape') ?? [], dimension);
  if (shape.length !== 2) throw Error('Invalid sparse H5AD matrix dimensions.');
  return { node, encoding, shape };
}

function sparseArrays(layout) {
  const { node, shape, encoding } = layout;
  const pointers = node.get('indptr').value,
    indices = node.get('indices'),
    data = node.get('data');
  const major = encoding === 'csr' ? shape[0] : shape[1];
  if (
    !numeric(data) ||
    !numeric(indices) ||
    data.shape.length !== 1 ||
    indices.shape.length !== 1 ||
    pointers.length !== major + 1 ||
    Number(pointers[0]) !== 0 ||
    Number(pointers[major]) !== data.shape[0] ||
    data.shape[0] !== indices.shape[0]
  ) {
    throw Error('Sparse matrix dimensions disagree with the index arrays.');
  }
  for (let i = 1; i < pointers.length; i++) {
    if (dimension(pointers[i]) < Number(pointers[i - 1]))
      throw Error('Invalid sparse matrix row or column pointers.');
  }
  return { pointers, indices, data };
}

function checkScalar(value, counts) {
  if (!Number.isFinite(value)) throw Error('The selected expression data contain a non-finite value.');
  if (counts && (!Number.isSafeInteger(value) || value < 0)) {
    throw Error(
      'Counts normalization requires nonnegative integer counts. Check the matrix and its expression scale.'
    );
  }
  return value;
}

// A CSC file stores each gene contiguously. Keep one bounded read cursor per
// requested gene so output blocks do not repeatedly decompress entire columns.
// Scipy also permits unsorted indices. Those columns use a full bounded scan
// for each block to preserve every entry, including duplicate coordinates.
function cscReaders(sparse, targets, nObs) {
  const batch = Math.max(128, Math.min(8192, Math.floor(524288 / targets.size)));
  return Array.from(targets, ([gene, column]) => {
    const start = Number(sparse.pointers[gene]),
      stop = Number(sparse.pointers[gene + 1]);
    let sorted = true,
      previous = -1;
    for (let a = start; a < stop; a += 1048576) {
      const ix = sparse.indices.slice([[a, Math.min(stop, a + 1048576)]]);
      for (const value of ix) {
        const row = dimension(value);
        if (row >= nObs) throw Error('A CSC row index is outside the matrix.');
        if (row < previous) sorted = false;
        previous = row;
      }
    }
    let cursor = start,
      index = 0,
      indices = [],
      values = [];
    return (lastRow, write) => {
      if (!sorted) {
        cursor = start;
        index = 0;
        indices = [];
      }
      while (cursor < stop || index < indices.length) {
        if (index === indices.length) {
          const next = Math.min(stop, cursor + batch);
          indices = sparse.indices.slice([[cursor, next]]);
          values = sparse.data.slice([[cursor, next]]);
          index = 0;
          cursor = next;
        }
        const row = Number(indices[index]);
        if (sorted && row > lastRow) return;
        write(row, column, Number(values[index]));
        index++;
      }
    };
  });
}

async function extractMatrix(layout, { indices: genes, rows, scale = 'stored', onBlock, onProgress }) {
  const [nObs, nVars] = layout.shape;
  if (!['counts', 'stored'].includes(scale)) throw Error('Choose the expression scale for this matrix.');
  if (!genes || !genes.length) throw Error('Choose at least one measured gene.');
  const n = rows ? rows.length : nObs,
    k = genes.length,
    counts = scale === 'counts';
  const targets = new Map();
  for (let j = 0; j < k; j++) {
    const gene = dimension(genes[j]);
    if (gene >= nVars || targets.has(gene))
      throw Error('Gene indices must be unique and within the selected expression matrix.');
    targets.set(gene, j);
  }
  if (rows) {
    let previous = -1;
    for (const row of rows) {
      if (dimension(row) >= nObs || row <= previous)
        throw Error('Selected matrix rows must be unique and sorted.');
      previous = row;
    }
  }
  const rowAt = (i) => (rows ? rows[i] : i);
  const output = onBlock ? null : new Float64Array(n * k);
  const sparse = layout.encoding === 'dense' ? null : sparseArrays(layout);
  const dest = new Int32Array(nObs).fill(-1);
  for (let i = 0; i < n; i++) dest[rowAt(i)] = i;

  // CSC count normalization needs every gene for each selected cell, including
  // genes outside the panel. Scan the sparse data in bounded contiguous reads.
  let cscTotals;
  if (counts && layout.encoding === 'csc') {
    cscTotals = new Float64Array(n);
    const total = sparse.data.shape[0];
    for (let a = 0; a < total; a += 1048576) {
      const b = Math.min(total, a + 1048576);
      const ix = sparse.indices.slice([[a, b]]),
        values = sparse.data.slice([[a, b]]);
      for (let t = 0; t < ix.length; t++) {
        const r = Number(ix[t]);
        if (!Number.isSafeInteger(r) || r < 0 || r >= nObs)
          throw Error('A CSC row index is outside the matrix.');
        const out = dest[r];
        if (out >= 0) cscTotals[out] += checkScalar(Number(values[t]), true);
      }
    }
  }
  const majorReaders = layout.encoding === 'csc' ? cscReaders(sparse, targets, nObs) : null;
  const maxElements = 1048576;
  const blockRows = Math.min(
    4096,
    Math.max(1, Math.floor(maxElements / (layout.encoding === 'dense' ? Math.max(k, nVars) : k)))
  );
  for (let offset = 0; offset < n; offset += blockRows) {
    const end = Math.min(n, offset + blockRows),
      length = end - offset;
    const result = new Float64Array(length * k);
    const totals = counts && !cscTotals ? new Float64Array(length) : null;
    if (majorReaders) {
      const last = rowAt(end - 1);
      const write = (row, column, value) => {
        const j = dest[row];
        if (j >= offset && j < end) result[(j - offset) * k + column] += checkScalar(value, counts);
      };
      for (const read of majorReaders) read(last, write);
    } else {
      // Keep each physical read bounded even when the filtered rows are far apart.
      let at = offset;
      while (at < end) {
        const first = rowAt(at);
        let stop = at + 1;
        while (stop < end) {
          const next = rowAt(stop),
            span = next - first + 1;
          const entries = sparse
            ? Number(sparse.pointers[next + 1]) - Number(sparse.pointers[first])
            : span * nVars;
          if (entries > maxElements || span > 4096) break;
          stop++;
        }
        const last = rowAt(stop - 1) + 1;
        if (sparse) {
          const startPtr = Number(sparse.pointers[first]),
            endPtr = Number(sparse.pointers[last]);
          if (endPtr > startPtr) {
            const ix = sparse.indices.slice([[startPtr, endPtr]]),
              val = sparse.data.slice([[startPtr, endPtr]]);
            for (let i = at; i < stop; i++) {
              const r = rowAt(i),
                j = i - offset;
              for (
                let t = Number(sparse.pointers[r]) - startPtr,
                  lim = Number(sparse.pointers[r + 1]) - startPtr;
                t < lim;
                t++
              ) {
                const g = Number(ix[t]);
                if (!Number.isSafeInteger(g) || g < 0 || g >= nVars)
                  throw Error('A CSR column index is outside the matrix.');
                const c = targets.get(g);
                if (counts || c !== undefined) {
                  const v = checkScalar(Number(val[t]), counts);
                  if (counts) totals[j] += v;
                  if (c !== undefined) result[j * k + c] += v;
                }
              }
            }
          }
        } else {
          const data = layout.node.slice([
            [first, last],
            [0, nVars]
          ]);
          for (let i = at; i < stop; i++) {
            const base = (rowAt(i) - first) * nVars,
              j = i - offset;
            if (counts)
              for (let g = 0; g < nVars; g++) totals[j] += checkScalar(Number(data[base + g]), true);
            for (const [g, c] of targets) result[j * k + c] = checkScalar(Number(data[base + g]), counts);
          }
        }
        at = stop;
      }
    }
    if (counts) {
      for (let j = 0; j < length; j++) {
        const sum = cscTotals ? cscTotals[offset + j] : totals[j];
        if (!Number.isSafeInteger(sum))
          throw Error('A cell library size exceeds precise integer representation.');
        if (sum > 0)
          for (let c = 0; c < k; c++) result[j * k + c] = Math.log1p((10000 * result[j * k + c]) / sum);
      }
    }
    if (onBlock) await onBlock({ offset, values: result, rowCount: length });
    else output.set(result, offset * k);
    if (onProgress) onProgress(end, n);
  }
  return output ?? undefined;
}

export async function openH5ad(source) {
  const module = await hdf5Library();
  const fs = module.FS,
    directory = '/study-' + ++mountSerial;
  const isFile = source instanceof Blob;
  const backendRead = isFile ? null : rangeReader(source);
  fs.mkdir(directory);
  let handle,
    mounted = false;
  function close() {
    try {
      handle?.close();
    } finally {
      handle = null;
      if (mounted) {
        fs.unmount(directory);
        mounted = false;
        fs.rmdir(directory);
      }
    }
  }
  try {
    // The internal mount name is fixed. No study filename becomes a filesystem path.
    const root = fs.mount(
      fs.filesystems.WORKERFS,
      { blobs: [{ name: 'input.h5ad', data: isFile ? source : { size: dimension(source.size) } }] },
      directory
    );
    mounted = true;
    if (backendRead) {
      const node = root.contents['input.h5ad'];
      node.stream_ops = { ...node.stream_ops, read: backendRead };
    }
    handle = new h5wasm.File(directory + '/input.h5ad', 'r');
    const obs = optional(handle, 'obs'),
      varFrame = optional(handle, 'var');
    if (obs?.type !== 'Group' || varFrame?.type !== 'Group')
      throw Error('Choose an AnnData H5AD with obs and var dataframes.');
    const nObs = vectorLength(frameIndex(obs));
    if (nObs === null || nObs === 0) throw Error('The H5AD observation index is missing or empty.');
    const obsColumns = describeFrame(obs, nObs);
    const matrices = new Map(),
      frames = new Map();
    function register(id, node, frame) {
      if (!node || !frame) return;
      const layout = matrixLayout(node);
      if (!layout || layout.shape[0] !== nObs || vectorLength(frameIndex(frame)) !== layout.shape[1])
        return;
      if (!frames.has(frame.path))
        frames.set(
          frame.path,
          describeFrame(frame, layout.shape[1])
            .filter((f) => ['string', 'categorical'].includes(f.kind))
            .map((f) => f.name)
        );
      matrices.set(id, {
        layout,
        frame,
        meta: { id, shape: layout.shape, encoding: layout.encoding, geneColumns: frames.get(frame.path) }
      });
    }
    register('X', optional(handle, 'X'), varFrame);
    const layers = optional(handle, 'layers');
    for (const key of layers?.keys() ?? []) register('layers/' + key, layers.get(key), varFrame);
    const raw = optional(handle, 'raw');
    if (raw?.type === 'Group') register('raw/X', optional(raw, 'X'), optional(raw, 'var'));
    const obsm = optional(handle, 'obsm'),
      maps = new Map();
    for (const key of obsm?.keys() ?? []) {
      const node = obsm.get(key);
      const layout = matrixLayout(node);
      if (layout?.shape[0] === nObs && layout.shape[1] >= 2) maps.set(key, { layout });
      else if (
        node.type === 'Group' &&
        attribute(node, 'encoding-type') === 'dataframe' &&
        vectorLength(frameIndex(node)) === nObs
      ) {
        const cols = Array.from(attribute(node, 'column-order') ?? []);
        if (cols.length >= 2 && cols.slice(0, 2).every((c) => numeric(node.get(c))))
          maps.set(key, { node, cols: cols.slice(0, 2) });
      }
    }
    const schema = {
      name: source.name,
      nObs,
      obsColumns,
      embeddings: [...maps.keys()],
      layers: [...matrices.values()].map((s) => s.meta)
    };
    const legacy = optional(obs, '__categories');
    const readColumn = (name) => {
      if (!obsColumns.some((c) => c.name === name)) throw Error('Choose an available observation column.');
      return columnValues(obs.get(name), optional(legacy, name));
    };
    return {
      schema,
      readColumn,
      close,
      genes(id, symbolColumn = '') {
        const m = matrices.get(id);
        if (!m || (symbolColumn && !m.meta.geneColumns.includes(symbolColumn)))
          throw Error('Choose an available matrix and gene-symbol column.');
        const node = symbolColumn ? m.frame.get(symbolColumn) : frameIndex(m.frame);
        return columnValues(
          node,
          symbolColumn ? optional(optional(m.frame, '__categories'), symbolColumn) : null
        ).map((v) => v ?? '');
      },
      async readCells({ columns, embedding }) {
        const annotation = Object.fromEntries(
          Array.from(new Set(columns), (name) => [name, readColumn(name)])
        );
        if (embedding == null) return { columns: annotation, xy: null };
        const info = maps.get(embedding);
        if (!info) throw Error('Choose a stored embedding with at least two coordinates per observation.');
        let xy;
        if (info.node) {
          const x = info.node.get(info.cols[0]).value,
            y = info.node.get(info.cols[1]).value;
          xy = new Float64Array(nObs * 2);
          for (let i = 0; i < nObs; i++) {
            xy[i * 2] = Number(x[i]);
            xy[i * 2 + 1] = Number(y[i]);
          }
        } else if (info.layout.encoding === 'dense') {
          xy = Float64Array.from(
            info.layout.node.slice([
              [0, nObs],
              [0, 2]
            ]),
            Number
          );
        } else xy = await extractMatrix(info.layout, { indices: [0, 1], scale: 'stored' });
        return { columns: annotation, xy };
      },
      readExpression({ layer, ...options }) {
        const m = matrices.get(layer);
        if (!m) throw Error('Choose an available expression matrix.');
        return extractMatrix(m.layout, options);
      }
    };
  } catch (error) {
    close();
    throw error;
  }
}
