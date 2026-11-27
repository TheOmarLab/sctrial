const binsPerFeature = 32;
const visits = ['pre', 'post'];
const normalizationUnits = {
  counts: 'log1p(CP10K)',
  log1p: 'stored log1p expression',
  as_is: 'stored expression'
};

const pathwayNames = [
  ['ifn_ab', 'Interferon alpha/beta signaling', 'R-HSA-909733'],
  ['tcr', 'TCR signaling', 'R-HSA-202403'],
  ['mhc_ii', 'MHC class II antigen presentation', 'R-HSA-2132295']
];
let pathwayCache;

export async function listPathways() {
  if (!pathwayCache) {
    // The reference study and uploaded studies use the same frozen Enrichr Reactome_2022 GMT.
    const response = await fetch(new URL('./data/reactome-2022.gmt', import.meta.url));
    if (!response.ok)
      throw Error('Could not load the bundled Reactome gene sets. Reload the atlas to retry.');
    const lines = (await response.text()).trim().split(/\r?\n/);
    const byAccession = new Map(
      lines.map((line) => {
        const [name, , ...members] = line.split('\t');
        return [name.split(' ').pop(), [...new Set(members.filter(Boolean))]];
      })
    );
    pathwayCache = pathwayNames.map(([id, label, accession]) => {
      const members = byAccession.get(accession);
      if (!members?.length) throw Error('The bundled Reactome gene set is incomplete: ' + accession);
      return { id, label, accession, members };
    });
  }
  return pathwayCache.map((set) => ({ ...set, members: [...set.members] }));
}

const present = (v) => v !== null && v !== undefined && String(v).trim() !== '';
const normalizedLabel = (v) => (present(v) ? String(v) : null);
const unique = (values) => [...new Set(values)];

function validateMapping(schema, config) {
  const fields = [config.population, config.visit, config.arm, config.participant].filter(Boolean);
  if (!config.population || !config.visit) throw Error('Map the population and visit columns.');
  if (new Set(fields).size !== fields.length)
    throw Error('Use separate columns for population, visit, arm and participant.');
  const names = new Set(schema.obsColumns.map((c) => c.name));
  for (const field of fields) {
    if (!names.has(field)) throw Error('Observation column not found: ' + field);
  }
  if (
    !present(config.baseline) ||
    !present(config.followup) ||
    String(config.baseline) === String(config.followup)
  ) {
    throw Error('Choose two different observed visits.');
  }
  if (config.embedding) {
    if (!schema.embeddings.includes(config.embedding))
      throw Error('The selected two-dimensional embedding is unavailable.');
  } else if (
    !Array.isArray(config.axisGenes) ||
    config.axisGenes.length !== 2 ||
    !config.axisGenes.every(present) ||
    config.axisGenes[0] === config.axisGenes[1]
  ) {
    throw Error(
      'Select an embedding or two different measured genes for the horizontal and vertical axes.'
    );
  }
  return fields;
}

function regions(labels, counts, prefix, missingLabel) {
  const index = new Map();
  if (counts.has(null)) {
    while (counts.has(missingLabel)) missingLabel += ' (missing)';
  }
  const entries = [...counts.keys()].sort((a, b) => {
    if (prefix === 'p' && counts.get(a) !== counts.get(b)) return counts.get(b) - counts.get(a);
    return String(a ?? missingLabel).localeCompare(String(b ?? missingLabel));
  });
  for (const key of entries) {
    const i = labels.length;
    index.set(key, i);
    labels.push({ id: prefix + i, label: key ?? missingLabel });
  }
  return index;
}

function selectCells(schema, config, cellData) {
  const { columns, xy } = cellData;
  const visitColumn = columns[config.visit];
  const popColumn = columns[config.population];
  const armColumn = config.arm ? columns[config.arm] : null;
  const personColumn = config.participant ? columns[config.participant] : null;
  const before = String(config.baseline),
    after = String(config.followup);
  const popCounts = new Map(),
    armCounts = new Map(),
    participantIds = new Set(),
    rows = [];
  const audit = {
    missingVisit: 0,
    otherVisits: 0,
    missingPopulation: 0,
    missingArm: 0,
    missingParticipant: 0
  };
  const nVisit = [0, 0];
  if (config.embedding && xy?.length !== 2 * schema.nObs)
    throw Error('The embedding does not contain two coordinates for every observation.');
  for (let row = 0; row < schema.nObs; row++) {
    const label = normalizedLabel(visitColumn[row]);
    if (label !== before && label !== after) {
      if (label === null) audit.missingVisit++;
      else audit.otherVisits++;
      continue;
    }
    rows.push(row);
    nVisit[Number(label === after)]++;
    const pop = normalizedLabel(popColumn[row]);
    if (pop === null) audit.missingPopulation++;
    popCounts.set(pop, (popCounts.get(pop) || 0) + 1);
    const arm = armColumn ? normalizedLabel(armColumn[row]) : null;
    if (armColumn) {
      if (arm === null) audit.missingArm++;
      armCounts.set(arm, (armCounts.get(arm) || 0) + 1);
    }
    if (personColumn) {
      const person = normalizedLabel(personColumn[row]);
      if (person === null) audit.missingParticipant++;
      else participantIds.add(person);
    }
  }
  if (!nVisit[0] || !nVisit[1]) throw Error('Each selected visit must contain at least one observation.');
  const populations = [],
    arms = [{ id: 'all', label: armColumn ? 'All arms' : 'All cells' }];
  const popMap = regions(populations, popCounts, 'p', 'Missing population label');
  const armMap = regions(arms, armCounts, 'a', 'Missing arm label');
  const n = rows.length,
    popCodes = new Uint32Array(n),
    armCodes = new Uint32Array(n),
    visitCodes = new Uint8Array(n);
  const coordinates = new Float64Array(n * 2);
  const people = personColumn ? new Array(n) : null;
  for (let i = 0; i < n; i++) {
    const row = rows[i];
    popCodes[i] = popMap.get(normalizedLabel(popColumn[row]));
    if (armColumn) armCodes[i] = armMap.get(normalizedLabel(armColumn[row]));
    visitCodes[i] = Number(String(visitColumn[row]) === after);
    if (people) people[i] = normalizedLabel(personColumn[row]);
    if (config.embedding) {
      const x = xy[2 * row],
        y = xy[2 * row + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw Error(
          'The selected embedding has non-finite coordinates at the chosen visits. Select another embedding or use gene axes.'
        );
      }
      coordinates[2 * i] = x;
      coordinates[2 * i + 1] = y;
    }
  }
  return {
    rows,
    popCodes,
    armCodes,
    visitCodes,
    coordinates,
    people,
    participantIds,
    populations,
    arms,
    audit,
    scopeIndices: arms.map((_, arm) => (arm ? [0, arm] : [0]))
  };
}

async function requestedSets(config) {
  const builtIns = config.pathways?.length ? await listPathways() : [];
  const byId = new Map(builtIns.map((set) => [set.id, set]));
  const selected = unique(config.pathways || []).map((id) => {
    if (!byId.has(id)) throw Error('Unknown Reactome gene set: ' + id);
    return byId.get(id);
  });
  for (const [i, set] of (config.customPathways || []).entries()) {
    if (!present(set.label) || !Array.isArray(set.members) || !set.members.length) {
      throw Error('Each custom pathway needs a name and a list of measured gene identifiers.');
    }
    selected.push({ id: 'custom' + i, label: String(set.label), members: unique(set.members.map(String)) });
  }
  return selected;
}

async function panelFor(reader, config, warnings) {
  const chosen = unique(config.genes || []),
    sets = await requestedSets(config);
  const axes = config.embedding ? [] : config.axisGenes;
  if (!chosen.length && !sets.length && !axes.length)
    return { features: [], indices: [], axisCols: [], featureCols: [] };
  const unit = normalizationUnits[config.normalization];
  if (!unit) throw Error('Specify the measurement scale of the expression matrix.');
  const names = await reader.genes(config.layer, config.symbolColumn || '');
  const index = new Map();
  for (let i = 0; i < names.length; i++) index.set(names[i], index.has(names[i]) ? -1 : i);
  function requireGene(gene) {
    if (!index.has(gene)) throw Error('Gene not found in the selected feature column: ' + gene);
    if (index.get(gene) < 0)
      throw Error('The identifier ' + gene + ' occurs more than once. Choose a unique feature column.');
  }
  for (const gene of [...chosen, ...axes]) requireGene(gene);
  const features = chosen.map((label, i) => ({ id: 'g' + i, label, kind: 'gene', unit }));
  const featureGenes = chosen.map((g) => [g]);
  for (const set of sets) {
    const members = unique(set.members.filter((g) => index.has(g) && index.get(g) >= 0));
    const ambiguous = set.members.filter((g) => index.get(g) === -1);
    if (ambiguous.length)
      warnings.push(set.label + ': omitted ' + ambiguous.length + ' ambiguous member identifiers.');
    if (!members.length) {
      warnings.push(set.label + ' has no uniquely matched members in the selected matrix.');
      continue;
    }
    const feature = {
      id: set.id,
      label: set.label,
      kind: 'pathway',
      unit: 'Mean member ' + unit,
      memberUnit: unit,
      members,
      memberCount: members.length,
      sourceMemberCount: set.members.length,
      source: set.accession ? 'Human Reactome_2022 (Enrichr)' : 'User-provided gene set'
    };
    if (set.accession) {
      feature.accession = set.accession;
      feature.url = 'https://reactome.org/content/detail/' + set.accession;
    }
    features.push(feature);
    featureGenes.push(members);
  }
  const panel = unique([...axes, ...featureGenes.flat()]);
  const panelIndex = new Map(panel.map((gene, i) => [gene, i]));
  return {
    features,
    indices: panel.map((g) => index.get(g)),
    axisCols: axes.map((g) => panelIndex.get(g)),
    featureCols: featureGenes.map((genes) => genes.map((g) => panelIndex.get(g)))
  };
}

function makeCapture() {
  return { cells: 0, people: new Map(), populations: new Map() };
}

function makePopulation(nFeatures, nColumns) {
  return {
    cells: 0,
    people: new Map(),
    sums: new Float64Array(nFeatures),
    zeros: new Uint32Array(nFeatures),
    bins: new Uint32Array(nFeatures * binsPerFeature),
    columnSums: new Float64Array(nColumns)
  };
}

function countCells(cells, panel) {
  const captures = cells.arms.map(() => [makeCapture(), makeCapture()]);
  for (let i = 0; i < cells.rows.length; i++) {
    const arm = cells.armCodes[i],
      visit = cells.visitCodes[i],
      pop = cells.popCodes[i],
      person = cells.people?.[i];
    for (const scope of cells.scopeIndices[arm]) {
      const cap = captures[scope][visit];
      cap.cells++;
      if (!cap.populations.has(pop))
        cap.populations.set(pop, makePopulation(panel.features.length, panel.indices.length));
      const group = cap.populations.get(pop);
      group.cells++;
      if (person != null) {
        cap.people.set(person, (cap.people.get(person) || 0) + 1);
        if (!group.people.has(person))
          group.people.set(person, { cells: 0, sums: new Float64Array(panel.features.length) });
        group.people.get(person).cells++;
      }
    }
  }
  return captures;
}

function recordExpression(cells, panel, captures, signals, maxima, block) {
  const { values, offset, rowCount } = block;
  const { features, indices, featureCols, axisCols } = panel;
  const nCols = indices.length,
    nFeatures = features.length;
  for (let row = 0; row < rowCount; row++) {
    const i = offset + row,
      start = row * nCols,
      signalStart = i * nFeatures;
    for (let j = 0; j < nCols; j++) {
      if (!Number.isFinite(values[start + j]) || values[start + j] < 0) {
        throw Error(
          'Select a nonnegative expression layer. Centered or residual matrices do not support these expression and pathway displays.'
        );
      }
    }
    if (axisCols.length) {
      cells.coordinates[2 * i] = values[start + axisCols[0]];
      cells.coordinates[2 * i + 1] = values[start + axisCols[1]];
    }
    for (let f = 0; f < nFeatures; f++) {
      const columns = featureCols[f];
      let value = 0;
      for (const j of columns) value += values[start + j];
      value /= columns.length;
      signals[signalStart + f] = value;
      if (value > maxima[f]) maxima[f] = value;
    }
    const arm = cells.armCodes[i],
      visit = cells.visitCodes[i],
      pop = cells.popCodes[i],
      person = cells.people?.[i];
    for (const scope of cells.scopeIndices[arm]) {
      const g = captures[scope][visit].populations.get(pop);
      const individual = person != null ? g.people.get(person) : null;
      for (let f = 0; f < nFeatures; f++) {
        const value = signals[signalStart + f];
        g.sums[f] += value;
        if (individual) individual.sums[f] += value;
        if (value === 0) g.zeros[f]++;
      }
      for (let j = 0; j < nCols; j++) g.columnSums[j] += values[start + j];
    }
  }
}

function coordinateDomains(xy) {
  let xmin = Infinity,
    xmax = -Infinity,
    ymin = Infinity,
    ymax = -Infinity;
  for (let i = 0; i < xy.length; i += 2) {
    const x = xy[i],
      y = xy[i + 1];
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  return [
    [xmin, xmax],
    [ymin, ymax]
  ];
}

function encodePoints(config, cells, panel, captures, signals, maxima) {
  const [xDomain, yDomain] = coordinateDomains(cells.coordinates);
  const dx = xDomain[1] - xDomain[0],
    dy = yDomain[1] - yDomain[0];
  if (!config.embedding && (dx === 0 || dy === 0)) {
    throw Error(
      'A selected gene axis is constant across the two visits. Choose two genes with variation or use a stored embedding.'
    );
  }
  const maxCategory = Math.max(cells.arms.length, cells.populations.length) - 1;
  const width = maxCategory <= 255 ? 1 : maxCategory <= 65535 ? 2 : 4;
  const nFeatures = panel.features.length,
    valueOffset = 5 + 2 * width,
    stride = valueOffset + nFeatures;
  const points = new ArrayBuffer(cells.rows.length * stride);
  const view = new DataView(points);
  const writeCategory = width === 1 ? 'setUint8' : width === 2 ? 'setUint16' : 'setUint32';
  for (let i = 0; i < cells.rows.length; i++) {
    const pos = i * stride,
      arm = cells.armCodes[i],
      visit = cells.visitCodes[i],
      pop = cells.popCodes[i];
    view.setUint16(
      pos,
      dx ? Math.round((65535 * (cells.coordinates[2 * i] - xDomain[0])) / dx) : 32768,
      true
    );
    view.setUint16(
      pos + 2,
      dy ? Math.round((65535 * (cells.coordinates[2 * i + 1] - yDomain[0])) / dy) : 32768,
      true
    );
    view[writeCategory](pos + 4, pop, true);
    view[writeCategory](pos + 4 + width, arm, true);
    view.setUint8(pos + 4 + 2 * width, visit);
    const pooled = captures[0][visit].populations.get(pop);
    const stratum = arm ? captures[arm][visit].populations.get(pop) : null;
    for (let f = 0; f < nFeatures; f++) {
      const value = signals[i * nFeatures + f],
        fraction = maxima[f] ? value / maxima[f] : 0;
      view.setUint8(pos + valueOffset + f, Math.round(255 * fraction));
      const bin = f * binsPerFeature + Math.min(binsPerFeature - 1, Math.floor(binsPerFeature * fraction));
      pooled.bins[bin]++;
      if (stratum) stratum.bins[bin]++;
    }
  }
  const map = {
    rows: cells.rows.length,
    stride,
    categoryBytes: width,
    featureIds: panel.features.map((f) => f.id),
    xDomain,
    yDomain
  };
  return { map, points };
}

function pairedFractions(list, caps, pop) {
  return list.map((person) => {
    const pair = {};
    for (const [i, key] of visits.entries()) {
      const totalCells = caps[i].people.get(person);
      const cells = caps[i].populations.get(pop)?.people.get(person)?.cells || 0;
      pair[key] = { cells, totalCells, fraction: cells / totalCells };
    }
    return pair;
  });
}

function summarizeGroups(cells, panel, captures) {
  const groups = [];
  const paired = [];
  for (const [arm, caps] of captures.entries()) {
    const matches = [...caps[0].people.keys()].filter((id) => caps[1].people.has(id));
    paired.push(matches.length);
    for (let pop = 0; pop < cells.populations.length; pop++) {
      const pairs = pairedFractions(matches, caps, pop);
      for (const [v, visit] of visits.entries()) {
        const capture = caps[v],
          group = capture.populations.get(pop),
          n = group?.cells || 0;
        const signals = {};
        for (const [f, feature] of panel.features.entries()) {
          let sumOfIndividualMeans = 0;
          if (group) for (const p of group.people.values()) sumOfIndividualMeans += p.sums[f] / p.cells;
          const signal = {
            mean: n ? group.sums[f] / n : null,
            participantMean: group?.people.size ? sumOfIndividualMeans / group.people.size : null,
            zeros: group?.zeros[f] || 0,
            bins: group
              ? Array.from(group.bins.subarray(f * binsPerFeature, (f + 1) * binsPerFeature))
              : Array(binsPerFeature).fill(0)
          };
          if (feature.kind === 'pathway') {
            signal.memberMeans = panel.featureCols[f].map((c) => (n ? group.columnSums[c] / n : null));
          }
          signals[feature.id] = signal;
        }
        groups.push({
          arm: cells.arms[arm].id,
          visit,
          population: cells.populations[pop].id,
          cells: n,
          totalCells: capture.cells,
          fraction: capture.cells ? n / capture.cells : null,
          participants: cells.people ? group?.people.size || 0 : null,
          totalParticipants: cells.people ? capture.people.size : null,
          participantPairs: pairs,
          signals
        });
      }
    }
  }
  return { groups, paired };
}

function embeddingName(key) {
  if (!key) return 'Gene space';
  const base = key.split('/').pop().replace(/^X_/, '');
  return /^(umap|pca|tsne)$/i.test(base) ? base.toUpperCase() : base;
}

export async function buildAtlas(reader, config, onProgress = () => {}) {
  const fields = validateMapping(reader.schema, config);
  onProgress('Reading observation annotations and coordinates…');
  const cellData = await reader.readCells({ columns: fields, embedding: config.embedding || null });
  const cells = selectCells(reader.schema, config, cellData);
  const warnings = [];
  const panel = await panelFor(reader, config, warnings);
  const captures = countCells(cells, panel),
    n = cells.rows.length,
    k = panel.features.length;
  const signals = new Float64Array(n * k),
    maxima = new Float64Array(k);
  if (panel.indices.length) {
    let read = 0,
      lastReported = -1;
    await reader.readExpression({
      layer: config.layer,
      indices: panel.indices,
      rows: cells.rows,
      scale: config.normalization === 'counts' ? 'counts' : 'stored',
      onBlock(block) {
        if (block.offset !== read || block.values.length !== block.rowCount * panel.indices.length) {
          throw Error('The expression matrix row order does not match the selected observations.');
        }
        recordExpression(cells, panel, captures, signals, maxima, block);
        read += block.rowCount;
        const percent = Math.floor((100 * read) / n);
        if (percent !== lastReported) {
          onProgress('Reading measured expression · ' + percent + '%');
          lastReported = percent;
        }
      }
    });
    if (read !== n) throw Error('The expression matrix is incomplete for the selected observations.');
  }
  panel.features.forEach((f, i) => {
    f.domain = [0, maxima[i]];
  });
  onProgress('Preparing cell coordinates and visit comparisons…');
  const { map, points } = encodePoints(config, cells, panel, captures, signals, maxima);
  const { groups, paired } = summarizeGroups(cells, panel, captures);
  const label = embeddingName(config.embedding);
  map.axisLabels = config.embedding ? [label + ' 1', label + ' 2'] : [...config.axisGenes];
  const unit = normalizationUnits[config.normalization];
  if (!config.embedding) map.coordinateUnit = unit;
  const study = {
    title: String(config.title || reader.schema.name || 'Uploaded H5AD').replace(/\.h5ad$/i, ''),
    uploaded: true,
    cells: n,
    sourceRows: reader.schema.nObs,
    participants: cells.people ? cells.participantIds.size : null,
    samples: null,
    embeddingLabel: label,
    embedding: config.embedding
      ? 'The map uses the first two coordinates of ' + config.embedding + ' stored in the H5AD.'
      : 'The map shows measured ' +
        config.axisGenes[0] +
        ' and ' +
        config.axisGenes[1] +
        ' expression in ' +
        unit +
        '.',
    expression: !panel.indices.length
      ? 'No expression features were selected.'
      : config.normalization === 'counts'
        ? 'Each cell was normalized to 10,000 counts using all features in the selected matrix, followed by log1p transformation.'
        : 'The atlas uses the values stored in the selected matrix in units of ' + unit + '.',
    annotation:
      'Population labels come from ' +
      config.population +
      '. All cells at the two selected visits contribute to pooled fractions.',
    pathway:
      'Each cell’s pathway score is the arithmetic mean across its uniquely matched gene-set members, including zero expression.',
    mapping: {
      population: config.population,
      visit: config.visit,
      baseline: String(config.baseline),
      followup: String(config.followup),
      arm: config.arm || null,
      participant: config.participant || null,
      embedding: config.embedding || null,
      axisGenes: config.embedding ? null : [...config.axisGenes],
      layer: config.layer || null,
      symbolColumn: config.symbolColumn || null,
      normalization: config.normalization || null
    },
    pairedParticipants: cells.people ? paired[0] : null,
    unpairedParticipants: cells.people ? cells.participantIds.size - paired[0] : null,
    audit: cells.audit,
    warnings,
    histogramBins: binsPerFeature
  };
  return {
    study: {
      study,
      arms: cells.arms,
      visits: [
        { id: 'pre', label: String(config.baseline) },
        { id: 'post', label: String(config.followup) }
      ],
      populations: cells.populations,
      features: panel.features,
      groups,
      map
    },
    points
  };
}
