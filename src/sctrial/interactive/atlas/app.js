import { setupImporter } from './importer.js';
import { createScene } from './scene.js';
import { renderEvidence, selectEvidenceGene } from './evidence.js';
import { createPathwayScene } from './pathway-scene.js';

const icons = {
  settings:
    '<path d="M3 6h9m4 0h5M3 12h3m4 0h11M3 18h11m4 0h3"/><circle cx="14" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="16" cy="18" r="2"/>',
  lens: '<circle cx="10" cy="10" r="6.5"/><path d="m15 15 6 6M7 10h6m-3-3v6"/>',
  locate:
    '<circle cx="12" cy="12" r="6.5"/><circle cx="12" cy="12" r="1.5"/><path d="M12 1v4m0 14v4M1 12h4m14 0h4"/>',
  book: '<path d="M3 4.5h5a6 6 0 0 1 4 1.6 6 6 0 0 1 4-1.6h5v14h-5a6 6 0 0 0-4 1.5 6 6 0 0 0-4-1.5H3zM12 6v14"/>',
  upRight: '<path d="M6 18 18 6M7 6h11v11"/>',
  dots: '<circle cx="8" cy="6" r="2.2"/><circle cx="17" cy="10" r="2.2"/><circle cx="9" cy="17" r="3"/><path d="m3 11 1 1m14 6 1 1"/>',
  wave: '<path d="M2 13h4l3-8 4 14 3-9 2 3h4"/>',
  molecule:
    '<circle cx="5" cy="13" r="2.5"/><circle cx="13" cy="5" r="2.5"/><circle cx="19" cy="17" r="2.5"/><path d="m7 11 4-4m3 0 4 7M8 14l8 2"/>',
  branch:
    '<circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="6" r="2"/><path d="M6 7v10m0-4c7 0 12 0 12-5"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  plus: '<path d="M12 4v16M4 12h16"/>',
  minus: '<path d="M4 12h16"/>',
  fit: '<path d="M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5"/><circle cx="12" cy="12" r="2"/>',
  expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 3l6 6m12-6-6 6M3 21l6-6m12 6-6-6"/>',
  compress: '<path d="M3 9h6V3m6 0v6h6M9 21v-6H3m18 0h-6v6"/>',
  play: '<path d="m7 4 13 8-13 8z"/>',
  pause: '<path d="M7 5h3v14H7zM14 5h3v14h-3z"/>',
  layers: '<path d="m2 8 10-5 10 5-10 5zM2 12l10 5 10-5M2 16l10 5 10-5"/>',
  target:
    '<circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="1.5"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/>',
  list: '<path d="M9 6h12M9 12h12M9 18h12M3 6h1M3 12h1M3 18h1"/>',
  close: '<path d="m5 5 14 14M19 5 5 19"/>',
  folder: '<path d="M3 7V5h6l3 3h8v3M2 10h20l-3 9H4z"/>'
};
const $ = (id) => document.getElementById(id);
function icon(name) {
  return '<svg aria-hidden="true" viewBox="0 0 24 24">' + (icons[name] || '') + '</svg>';
}
for (const el of document.querySelectorAll('[data-icon]')) el.innerHTML = icon(el.dataset.icon);

const colors = {
  cd8_t: '#69e0bd',
  cd4_t: '#8bafff',
  b_cell: '#f6bd78',
  t_unresolved: '#aac8bc',
  myeloid: '#d785bc',
  plasma: '#f19073',
  treg: '#be95f5',
  nk: '#79dbe8',
  unassigned: '#8a9aa7',
  ilc: '#e7dd79',
  dendritic: '#df99f0',
  mast: '#c9d596'
};
const format = new Intl.NumberFormat('en-US');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
let state = null;
let scene = null;
let pathwayScene = null;
let running = false;
let animation = null;
let autoStart = null;
let detailKey = '';
let expanded = false;
let importer;
let currentSummary;
let mountVersion = 0;
let zoom = 1;
const timeTrack = $('timeTrack');

function addParagraph(container, heading, body) {
  const h = document.createElement('h3');
  h.textContent = heading;
  const p = document.createElement('p');
  p.textContent = body;
  container.append(h, p);
}

function describeStudy(study, demo) {
  const s = study.study;
  const notes = $('studyMethods');
  notes.replaceChildren();
  $('notesSource').textContent = demo ? s.accession : 'H5AD study';
  $('studyScope').textContent =
    s.scope ||
    (Number.isFinite(s.sourceRows)
      ? format.format(s.cells) +
        ' of ' +
        format.format(s.sourceRows) +
        ' observations belong to the selected visits.'
      : '');
  if (s.mapping) {
    const m = s.mapping;
    const listings = [
      ['Population', m.population],
      ['Visit', m.visit],
      ['Group', m.arm],
      ['Participant', m.participant],
      ['Expression matrix', m.layer]
    ].filter(([, value]) => value);
    const heading = document.createElement('h3');
    heading.textContent = 'Mapped fields';
    const list = document.createElement('dl');
    list.className = 'study-mapping';
    for (const [role, column] of listings) {
      const dt = document.createElement('dt');
      dt.textContent = role;
      const dd = document.createElement('dd');
      dd.textContent = column;
      list.append(dt, dd);
    }
    notes.append(heading, list);
    const omissions = [];
    if (s.audit?.missingVisit)
      omissions.push(format.format(s.audit.missingVisit) + ' observations lack a visit label.');
    if (s.audit?.otherVisits)
      omissions.push(format.format(s.audit.otherVisits) + ' observations belong to other visits.');
    if (s.audit?.missingParticipant)
      omissions.push(
        format.format(s.audit.missingParticipant) +
          ' selected cells lack a participant identifier and contribute only to pooled summaries.'
      );
    if (omissions.length) addParagraph(notes, 'Visit and participant coverage', omissions.join(' '));
  }
  if (s.embedding || s.annotation) {
    addParagraph(notes, 'Reading the map', [s.embedding, s.annotation].filter(Boolean).join(' '));
  }
  const pairs = study.groups.some((g) => g.participantPairs?.length);
  addParagraph(
    notes,
    'Cell fractions',
    'Each pooled fraction divides the selected population’s cell count by all cells in that group and visit.' +
      (pairs
        ? ' Each thin line connects the two observed fractions for one participant in the selection.'
        : '')
  );
  if (s.expression) addParagraph(notes, 'Gene expression', s.expression);
  if (study.features.some((f) => f.kind === 'pathway') && s.pathway)
    addParagraph(notes, 'Pathway scores', s.pathway);
  if (s.warnings?.length) addParagraph(notes, 'Gene-set coverage', s.warnings.join(' '));
  const links = $('studyLinks');
  links.replaceChildren();
  if (demo) {
    // Citation links belong to the curated example, separate from local file metadata.
    for (const [label, url] of [
      [s.citation, s.doi],
      [s.accession, s.url]
    ]) {
      if (!label || !url || !/^https:\/\//.test(url)) continue;
      const a = document.createElement('a');
      a.href = url;
      a.target = '_blank';
      a.rel = 'noreferrer';
      a.textContent = label;
      links.append(a);
    }
  }
  links.hidden = !links.childElementCount;
}
function tooltip(hit) {
  const el = $('cellTip');
  if (!hit || !state) {
    el.hidden = true;
    return;
  }
  const name = document.createElement('strong');
  name.textContent = hit.label;
  const details = document.createElement('div');
  if (hit.visit && state.compare) {
    const visit = state.study.visits.find((v) => v.id === hit.visit);
    if (visit) name.append(document.createTextNode(' · ' + visit.label));
  }
  if (hit.feature) {
    const feature = state.study.features.find((f) => f.id === hit.feature);
    details.textContent = feature.label + ' ≈ ' + hit.value.toFixed(2) + ' ' + feature.unit;
  } else if (hit.umap) {
    details.textContent = state.axisLabels
      .map((name, i) => name + ' ' + hit.umap[i].toFixed(2))
      .join(' · ');
  } else details.textContent = 'Select to inspect population';
  el.replaceChildren(name, details);
  el.hidden = false;
  const field = $('field');
  el.style.left = Math.max(5, Math.min(hit.x + 13, field.clientWidth - el.offsetWidth - 6)) + 'px';
  el.style.top = Math.max(5, Math.min(hit.y + 13, field.clientHeight - el.offsetHeight - 6)) + 'px';
}
function locationText() {
  const d = state.study;
  return d.visits.find((v) => v.id === state.visit).label;
}
function drawLegend() {
  const legend = $('fieldLegend');
  if (state.layer === 'composition') {
    legend.textContent = state.axisLabels.join(' × ') + '  /  ' + zoom.toFixed(1) + '×';
  } else {
    const id = state.layer === 'expression' ? state.gene : state.pathway;
    const f = state.study.features.find((f) => f.id === id);
    const line = document.createElement('div');
    line.className = 'dim-scale';
    const lamp = document.createElement('i');
    lamp.className = 'lamp';
    line.append(
      document.createTextNode(f.domain[0] + ' '),
      lamp,
      document.createTextNode(' ' + f.domain[1].toFixed(2) + ' ' + f.unit)
    );
    const name = document.createElement('span');
    name.className = 'map-signal-name';
    name.textContent = f.label + ' · ' + state.study.populations.find((p) => p.id === state.pop).label;
    legend.replaceChildren(name, line);
  }
}
function updatePopList() {
  const list = $('popList');
  const groups = state.study.groups.filter((g) => g.arm === state.arm && g.visit === state.visit);
  const max = groups.reduce((m, g) => Math.max(m, g.fraction ?? 0), 1e-6);
  const rows = groups
    .slice()
    .sort((a, b) => b.fraction - a.fraction)
    .map((group) => {
      const pop = state.study.populations.find((p) => p.id === group.population);
      const row = document.createElement('button');
      row.type = 'button';
      row.setAttribute('aria-pressed', String(pop.id === state.pop));
      row.style.setProperty('--pop-color', pop.color);
      const dot = document.createElement('i');
      dot.className = 'swatch';
      const name = document.createElement('span');
      name.textContent = pop.label;
      const bar = document.createElement('span');
      bar.className = 'fraction-bar';
      const fill = document.createElement('i');
      fill.style.width = (100 * (group.fraction ?? 0)) / max + '%';
      bar.append(fill);
      const val = document.createElement('span');
      val.className = 'fraction';
      val.textContent = Number.isFinite(group.fraction) ? (100 * group.fraction).toFixed(1) + '%' : 'n/a';
      row.append(dot, name, bar, val);
      row.addEventListener('click', () => {
        $('populationDialog').close();
        update({ pop: pop.id });
      });
      return row;
    });
  list.replaceChildren(...rows);
}
function updateDetail() {
  const group = state.study.groups.find(
    (g) => g.arm === state.arm && g.visit === state.visit && g.population === state.pop
  );
  const pop = state.study.populations.find((p) => p.id === state.pop);
  currentSummary = group;
  $('popName').textContent = pop.label;
  $('popSwatch').style.setProperty('--pop-color', pop.color);
  $('popMeta').textContent =
    format.format(group.cells) +
    ' cells' +
    (Number.isFinite(group.totalParticipants)
      ? '  ·  ' + group.participants + ' of ' + group.totalParticipants + ' participants'
      : '');
  for (const btn of document.querySelectorAll('[data-layer]'))
    btn.setAttribute('aria-pressed', String(state.layer === btn.dataset.layer));
  for (const btn of document.querySelectorAll('button[data-visit]')) {
    btn.textContent = state.study.visits.find((v) => v.id === btn.dataset.visit).label;
    btn.setAttribute('aria-pressed', String(state.visit === btn.dataset.visit));
  }
  $('arm').value = state.arm;
  const quick = $('quickFeature');
  $('quickFeatureWrap').hidden = state.layer === 'composition';
  if (state.layer !== 'composition') {
    const kind = state.layer === 'expression' ? 'gene' : 'pathway';
    const options = state.study.features.filter((f) => f.kind === kind);
    if (quick.dataset.kind !== kind) {
      quick.replaceChildren(...options.map((f) => new Option(f.label, f.id)));
      quick.dataset.kind = kind;
    }
    quick.value = state.layer === 'expression' ? state.gene : state.pathway;
  }
  const isPathway = state.layer === 'pathway';
  const members = isPathway && state.projection === 'members';
  $('pathwaySwitch').hidden = !isPathway;
  $('pathwayStage').hidden = !members;
  $('field').dataset.projection = members ? 'members' : 'map';
  $('projectionType').textContent = members ? 'Gene set' : state.embeddingLabel;
  $('reference').hidden = members;
  $('reference').setAttribute('aria-pressed', String(state.compare));
  $('fly').hidden = members;
  $('map').tabIndex = members ? -1 : 0;
  $('map').setAttribute('aria-hidden', String(members));
  for (const b of document.querySelectorAll('[data-projection][type="button"]'))
    b.setAttribute('aria-pressed', String(b.dataset.projection === state.projection));
  $('footerScene').replaceChildren();
  const dot = document.createElement('i');
  dot.className = 'mini-point';
  const hint = members
    ? 'Each spoke is one measured pathway member'
    : state.compare
      ? 'Drag the lens or use the arrow keys to inspect the other visit'
      : 'Each point is one captured cell';
  $('footerScene').append(dot, document.createTextNode(hint));
  renderEvidence($('evidence'), state, update);
  drawLegend();
  if ($('populationDialog').open) updatePopList();
}
function render() {
  if (!state) return;
  const key = [
    state.arm,
    state.visit,
    state.pop,
    state.layer,
    state.gene,
    state.pathway,
    state.compare,
    state.projection
  ].join('/');
  const changed = key !== detailKey;
  if (changed) {
    detailKey = key;
    updateDetail();
  }
  if (state.layer === 'pathway' && state.projection === 'members') {
    pathwayScene.setState(state);
    if (changed && state.member) pathwayScene.selectGene(state.member);
  } else scene.setState(state);
  const blending = state.progress > 0 && state.progress < 1;
  $('visitBadge').textContent = blending ? 'Visit blend' : locationText();
  const members = state.layer === 'pathway' && state.projection === 'members';
  $('captureCount').textContent =
    members || blending ? '' : format.format(currentSummary.totalCells) + ' cells';
  $('blendLabel').textContent = blending ? 'Visit blend' : '';
  timeTrack.value = String(Math.round(state.progress * 100));
  timeTrack.style.setProperty('--played', 100 * state.progress + '%');
  timeTrack.setAttribute(
    'aria-valuetext',
    blending
      ? Math.round(state.progress * 100) + '% ' + state.study.visits[1].label + ' layer opacity'
      : locationText()
  );
}
function change(fields) {
  state = { ...state, ...fields };
  render();
}
function playButton(active) {
  running = active;
  const play = $('play'),
    label = active ? 'Pause playback' : 'Play visits';
  play.setAttribute('aria-label', label);
  play.title = label;
  const span = play.firstElementChild;
  span.dataset.icon = active ? 'pause' : 'play';
  span.innerHTML = icon(span.dataset.icon);
}
function stopPlayback() {
  if (autoStart !== null) {
    clearTimeout(autoStart);
    autoStart = null;
  }
  if (animation !== null) {
    cancelAnimationFrame(animation);
    animation = null;
  }
  if (running) playButton(false);
}
function update(fields) {
  if (!state) return;
  if (fields.member) {
    state = { ...state, member: fields.member };
    pathwayScene?.selectGene(fields.member);
    return;
  }
  if (fields.pathway && fields.pathway !== state.pathway) {
    const preferred = state.member;
    const names = state.study.features.find((f) => f.id === fields.pathway)?.members || [];
    fields = { ...fields, member: names.includes(preferred) ? preferred : names[0] };
  }
  stopPlayback();
  const endpoint = state.visit === 'post' ? 1 : 0;
  change({ ...fields, progress: endpoint });
}
function advanceTo(target, duration, hold = 0) {
  stopPlayback();
  if (!state) return;
  if (state.compare) change({ compare: false });
  const start = state.progress;
  const visit = target ? 'post' : 'pre';
  if (state.reduced || Math.abs(start - target) < 0.001) {
    change({ progress: target, visit });
    return;
  }
  playButton(true);
  const epoch = performance.now();
  function frame(now) {
    const t = Math.max(0, Math.min(1, (now - epoch - hold) / duration));
    const eased = t * t * (3 - 2 * t);
    const pos = t === 1 ? target : start + (target - start) * eased;
    change({ progress: pos, visit: pos < 0.5 ? 'pre' : 'post' });
    if (t < 1) animation = requestAnimationFrame(frame);
    else {
      animation = null;
      playButton(false);
      $('announcer').textContent = 'Showing ' + locationText() + ' visit.';
    }
  }
  animation = requestAnimationFrame(frame);
}
function playVisits() {
  if (!state) return;
  if (running) {
    stopPlayback();
    return;
  }
  if (state.progress === 1) change({ progress: 0, visit: 'pre' });
  advanceTo(1, 1450, state.progress === 0 ? 450 : 0);
}
function setExpanded(value) {
  expanded = value;
  document.body.classList.toggle('map-expanded', value);
  const b = $('focus');
  b.setAttribute('aria-pressed', String(value));
  b.setAttribute('aria-label', value ? 'Restore inspector' : 'Expand map');
  b.title = b.getAttribute('aria-label');
  b.firstElementChild.innerHTML = icon(value ? 'compress' : 'expand');
  if (value) window.scrollTo(0, 0);
}
function showModal(id) {
  stopPlayback();
  if (!state) return;
  if (id === 'populationDialog') updatePopList();
  if (!$(id).open) $(id).showModal();
}
function events() {
  $('openH5ad').addEventListener('click', () => {
    stopPlayback();
    importer.open();
  });
  $('mapFields').addEventListener('click', () => {
    stopPlayback();
    importer.open({ remap: true });
  });
  for (const b of document.querySelectorAll('[data-layer]'))
    b.addEventListener('click', () => update({ layer: b.dataset.layer }));
  for (const b of document.querySelectorAll('button[data-visit]'))
    b.addEventListener('click', () => advanceTo(b.dataset.visit === 'post' ? 1 : 0, 650));
  for (const b of document.querySelectorAll('[data-camera]'))
    b.addEventListener('click', () => {
      stopPlayback();
      if (!scene) return;
      if (b.dataset.camera === 'reset') scene.reset();
      else scene.zoomBy(b.dataset.camera === 'in' ? 1.45 : 1 / 1.45);
    });
  for (const b of document.querySelectorAll('button[data-projection]'))
    b.addEventListener('click', () => update({ projection: b.dataset.projection }));
  $('fly').addEventListener('click', () => {
    stopPlayback();
    scene?.focusPopulation();
  });
  $('quickFeature').addEventListener('change', (e) =>
    update({ [state.layer === 'expression' ? 'gene' : 'pathway']: e.target.value })
  );
  $('arm').addEventListener('change', (e) => update({ arm: e.target.value }));
  $('play').addEventListener('click', playVisits);
  timeTrack.addEventListener('input', () => {
    if (!state) return;
    stopPlayback();
    const p = +timeTrack.value / 100;
    change({ progress: p, visit: p < 0.5 ? 'pre' : 'post', compare: false });
  });
  $('reference').addEventListener('click', () => state && update({ compare: !state.compare }));
  $('focus').addEventListener('click', () => setExpanded(!expanded));
  $('map').addEventListener('pointerdown', stopPlayback);
  $('map').addEventListener('wheel', stopPlayback, { passive: true });
  $('map').addEventListener('keydown', stopPlayback);
  for (const id of ['openNotes', 'footerNotes']) $(id).addEventListener('click', () => showModal('notes'));
  for (const id of ['openPop', 'choosePop'])
    $(id).addEventListener('click', () => showModal('populationDialog'));
  for (const b of document.querySelectorAll('#notes [data-close],#populationDialog [data-close]'))
    b.addEventListener('click', () => b.closest('dialog').close());
  for (const d of [$('notes'), $('populationDialog')])
    d.addEventListener('click', (e) => {
      if (e.target !== d) return;
      const r = d.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) d.close();
    });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && expanded && !document.querySelector('dialog[open]')) setExpanded(false);
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopPlayback();
  });
  motion.addEventListener('change', () => {
    if (state) update({ reduced: motion.matches });
  });
}
async function getData() {
  const resources = ['./data/study.json', './data/map.bin'];
  const [meta, map] = await Promise.all(
    resources.map(async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw Error('Missing study resource');
      return url.endsWith('.json') ? response.json() : response.arrayBuffer();
    })
  );
  if (!meta.map || map.byteLength !== meta.map.rows * meta.map.stride) throw Error('Incomplete cell map');
  return [meta, map];
}
// Golden-angle spacing keeps additional population colors distinct when an H5AD has many labels.
function populationColor(index) {
  const hue = (index * 137.508 * Math.PI) / 180;
  const channel = (offset) =>
    Math.round(155 + 83 * Math.cos(hue + offset))
      .toString(16)
      .padStart(2, '0');
  return '#' + [0, (-2 * Math.PI) / 3, (2 * Math.PI) / 3].map(channel).join('');
}
function mountStudy(study, points, demo = false) {
  if (
    study.visits?.length !== 2 ||
    !study.arms?.length ||
    !study.populations?.length ||
    !study.groups?.length ||
    !study.map ||
    points.byteLength !== study.map.rows * study.map.stride
  ) {
    throw Error('This study does not contain a complete two-visit cell map.');
  }
  stopPlayback();
  mountVersion++;
  detailKey = '';
  zoom = 1;
  setExpanded(false);
  $('quickFeature').removeAttribute('data-kind');
  const palette = Object.values(colors);
  study.populations.forEach((population, index) => {
    population.color = (demo && colors[population.id]) || palette[index] || populationColor(index);
  });
  const meta = study.study;
  const arm = demo ? 'combo' : study.arms[0].id;
  const pop = demo
    ? 'cd8_t'
    : (study.groups
        .filter((g) => g.arm === arm && g.visit === 'pre')
        .reduce((largest, g) => (!largest || g.cells > largest.cells ? g : largest), null)?.population ??
      study.populations[0].id);
  const gene = study.features.find((f) => f.kind === 'gene');
  const pathway = study.features.find((f) => f.kind === 'pathway');
  const member = pathway?.members?.includes('IFIT1') && demo ? 'IFIT1' : pathway?.members?.[0];
  const title = demo ? 'Tumor immune atlas' : meta.title;
  // Offer line breaks at filename separators while preserving the actual name.
  const parts = String(title).split(/(?<=[_./-])/);
  $('atlasTitle').replaceChildren(
    ...parts.flatMap((part, index) => (index ? [document.createElement('wbr'), part] : [part]))
  );
  document.title = 'sctrial · ' + $('atlasTitle').textContent;
  $('sourceLabel').textContent = demo ? meta.accession + '  /  scRNA-seq' : 'H5AD study';
  $('studySubtitle').textContent = demo
    ? meta.title + ' · Paired tumor biopsies'
    : study.visits.map((v) => v.label).join(' → ');
  $('cohortCells').textContent = format.format(meta.cells);
  $('cohortPeople').textContent = Number.isFinite(meta.participants)
    ? format.format(meta.participants)
    : '';
  $('participantStat').hidden = !Number.isFinite(meta.participants);
  $('cohortVisits').textContent = String(study.visits.length);
  $('studyStats').hidden = false;
  $('arm').replaceChildren(...study.arms.map((a) => new Option(a.label, a.id)));
  $('armPicker').hidden = study.arms.length === 1;
  $('mapFields').hidden = demo;
  $('openNotes').disabled = false;
  $('atlasControls').inert = false;
  for (const button of document.querySelectorAll('button[data-layer]')) {
    const layer = button.dataset.layer;
    button.disabled = (layer === 'expression' && !gene) || (layer === 'pathway' && !pathway);
    button.title = button.disabled
      ? layer === 'expression'
        ? 'No genes loaded for this study.'
        : 'No pathway gene sets loaded for this study.'
      : '';
  }
  describeStudy(study, demo);
  state = {
    study,
    points,
    embeddingLabel: meta.embeddingLabel || (demo ? 'UMAP' : 'Embedding'),
    projection: 'members',
    member,
    arm,
    visit: 'pre',
    pop,
    gene: gene?.id,
    pathway: pathway?.id,
    layer: 'composition',
    compare: false,
    reduced: motion.matches,
    progress: 0
  };
  state.axisLabels = study.map.axisLabels || [state.embeddingLabel + ' 1', state.embeddingLabel + ' 2'];
  render();
  scene.reset();
  $('busy').hidden = true;
  $('instrument').setAttribute('aria-busy', 'false');
  if (demo && !state.reduced && !document.hidden && !document.querySelector('dialog[open]')) {
    autoStart = setTimeout(() => {
      autoStart = null;
      advanceTo(1, 1450, 0);
    }, 3500);
  }
}
async function start() {
  scene = createScene($('map'), {
    onSelect: (id) => update({ pop: id }),
    onInspect: tooltip,
    onCamera: (camera) => {
      zoom = camera.zoom;
      if (state) drawLegend();
    }
  });
  pathwayScene = createPathwayScene($('pathwayStage'), {
    onInspect: (hit) => {
      if (hit?.gene && state) {
        state = { ...state, member: hit.gene };
        selectEvidenceGene($('evidence'), hit.gene);
      }
    }
  });
  importer = setupImporter({
    onStudy: mountStudy,
    onStatus: (message) => {
      stopPlayback();
      if (typeof message === 'string') $('announcer').textContent = message;
    }
  });
  events();
  document.addEventListener(
    'pointerdown',
    () => {
      if (autoStart !== null) stopPlayback();
    },
    { capture: true }
  );
  document.addEventListener(
    'keydown',
    () => {
      if (autoStart !== null) stopPlayback();
    },
    { capture: true }
  );
  const initial = mountVersion;
  if (await importer.autoOpen()) {
    if (!state) {
      $('busy').textContent = 'Open an H5AD and select two visits to explore.';
      $('visitBadge').textContent = 'Select a study';
      $('instrument').setAttribute('aria-busy', 'false');
    }
    return;
  }
  const [study, points] = await getData();
  if (initial === mountVersion) mountStudy(study, points, true);
}
start().catch((error) => {
  console.error(error);
  $('busy').textContent = 'The study could not load. Open an H5AD or reload to retry.';
  $('visitBadge').textContent = 'Study unavailable';
  $('instrument').setAttribute('aria-busy', 'false');
});
