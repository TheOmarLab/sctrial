// File objects stay in this browser and are read only by the local HDF5 worker.
const aliases = {
  population: [
    'cell_type',
    'celltype',
    'cell_type_label',
    'cell_type_major',
    'population',
    'annotation',
    'leiden'
  ],
  visit: ['visit', 'timepoint', 'time_point', 'visit_id', 'collection_time', 'time', 'day'],
  participant: [
    'participant_id',
    'patient_id',
    'donor_id',
    'subject_id',
    'participant',
    'patient',
    'donor',
    'subject'
  ],
  arm: ['arm', 'treatment_arm', 'treatment_group', 'group', 'condition', 'treatment']
};
const measuredMarkers = ['GZMB', 'NKG7', 'ISG15', 'IFIT1', 'CD79A', 'MS4A1'];

function workerClient(onProgress) {
  const worker = new Worker(new URL('./import-worker.js', import.meta.url), { type: 'module' });
  let nextId = 0;
  const requests = new Map();
  function rejectAll(message) {
    for (const { reject } of requests.values()) reject(Error(message));
    requests.clear();
  }
  worker.onmessage = ({ data }) => {
    const pending = requests.get(data.id);
    if (!pending) return;
    if ('progress' in data) {
      onProgress(data.progress);
      return;
    }
    requests.delete(data.id);
    if (data.error) pending.reject(Error(data.error));
    else pending.resolve(data.result);
  };
  worker.onerror = (e) => {
    e.preventDefault();
    rejectAll(e.message || 'The local HDF5 reader could not start.');
  };
  worker.onmessageerror = () => rejectAll('The browser could not read the response from the HDF5 worker.');
  return {
    request(type, args = {}) {
      return new Promise((resolve, reject) => {
        const id = ++nextId;
        requests.set(id, { resolve, reject });
        try {
          worker.postMessage({ id, type, ...args });
        } catch (e) {
          requests.delete(id);
          reject(e);
        }
      });
    },
    close() {
      worker.terminate();
      rejectAll('Import canceled.');
    }
  };
}

const dialogHTML = `
  <div class="im-head"><span class="im-kicker">Study files</span><button type="button" data-im="close" aria-label="Close H5AD import">×</button></div>
  <h2 id="im-title">Open an H5AD</h2>
  <section data-im="pick">
    <p>The atlas reads your file in this browser. It stays on your device.</p>
    <label class="im-drop"><strong>Choose an H5AD file</strong><span>or drop it here</span><input data-im="file" type="file" accept=".h5ad,.h5,.hdf5" aria-label="Choose an H5AD file"></label>
  </section>
  <form data-im="form" hidden>
    <div class="im-file"><strong data-im="name"></strong><span data-im="size"></span><button type="button" data-im="replace">Change file</button></div>
    <fieldset><legend>Study design</legend>
      <div class="im-grid">
        <label>Population column<select data-im="population" required></select></label>
        <label>Visit column<select data-im="visit" required></select></label>
        <label>Reference visit<select data-im="baseline" required></select></label>
        <label>Comparison visit<select data-im="followup" required></select></label>
        <label>Participant column <small>optional</small><select data-im="participant"></select></label>
        <label>Arm column <small>optional</small><select data-im="arm"></select></label>
      </div>
      <small>For paired lines, choose an ID that identifies the same participant throughout the file.</small>
    </fieldset>
    <fieldset><legend>Coordinates and expression</legend>
      <div class="im-grid">
        <label>Map coordinates<select data-im="embedding"></select></label>
        <label>Expression matrix<select data-im="layer"></select></label>
        <label>Gene identifiers<select data-im="symbols"></select></label>
        <label>Expression scale<select data-im="normalization"><option value="as_is">Use nonnegative values as stored</option><option value="log1p">Already log1p-normalized</option><option value="counts">Raw counts → log1p per 10,000</option></select></label>
      </div>
      <div class="im-grid im-axis" data-im="axes" hidden>
        <label>Horizontal gene<input data-im="axisX" list="im-gene-suggestions" autocomplete="off"></label>
        <label>Vertical gene<input data-im="axisY" list="im-gene-suggestions" autocomplete="off"></label>
      </div>
    </fieldset>
    <fieldset data-im="signals"><legend>Genes and pathways</legend>
      <label>Find a gene in this matrix</label>
      <div class="im-add"><input data-im="gene" list="im-gene-suggestions" autocomplete="off" aria-label="Find a gene" placeholder="Type a gene identifier"><button type="button" data-im="add">Add gene</button></div>
      <datalist id="im-gene-suggestions"></datalist><div class="im-chips" data-im="chips"></div>
      <div class="im-path-title">Human Reactome gene sets <small>2022</small></div><div class="im-paths" data-im="pathways"></div>
      <details data-im="custom"><summary>Use your own gene sets</summary><label>GMT file <input type="file" data-im="gmtFile" accept=".gmt,.txt"></label><label>Or paste tab-separated GMT rows<textarea data-im="gmt" rows="3" spellcheck="false" placeholder="Name&#9;Description&#9;Gene1&#9;Gene2"></textarea></label><small>Gene identifiers must match the selected matrix.</small></details>
    </fieldset>
    <div class="im-actions"><button type="button" data-im="cancel">Cancel</button><button type="submit" data-im="build">Explore study</button></div>
  </form>
  <p class="im-status" data-im="status" role="status" aria-live="polite"></p>
  <p class="im-error" data-im="error" role="alert" hidden></p>`;

function setOptions(select, entries, selected) {
  const options = entries.map(([value, label]) => new Option(label, value));
  select.replaceChildren(...options);
  if (entries.some(([value]) => value === selected)) select.value = selected;
}

function readGMT(text) {
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      const [label, , ...raw] = line.split('\t');
      const members = [...new Set(raw.map((s) => s.trim()).filter(Boolean))];
      if (!label?.trim() || !members.length)
        throw Error(
          'GMT row ' +
            (index + 1) +
            ' needs a name, a description field and at least one tab-separated gene.'
        );
      return { label: label.trim(), members };
    });
}

export function setupImporter({ onStudy, onStatus = () => {} }) {
  const dialog = document.createElement('dialog');
  dialog.className = 'im-dialog';
  dialog.setAttribute('aria-labelledby', 'im-title');
  dialog.innerHTML = dialogHTML;
  document.body.append(dialog);
  const ui = Object.fromEntries([...dialog.querySelectorAll('[data-im]')].map((el) => [el.dataset.im, el]));
  const suggestions = dialog.querySelector('datalist');
  let worker = null,
    file = null,
    schema = null,
    library = [];
  let geneNames = [],
    geneIndex = new Set(),
    selectedGenes = new Set(),
    selectedPaths = new Set();
  let last = null,
    busy = false,
    epoch = 0,
    levelsTask = 0,
    genesTask = 0;
  let loadingVisits = false,
    loadingGenes = false,
    loadingGMT = false,
    gmtTask = 0;

  function notify(message) {
    ui.status.textContent = message || '';
    onStatus(message || '');
  }
  function error(message) {
    ui.error.textContent = message || '';
    ui.error.hidden = !message;
  }
  function setReady() {
    ui.build.disabled =
      busy ||
      loadingVisits ||
      loadingGenes ||
      loadingGMT ||
      !ui.visit.value ||
      !ui.population.value ||
      !ui.baseline.value ||
      !ui.followup.value ||
      ui.baseline.value === ui.followup.value;
  }
  function lock(value) {
    busy = value;
    for (const field of ui.form.querySelectorAll('fieldset')) field.disabled = value;
    ui.replace.disabled = value;
    setReady();
  }
  function release() {
    epoch++;
    worker?.close();
    worker = null;
    loadingGenes = loadingVisits = loadingGMT = false;
    lock(false);
  }
  function showPick() {
    release();
    file = schema = null;
    ui.form.hidden = true;
    ui.pick.hidden = false;
    ui.file.value = '';
    dialog.querySelector('h2').textContent = 'Open an H5AD';
    notify('');
    error('');
  }
  function dismiss() {
    release();
    if (dialog.open) dialog.close();
    notify('');
  }
  function chips() {
    ui.chips.replaceChildren(
      ...[...selectedGenes].map((name) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = name + ' ×';
        button.setAttribute('aria-label', 'Remove gene ' + name);
        button.addEventListener('click', () => {
          selectedGenes.delete(name);
          chips();
        });
        return button;
      })
    );
  }
  function paths() {
    ui.pathways.replaceChildren(
      ...library.map((item) => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = item.id;
        const overlap = item.members.reduce((n, g) => n + Number(geneIndex.has(g)), 0);
        checkbox.disabled = !overlap;
        if (!overlap) selectedPaths.delete(item.id);
        checkbox.checked = selectedPaths.has(item.id);
        checkbox.addEventListener('change', () => {
          if (checkbox.checked) selectedPaths.add(item.id);
          else selectedPaths.delete(item.id);
        });
        const title = document.createElement('span');
        title.textContent = item.label;
        const coverage = document.createElement('small');
        coverage.textContent = overlap + '/' + item.members.length + ' genes';
        label.append(checkbox, title, coverage);
        return label;
      })
    );
  }
  function suggest(text) {
    const query = text.toLowerCase();
    const found = [];
    if (query)
      for (const name of geneNames) {
        if (name.toLowerCase().includes(query)) found.push(name);
        if (found.length === 40) break;
      }
    suggestions.replaceChildren(...found.map((name) => new Option(name, name)));
  }
  async function loadGenes(previous, initial) {
    const call = ++genesTask,
      active = epoch;
    const matrix = schema?.layers.find((m) => m.id === ui.layer.value);
    loadingGenes = true;
    setReady();
    error('');
    try {
      let names = [];
      if (matrix)
        names = await worker.request('genes', { layer: matrix.id, symbolColumn: ui.symbols.value });
      if (call !== genesTask || active !== epoch) return;
      geneIndex = new Set();
      const ambiguous = new Set();
      for (const name of names) {
        if (geneIndex.has(name)) ambiguous.add(name);
        else if (name) geneIndex.add(name);
      }
      for (const name of ambiguous) geneIndex.delete(name);
      geneNames = [...geneIndex];
      if (previous) selectedGenes = new Set(previous.genes.filter((n) => geneIndex.has(n)));
      else if (initial)
        selectedGenes = new Set(measuredMarkers.filter((n) => geneIndex.has(n)).slice(0, 3));
      else selectedGenes = new Set([...selectedGenes].filter((n) => geneIndex.has(n)));
      ui.signals.hidden = !matrix;
      chips();
      paths();
      suggest('');
    } catch (e) {
      if (call === genesTask && active === epoch) {
        geneNames = [];
        geneIndex = new Set();
        selectedGenes.clear();
        chips();
        paths();
        error(e.message);
      }
    } finally {
      if (call === genesTask && active === epoch) {
        loadingGenes = false;
        setReady();
      }
    }
  }
  async function loadMatrix(previous, initial = false) {
    const layer = schema.layers.find((m) => m.id === ui.layer.value);
    const cols = layer?.geneColumns || [];
    const preferred = cols.find((n) =>
      ['gene_symbols', 'gene_symbol', 'gene_name', 'feature_name', 'symbol'].includes(n.toLowerCase())
    );
    setOptions(
      ui.symbols,
      [['', 'Variable index'], ...cols.map((n) => [n, n])],
      previous?.symbolColumn ?? preferred ?? ''
    );
    ui.normalization.value =
      previous?.normalization ??
      (/(^|\/)counts$/i.test(layer?.id || '')
        ? 'counts'
        : /(^|\/)log1p_norm$/i.test(layer?.id || '')
          ? 'log1p'
          : 'as_is');
    await loadGenes(previous, initial);
  }
  async function loadLevels(previous) {
    const call = ++levelsTask,
      active = epoch;
    loadingVisits = true;
    setReady();
    error('');
    ui.baseline.replaceChildren();
    ui.followup.replaceChildren();
    try {
      if (!ui.visit.value) return;
      const values = (await worker.request('levels', { name: ui.visit.value })).map(String);
      if (call !== levelsTask || active !== epoch) return;
      const list = values.map((n) => [n, n]);
      const before =
        previous?.baseline ??
        values.find((n) => /^(pre|baseline|pre.treatment|before|day[ _-]?0|t0|0)$/i.test(n)) ??
        values[0];
      const after =
        previous?.followup ??
        values.find((n) => n !== before && /^(post|follow.?up|on.treatment|after)$/i.test(n)) ??
        values.find((n) => n !== before);
      setOptions(ui.baseline, list, before);
      setOptions(ui.followup, list, after);
      if (values.length < 2) error('Choose a visit column with at least two observed values.');
    } catch (e) {
      if (call === levelsTask && active === epoch) error(e.message);
    } finally {
      if (call === levelsTask && active === epoch) {
        loadingVisits = false;
        setReady();
      }
    }
  }
  async function inspect(source, previous = null) {
    release();
    const active = epoch;
    file = source;
    ui.pick.hidden = true;
    ui.form.hidden = true;
    ui.status.hidden = false;
    error('');
    notify('Reading H5AD metadata…');
    try {
      worker = workerClient((message) => {
        if (epoch === active) notify(message);
      });
      const result = await worker.request('open', { file: source });
      if (epoch !== active) return;
      ({ schema, pathways: library } = result);
      library = Array.isArray(library) ? library : [];
      ui.name.textContent = schema.name;
      ui.size.textContent = schema.nObs.toLocaleString() + ' cells';
      dialog.querySelector('h2').textContent = 'Map your study';
      const cols = schema.obsColumns.map((c) => c.name);
      for (const key of Object.keys(aliases)) {
        const recommended = aliases[key]
          .map((alias) => cols.find((name) => name.toLowerCase() === alias))
          .find(Boolean);
        setOptions(
          ui[key],
          [
            ['', key === 'arm' || key === 'participant' ? 'None' : 'Choose a column'],
            ...cols.map((n) => [n, n])
          ],
          previous?.[key] ?? recommended ?? ''
        );
      }
      const embs = schema.embeddings;
      const projection =
        previous?.embedding ?? embs.find((n) => n.toLowerCase().includes('umap')) ?? embs[0] ?? '';
      setOptions(ui.embedding, [...embs.map((n) => [n, n]), ['', 'Two measured genes']], projection);
      ui.axes.hidden = !!ui.embedding.value;
      ui.axisX.value = previous?.axisGenes?.[0] || '';
      ui.axisY.value = previous?.axisGenes?.[1] || '';
      const matrices = schema.layers.map((m) => m.id);
      setOptions(
        ui.layer,
        matrices.length ? matrices.map((n) => [n, n]) : [['', 'No expression matrix']],
        previous?.layer ?? (matrices.includes('layers/log1p_norm') ? 'layers/log1p_norm' : matrices[0])
      );
      selectedPaths = new Set(previous?.pathways || []);
      ui.gmtFile.value = '';
      ui.gmt.value = previous?.gmtText || '';
      ui.gene.value = '';
      ui.custom.open = !!ui.gmt.value;
      ui.form.hidden = false;
      await Promise.all([loadLevels(previous), loadMatrix(previous, true)]);
      if (epoch === active) notify('');
    } catch (e) {
      if (active !== epoch) return;
      showPick();
      error(e.message);
    }
  }
  function open({ remap = false } = {}) {
    showPick();
    if (!dialog.open) dialog.showModal();
    if (remap && last) void inspect(last.file, last.config);
  }
  function addGene() {
    const name = ui.gene.value.trim();
    if (!name) return;
    if (!geneIndex.has(name)) {
      error('Choose a gene identifier that occurs exactly once in the selected matrix.');
      return;
    }
    error('');
    selectedGenes.add(name);
    ui.gene.value = '';
    suggest('');
    chips();
  }
  async function build(event) {
    event.preventDefault();
    if (ui.build.disabled || !worker) return;
    const active = epoch;
    try {
      error('');
      const gmtText = ui.gmt.value;
      const config = {
        title: schema.name,
        population: ui.population.value,
        visit: ui.visit.value,
        participant: ui.participant.value || null,
        arm: ui.arm.value || null,
        baseline: ui.baseline.value,
        followup: ui.followup.value,
        embedding: ui.embedding.value,
        axisGenes: ui.embedding.value ? [] : [ui.axisX.value.trim(), ui.axisY.value.trim()],
        layer: ui.layer.value,
        symbolColumn: ui.symbols.value,
        normalization: ui.normalization.value,
        genes: [...selectedGenes],
        pathways: [...selectedPaths],
        customPathways: readGMT(gmtText)
      };
      if (ui.gene.value.trim()) throw Error('Add the gene you typed or clear the gene field.');
      const roles = [config.population, config.visit, config.participant, config.arm].filter(Boolean);
      if (new Set(roles).size !== roles.length)
        throw Error('Assign a different observation column to each study field.');
      if (
        !config.embedding &&
        (!config.axisGenes.every((g) => geneIndex.has(g)) || config.axisGenes[0] === config.axisGenes[1])
      )
        throw Error('Choose two distinct measured genes or an existing embedding.');
      lock(true);
      notify('Preparing your selected visits…');
      const result = await worker.request('build', { config });
      if (epoch !== active) return;
      await onStudy(result.study, result.points);
      last = { file, config: { ...config, gmtText } };
      dismiss();
    } catch (e) {
      if (epoch === active) {
        lock(false);
        notify('');
        error(e.message);
      }
    }
  }
  ui.form.addEventListener('submit', build);
  ui.close.onclick = ui.cancel.onclick = dismiss;
  dialog.addEventListener('cancel', (e) => {
    e.preventDefault();
    dismiss();
  });
  ui.replace.onclick = showPick;
  ui.file.onchange = () => {
    if (ui.file.files.length) void inspect(ui.file.files[0]);
  };
  ui.pick.addEventListener('dragover', (e) => e.preventDefault());
  ui.pick.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) void inspect(e.dataTransfer.files[0]);
  });
  ui.visit.onchange = () => void loadLevels();
  ui.baseline.onchange = ui.followup.onchange = ui.population.onchange = setReady;
  ui.layer.onchange = () => void loadMatrix();
  ui.symbols.onchange = () => void loadGenes();
  ui.embedding.onchange = () => {
    ui.axes.hidden = !!ui.embedding.value;
  };
  ui.add.onclick = addGene;
  for (const input of [ui.gene, ui.axisX, ui.axisY]) input.oninput = () => suggest(input.value);
  ui.gene.onkeydown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addGene();
    }
  };
  ui.gmtFile.onchange = async () => {
    const active = epoch,
      call = ++gmtTask;
    const chosen = ui.gmtFile.files[0];
    loadingGMT = Boolean(chosen);
    setReady();
    if (!chosen) return;
    ui.gmt.value = '';
    error('');
    try {
      const text = await chosen.text();
      if (active !== epoch || call !== gmtTask) return;
      ui.gmt.value = text;
      readGMT(text);
    } catch (e) {
      if (active === epoch && call === gmtTask) error(e.message);
    } finally {
      if (active === epoch && call === gmtTask) {
        loadingGMT = false;
        setReady();
      }
    }
  };
  return {
    open,
    async autoOpen() {
      const local = window.SCTRIAL_LOCAL;
      if (local?.file) {
        open();
        void inspect({ ...local.file, session: local.session });
        return true;
      }
      if (location.hash === '#open') {
        open();
        return true;
      }
      return false;
    }
  };
}
