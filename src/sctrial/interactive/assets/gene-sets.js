(() => {
  'use strict';

  const maxBytes = 16 * 1024 * 1024;
  const maxSets = 30000;
  const maxMembers = 25000;
  const maxOptions = 120;
  const libraries = new WeakMap();

  function parse(text) {
    const records = new Map();
    const lines = text.replace(/^\uFEFF/, '').split(/\r\n|[\r\n]/);
    for (let i = 0; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      const [rawName, rawDescription, ...fields] = lines[i].split('\t');
      const name = rawName.trim();
      const description = rawDescription?.trim() || '';
      const error = (message) => new Error('GMT line ' + (i + 1) + '. ' + message);
      if (!name || name.length > 512 || fields.length === 0)
        throw error('Supply a set name, description field and tab-separated gene symbols.');
      if (records.has(name)) throw error('A second set uses the name “' + name + '”.');
      if (records.size >= maxSets) throw error('Use a file with at most 30,000 sets.');
      if (/[\u0000-\u001f\u007f]/.test(name + description))
        throw error('Remove control characters from the name and description.');
      const members = new Set();
      for (const field of fields) {
        const symbol = field.trim();
        if (!symbol) continue;
        if (symbol.length > 256 || /[\s,\u0000-\u001f\u007f]/.test(symbol))
          throw error('Use one gene symbol per tab-separated field.');
        members.add(symbol);
        if (members.size > maxMembers) throw error('Use at most 25,000 distinct members per set.');
      }
      if (!members.size) throw error('Include at least one gene symbol.');
      records.set(name, { name, description, members: [...members] });
    }
    if (!records.size) throw new Error('The file contains no gene sets.');
    return records;
  }

  const content = `
    <form id="geneSetForm">
      <p class="hint">Link gene-set membership to the measured pathway <strong id="geneSetTarget"></strong>. Membership is kept for this open study.</p>
      <label class="field"><span>GMT file</span>
        <input id="geneSetFile" type="file" accept=".gmt,text/plain" class="field-input"></label>
      <p class="hint" id="geneSetStatus" role="status">Choose a local, tab-delimited GMT file.</p>
      <label class="field" id="geneSetSearchBox" hidden><span>Find a set</span>
        <input id="geneSetSearch" type="search" class="field-input" autocomplete="off"></label>
      <label class="field"><span>Gene set</span>
        <select id="geneSetChoice" class="field-input" disabled><option value="">Choose a gene set</option></select></label>
      <p class="hint" id="geneSetPreview" aria-live="polite"></p>
      <div class="error" id="geneSetError" role="alert" hidden></div>
      <div class="form-actions">
        <button type="button" class="btn" id="cancelGeneSet">Cancel</button>
        <button type="submit" class="btn accent" id="applyGeneSet" disabled>Link membership</button>
      </div>
    </form>`;

  function open({ study, featureId, show, close, onChange }) {
    const target = study.features.find((f) => f.id === featureId && f.kind === 'pathway');
    if (!target) {
      show(
        'Link gene set',
        '<p class="hint">Open a study with a measured pathway score column named <code>pathway:NAME</code>.</p>',
      );
      return;
    }
    show('Link gene set', content, 'gene-set');
    const form = document.getElementById('geneSetForm');
    const get = (id) => form.querySelector('#' + id);
    const picker = get('geneSetFile');
    const choice = get('geneSetChoice');
    const search = get('geneSetSearch');
    const preview = get('geneSetPreview');
    const status = get('geneSetStatus');
    const error = get('geneSetError');
    const apply = get('applyGeneSet');
    const name = target.label || target.id;
    let library = libraries.get(study);
    let chosen = null;
    let generation = 0;
    get('geneSetTarget').textContent = name;
    if (target.membershipSource) apply.textContent = 'Replace membership';

    function describe() {
      const set = library?.sets.get(chosen);
      apply.disabled = !set;
      preview.textContent = '';
      if (!set) return;
      const head = set.members.slice(0, 7).join(', ');
      const rest = set.members.length > 7 ? '…' : '';
      const count =
        set.members.length.toLocaleString() + (set.members.length === 1 ? ' member' : ' members');
      const description =
        set.description && !/^n\/?a$/i.test(set.description) ? ' · ' + set.description : '';
      preview.textContent = count + ' · ' + head + rest + description;
    }

    function options() {
      const query = search.value.trim().toLocaleLowerCase();
      const matches = [...library.sets.keys()].filter((key) =>
        key.toLocaleLowerCase().includes(query),
      );
      const shown = matches.slice(0, maxOptions);
      if (chosen && !shown.includes(chosen)) shown.unshift(chosen);
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = 'Choose a gene set';
      const options = shown.map((key) => {
        const option = document.createElement('option');
        option.value = option.textContent = key;
        return option;
      });
      choice.replaceChildren(placeholder, ...options);
      choice.value = chosen || '';
      choice.disabled = !shown.length;
      if (matches.length > maxOptions)
        status.textContent =
          library.filename +
          ' · Showing the first ' +
          maxOptions +
          ' of ' +
          matches.length.toLocaleString() +
          ' matches. Refine the search to see more.';
      else if (!matches.length) status.textContent = 'No set names match this search.';
      else
        status.textContent =
          library.filename +
          ' · ' +
          library.sets.size.toLocaleString() +
          (library.sets.size === 1 ? ' set' : ' sets');
      describe();
    }

    function selectLibrary(next) {
      library = next;
      search.value = '';
      get('geneSetSearchBox').hidden = library.sets.size <= maxOptions;
      const label = name.replace(/^pathway\s*:\s*/i, '').trim();
      const previous = target.membershipSource?.name;
      chosen =
        next === libraries.get(study) && next.sets.has(previous)
          ? previous
          : next.sets.has(label)
            ? label
            : null;
      options();
    }

    if (library) selectLibrary(library);
    picker.addEventListener('change', async () => {
      const file = picker.files[0];
      if (!file) return;
      const own = ++generation;
      library = null;
      chosen = null;
      search.value = '';
      get('geneSetSearchBox').hidden = true;
      choice.replaceChildren();
      choice.disabled = apply.disabled = true;
      error.hidden = true;
      preview.textContent = '';
      status.textContent = 'Reading ' + file.name + '…';
      const active = () => own === generation && form.isConnected && form.closest('dialog')?.open;
      try {
        if (file.size > maxBytes) throw new Error('Choose a GMT file of 16 MB or less.');
        const text = await file.text();
        if (!active()) return;
        selectLibrary({ filename: file.name, sets: parse(text) });
      } catch (err) {
        if (!active()) return;
        status.textContent = 'Choose a different GMT file.';
        error.textContent = err.message;
        error.hidden = false;
      }
    });
    search.addEventListener('input', () => {
      chosen = null;
      options();
    });
    choice.addEventListener('change', () => {
      chosen = choice.value || null;
      describe();
    });
    get('cancelGeneSet').addEventListener('click', close);
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const record = library?.sets.get(chosen);
      if (!record || study.features.find((f) => f.id === featureId) !== target) return;
      target.members = [...record.members];
      target.membershipSource = { name: record.name, description: record.description };
      libraries.set(study, library);
      close();
      onChange?.();
    });
  }

  window.SctrialGeneSets = Object.freeze({ open });
})();
