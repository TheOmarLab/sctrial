(() => {
  'use strict';

  const scoreLimit = 12;
  const roles = [
    { key: 'participant', label: 'Participant ID', empty: 'Choose a column' },
    { key: 'arm', label: 'Group', empty: 'One group' },
    { key: 'visit', label: 'Visit', empty: 'One collection occasion' },
    { key: 'population', label: 'Cell population', empty: 'All cells' },
  ];
  let active = null;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function makeButton(label, className, onClick) {
    const button = el('button', className, label);
    button.type = 'button';
    button.addEventListener('click', onClick);
    return button;
  }

  function available() {
    return typeof window.SCTRIAL_LOCAL?.session === 'string' && Boolean(window.SCTRIAL_LOCAL.session);
  }

  async function request(endpoint, payload, signal) {
    if (!available()) throw new Error('Open the workspace from the local H5AD launcher.');
    const headers = { Accept: 'application/json', 'X-sctrial-session': window.SCTRIAL_LOCAL.session };
    const options = {
      method: 'GET',
      headers,
      cache: 'no-store',
      credentials: 'same-origin',
      redirect: 'error',
      signal,
    };
    if (payload !== undefined) {
      options.method = 'POST';
      headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(payload);
    }
    let response;
    try {
      response = await fetch(endpoint, options);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      throw new Error('The local workspace could not be reached. Check the launcher and try again.');
    }
    let result;
    try {
      result = await response.json();
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      throw new Error('The local workspace returned an unreadable response.');
    }
    if (!response.ok) {
      throw new Error(
        typeof result?.error === 'string'
          ? result.error
          : 'The local workspace could not complete this request.',
      );
    }
    return result;
  }

  async function fit(study, config, signal) {
    if (!study?.studyId) throw new Error('Map the study fields before fitting a comparison.');
    if (!['did', 'paired', 'cross'].includes(config.design))
      throw new Error('Choose a supported comparison.');
    const body = {
      studyId: study.studyId,
      design: config.design,
      arm: config.arm,
      visit: config.visit,
      population: config.population,
    };
    if (config.design !== 'cross') body.baseline = config.baseline;
    if (config.design !== 'paired') body.reference = config.reference;
    if (Array.isArray(config.participants)) body.participants = config.participants;
    const result = await request('/local/fit', body, signal);
    if (!result || !Array.isArray(result.rows))
      throw new Error('The local fit returned no results table.');
    return result;
  }

  function inferVisitOrder(visits) {
    if (visits.length < 2) return visits.slice();
    const labels = visits.map((visit) => visit.trim().toLowerCase());
    const baseline = /^(?:baseline|before|pre|pre[- ]?treatment)$/;
    const follow = /^(?:after|post|post[- ]?treatment|follow[- ]?up)$/;
    if (
      visits.length === 2 &&
      labels.some((v) => baseline.test(v)) &&
      labels.some((v) => follow.test(v))
    ) {
      return baseline.test(labels[0]) ? visits.slice() : [visits[1], visits[0]];
    }
    if (labels.every((v) => /^\d{4}-\d{2}-\d{2}$/.test(v))) return visits.slice().sort();
    // Order only a single explicit scale. Mixed scales or unrecognized stages need confirmation.
    const units = {
      day: 'day',
      d: 'day',
      week: 'week',
      wk: 'week',
      w: 'week',
      month: 'month',
      mo: 'month',
      visit: 'visit',
      v: 'visit',
      cycle: 'cycle',
    };
    const unitSet = new Set();
    const times = [];
    for (const label of labels) {
      if (baseline.test(label)) {
        times.push(0);
        continue;
      }
      const parts = /^(day|d|week|wk|w|month|mo|visit|v|cycle)?\s*([+-]?\d+(?:\.\d+)?)$/.exec(label);
      if (!parts) return null;
      unitSet.add(units[parts[1]] || 'number');
      times.push(Number(parts[2]));
    }
    if (unitSet.size !== 1 || new Set(times).size !== times.length) return null;
    return visits
      .map((visit, index) => ({ visit, time: times[index] }))
      .sort((a, b) => a.time - b.time)
      .map((point) => point.visit);
  }

  function reviewVisitOrder(study, keepOnCancel = false) {
    const visits = study.visits.slice();
    if (visits.length < 2) return Promise.resolve(visits);
    const prior = document.activeElement;
    const dialog = el('dialog', 'local-mapping lm-visit-dialog');
    dialog.setAttribute('aria-labelledby', 'lm-visit-heading');
    const header = el('div', 'lm-header');
    const heading = el('h2', '', 'Visit order');
    heading.id = 'lm-visit-heading';
    const title = el('div');
    title.append(el('p', 'lm-eyebrow', study.title), heading);
    header.append(title);
    const body = el('div', 'lm-body');
    const label = el('label', 'lm-field');
    label.append(el('span', '', 'Earliest to latest'));
    const box = el('select');
    box.size = Math.min(7, Math.max(3, visits.length));
    box.id = 'lm-ordered-visits';
    label.htmlFor = box.id;
    const announce = el('p', 'lm-order-status');
    announce.setAttribute('role', 'status');
    const actions = el('div', 'lm-order-actions');
    const up = makeButton('↑ Earlier', 'lm-action', () => move(-1));
    const down = makeButton('↓ Later', 'lm-action', () => move(1));
    actions.append(up, down);
    visits.forEach((visit) => box.append(new Option(visit, visit)));
    function updateButtons() {
      up.disabled = box.selectedIndex <= 0;
      down.disabled = box.selectedIndex < 0 || box.selectedIndex >= box.length - 1;
    }
    function move(direction) {
      const index = box.selectedIndex;
      const target = index + direction;
      if (target < 0 || target >= box.length || index < 0) return;
      const option = box.options[index];
      box.insertBefore(option, direction < 0 ? box.options[target] : box.options[target].nextSibling);
      box.selectedIndex = target;
      announce.textContent = option.label + ', position ' + (target + 1) + ' of ' + box.length + '.';
      updateButtons();
      if (up.disabled || down.disabled) box.focus();
    }
    box.selectedIndex = 0;
    box.addEventListener('change', updateButtons);
    updateButtons();
    label.append(box);
    body.append(
      label,
      actions,
      el(
        'p',
        'lm-note',
        'This order sets the trajectory axis. Choose the comparison visits in the sheet.',
      ),
      announce,
    );
    const foot = el('div', 'lm-foot');
    const cancel = makeButton(keepOnCancel ? 'Keep file order' : 'Cancel', 'lm-action', () =>
      dialog.close(),
    );
    const confirm = makeButton('Apply order', 'lm-action lm-primary', () => dialog.close('apply'));
    foot.append(cancel, confirm);
    dialog.append(header, body, foot);
    document.body.append(dialog);
    return new Promise((resolve) => {
      dialog.addEventListener(
        'close',
        () => {
          const chosen =
            dialog.returnValue === 'apply'
              ? Array.from(box.options, (option) => option.value)
              : keepOnCancel
                ? visits
                : null;
          dialog.remove();
          if (prior?.isConnected) prior.focus({ preventScroll: true });
          resolve(chosen);
        },
        { once: true },
      );
      dialog.showModal();
      box.focus();
    });
  }

  async function openCSV(study, onLoaded) {
    const visits = inferVisitOrder(study.visits) || (await reviewVisitOrder(study));
    if (!visits) return false;
    onLoaded({ ...study, visits });
    return true;
  }

  function openMapping(onLoaded) {
    if (!available()) return Promise.resolve(null);
    if (active) return active.promise;
    const focusBeforeOpen = document.activeElement;
    const controller = new AbortController();
    const dialog = el('dialog', 'local-mapping');
    dialog.setAttribute('aria-labelledby', 'lm-heading');
    const form = el('form');
    const header = el('div', 'lm-header');
    const title = el('div');
    const heading = el('h2', '', 'Map study fields');
    heading.id = 'lm-heading';
    title.append(el('p', 'lm-eyebrow', 'Local H5AD'), heading);
    const dismiss = makeButton('×', 'lm-dismiss', () => dialog.close());
    dismiss.setAttribute('aria-label', 'Close mapping');
    header.append(title, dismiss);
    const body = el('div', 'lm-body'),
      foot = el('div', 'lm-foot');
    const errorBox = el('p', 'lm-error');
    errorBox.hidden = true;
    errorBox.setAttribute('role', 'alert');
    const cancel = makeButton('Cancel', 'lm-action', () => dialog.close());
    const submit = el('button', 'lm-action lm-primary', 'Open study');
    submit.type = 'submit';
    submit.disabled = true;
    foot.append(cancel, submit);
    form.append(header, body, foot);
    dialog.append(form);
    document.body.append(dialog);
    const status = el('div', 'lm-progress');
    status.setAttribute('role', 'status');
    const spinner = el('span', 'lm-spinner');
    spinner.setAttribute('aria-hidden', 'true');
    status.append(spinner, el('span', '', 'Reading study columns…'));
    body.append(status);

    let finishPromise,
      preparedStudy = null,
      settled = false;
    const promise = new Promise((resolve) => {
      finishPromise = resolve;
    });
    active = { promise };
    dialog.addEventListener(
      'close',
      async () => {
        if (settled) return;
        settled = true;
        controller.abort();
        dialog.remove();
        active = null;
        if (focusBeforeOpen?.isConnected) focusBeforeOpen.focus({ preventScroll: true });
        if (preparedStudy && !preparedStudy.orderedVisits) {
          preparedStudy.visits =
            inferVisitOrder(preparedStudy.visits) || (await reviewVisitOrder(preparedStudy, true));
        }
        finishPromise(preparedStudy);
        if (preparedStudy && typeof onLoaded === 'function') onLoaded(preparedStudy);
      },
      { once: true },
    );
    dialog.showModal();

    function showError(message) {
      errorBox.textContent = message;
      errorBox.hidden = false;
    }

    function setUp(schema) {
      if (!Array.isArray(schema?.columns) || !schema.columns.every((c) => typeof c?.name === 'string')) {
        throw new Error('The file did not provide a readable observation-column list.');
      }
      const suggested = schema.suggested || {};
      const columns = [...schema.columns].sort((a, b) =>
        a.name.localeCompare(b.name, undefined, { numeric: true }),
      );
      const numeric = columns.filter((c) => c.numeric).map((c) => c.name);
      const selected = new Set(
        (Array.isArray(suggested.scores) ? suggested.scores : [])
          .filter((s) => numeric.includes(s))
          .slice(0, scoreLimit),
      );
      const file = el('div', 'lm-file'),
        fileDetails = el('div');
      const cellCount = Number.isFinite(schema.nCells)
        ? schema.nCells.toLocaleString() + ' cells · '
        : '';
      fileDetails.append(
        el('b', '', schema.filename || 'Local study'),
        el('small', '', cellCount + columns.length + ' observation columns'),
      );
      const mark = el('span', 'lm-file-mark', 'H5');
      mark.setAttribute('aria-hidden', 'true');
      file.append(mark, fileDetails);
      const designSection = el('section', 'lm-section');
      designSection.append(el('h3', '', 'Study design'));
      const grid = el('div', 'lm-fields');
      const selectors = {};
      for (const role of roles) {
        const label = el('label', 'lm-field');
        const select = el('select');
        select.name = role.key;
        select.setAttribute('aria-label', role.label);
        select.append(new Option(role.empty, ''));
        columns.forEach((c) => select.append(new Option(c.name, c.name)));
        const suggestion = suggested[role.key];
        if (columns.some((c) => c.name === suggestion)) select.value = suggestion;
        selectors[role.key] = select;
        label.append(el('span', '', role.label), select);
        grid.append(label);
      }
      designSection.append(grid);

      const scoreSection = el('section', 'lm-section');
      const scoreHead = el('div', 'lm-outcome-head'),
        count = el('span', 'lm-count');
      scoreHead.append(el('h3', '', 'Outcome scores'), count);
      const search = el('input');
      search.type = 'search';
      search.placeholder = 'Filter numeric columns';
      search.setAttribute('aria-label', 'Filter numeric columns');
      const scoreList = el('div', 'lm-scores');
      scoreList.setAttribute('role', 'group');
      scoreList.setAttribute('aria-label', 'Outcome scores');
      const note = el(
        'p',
        'lm-note',
        'Choose up to 12 measurements across observation scores and genes.',
      );
      const typeOverrides = new Map();
      const geneBox = el('input');
      geneBox.type = 'text';
      geneBox.placeholder = 'Exact variable names, separated by commas';
      geneBox.setAttribute('aria-label', 'Genes from X');
      const geneLabel = el('label', 'lm-field lm-gene-input');
      geneLabel.append(el('span', '', 'Genes from X'), geneBox);
      const geneNote = el(
        'p',
        'lm-note',
        'Search var_names by typing the first two letters. Expression uses the values stored in X on their original scale.',
      );
      const suggestions = el('div', 'lm-gene-matches');
      suggestions.setAttribute('aria-label', 'Matching genes');
      suggestions.setAttribute('role', 'group');
      let lookupTimer, lookupRequest;
      function clearLookup() {
        clearTimeout(lookupTimer);
        lookupRequest?.abort();
        suggestions.replaceChildren();
      }
      function suggestGenes() {
        clearLookup();
        const parts = geneBox.value.split(',');
        const query = parts.pop().trim();
        if (query.length < 2 || query.length > 160) return;
        const selectedGenes = new Set(parts.map((part) => part.trim()));
        lookupRequest = new AbortController();
        const signal = lookupRequest.signal;
        lookupTimer = setTimeout(async () => {
          try {
            const result = await request('/local/genes', { query }, signal);
            if (signal.aborted || !dialog.open) return;
            for (const name of result.genes || []) {
              if (selectedGenes.has(name)) continue;
              const choose = makeButton(name, 'lm-gene-match', () => {
                clearLookup();
                geneBox.value = [...parts.map((part) => part.trim()).filter(Boolean), name, ''].join(
                  ', ',
                );
                updateState();
                geneBox.focus();
              });
              suggestions.append(choose);
            }
            if (!suggestions.childElementCount) {
              suggestions.append(el('span', 'lm-note', 'No additional matching variables.'));
            }
          } catch (error) {
            if (!signal.aborted && dialog.open) suggestions.append(el('span', 'lm-note', error.message));
          }
        }, 180);
      }
      controller.signal.addEventListener('abort', clearLookup, { once: true });
      geneBox.addEventListener('input', suggestGenes);
      geneBox.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown' && suggestions.querySelector('button')) {
          event.preventDefault();
          suggestions.querySelector('button').focus();
        } else if (event.key === 'Escape' && suggestions.childElementCount) {
          event.preventDefault();
          event.stopPropagation();
          clearLookup();
        }
      });
      const inputGenes = () =>
        geneBox.value
          .split(',')
          .map((gene) => gene.trim())
          .filter(Boolean);
      const validation = el('p', 'lm-note');
      validation.setAttribute('role', 'status');
      scoreSection.append(scoreHead, search, scoreList);
      if (schema.geneInput) scoreSection.append(geneLabel, suggestions, geneNote);
      scoreSection.append(note, validation);
      body.replaceChildren(file, designSection, scoreSection, errorBox);

      const mappedRoles = () => roles.map((r) => selectors[r.key].value).filter(Boolean);
      function updateState() {
        const mapped = mappedRoles();
        const overlap = new Set(mapped).size !== mapped.length;
        const genes = inputGenes();
        const total = selected.size + genes.length;
        const problem = !selectors.participant.value
          ? 'Choose the participant ID column.'
          : overlap
            ? 'Use a different column for each study role.'
            : new Set(genes).size !== genes.length
              ? 'Enter each X variable once.'
              : total === 0
                ? 'Choose at least one numeric score or X variable.'
                : total > scoreLimit
                  ? 'Choose up to 12 measurements in total.'
                  : '';
        submit.disabled = Boolean(problem);
        validation.textContent = problem;
        validation.hidden = !problem;
        count.textContent = total + ' selected';
        const locked = new Set(mapped);
        scoreList.querySelectorAll('input').forEach((box) => {
          box.disabled = locked.has(box.value) || (!box.checked && total >= scoreLimit);
        });
      }
      function drawScores() {
        const needle = search.value.trim().toLocaleLowerCase();
        const matches = numeric.filter((n) => n.toLocaleLowerCase().includes(needle));
        const fragment = document.createDocumentFragment();
        for (const name of matches) {
          const row = el('label', 'lm-score'),
            check = el('input');
          check.type = 'checkbox';
          check.value = name;
          check.checked = selected.has(name);
          check.addEventListener('change', () => {
            if (check.checked) selected.add(name);
            else selected.delete(name);
            errorBox.hidden = true;
            updateState();
          });
          row.append(check, el('span', '', name));
          const entry = el('div', 'lm-score-row');
          const type = el('select', 'lm-score-type');
          type.setAttribute('aria-label', 'Measurement type for ' + name);
          const implied = /^(gene|pathway|module):/i.exec(name)?.[1].toLowerCase() || 'score';
          for (const [key, label] of [
            ['score', 'Other score'],
            ['module', 'Module score'],
            ['pathway', 'Pathway score'],
            ['gene', 'Gene expression'],
          ]) {
            type.add(new Option(label, key));
          }
          type.value = typeOverrides.get(name) || implied;
          type.hidden = !check.checked;
          check.addEventListener('change', () => {
            type.hidden = !check.checked;
          });
          type.addEventListener('change', () => {
            if (type.value === implied) typeOverrides.delete(name);
            else typeOverrides.set(name, type.value);
          });
          entry.append(row, type);
          fragment.append(entry);
        }
        if (!matches.length)
          fragment.append(
            el(
              'p',
              'lm-empty',
              numeric.length ? 'No scores match this filter.' : 'No numeric columns found.',
            ),
          );
        scoreList.replaceChildren(fragment);
        updateState();
      }
      function roleChanged() {
        mappedRoles().forEach((name) => selected.delete(name));
        errorBox.hidden = true;
        drawScores();
      }
      Object.values(selectors).forEach((select) => select.addEventListener('change', roleChanged));
      search.addEventListener('input', drawScores);
      geneBox.addEventListener('input', updateState);
      roleChanged();
      selectors.participant.focus({ preventScroll: true });
      let sending = false;
      form.onsubmit = async (event) => {
        event.preventDefault();
        if (sending || submit.disabled) return;
        sending = true;
        clearLookup();
        const payload = Object.fromEntries(roles.map((r) => [r.key, selectors[r.key].value || null]));
        payload.scores = [...selected];
        const genes = inputGenes();
        if (genes.length) payload.genes = genes;
        const scoreKinds = [...typeOverrides].filter(([name]) => selected.has(name));
        if (scoreKinds.length) payload.scoreKinds = Object.fromEntries(scoreKinds);
        errorBox.hidden = true;
        submit.disabled = true;
        submit.textContent = 'Opening…';
        for (const control of body.querySelectorAll('input,select')) control.disabled = true;
        try {
          const result = await request('/local/study', payload, controller.signal);
          if (!dialog.open || controller.signal.aborted) return;
          if (
            !result?.studyId ||
            !Array.isArray(result.visits) ||
            !result.visits.length ||
            !Array.isArray(result.participants) ||
            !Array.isArray(result.features) ||
            !Array.isArray(result.observations)
          ) {
            throw new Error('The local workspace returned an incomplete study.');
          }
          preparedStudy = result;
          dialog.close();
        } catch (error) {
          if (controller.signal.aborted || !dialog.open) return;
          for (const control of body.querySelectorAll('input,select')) control.disabled = false;
          submit.textContent = 'Open study';
          updateState();
          showError(error.message);
          sending = false;
        }
      };
    }

    async function load() {
      errorBox.hidden = true;
      submit.hidden = false;
      submit.disabled = true;
      body.replaceChildren(status);
      try {
        const schema = await request('/local/schema', undefined, controller.signal);
        if (!controller.signal.aborted && dialog.open) setUp(schema);
      } catch (error) {
        if (controller.signal.aborted || !dialog.open) return;
        body.replaceChildren(errorBox);
        showError(error.message);
        submit.hidden = true;
        const retry = makeButton('Retry', 'lm-action lm-primary', () => {
          retry.remove();
          load();
        });
        foot.append(retry);
        retry.focus();
      }
    }
    load();
    return promise;
  }
  window.localStudyClient = Object.freeze({ available, openMapping, openCSV, reviewVisitOrder, fit });
})();
