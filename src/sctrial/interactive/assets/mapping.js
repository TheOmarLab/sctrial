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
      () => {
        if (settled) return;
        settled = true;
        controller.abort();
        dialog.remove();
        active = null;
        if (focusBeforeOpen?.isConnected) focusBeforeOpen.focus({ preventScroll: true });
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
      if (
        !Array.isArray(schema?.columns) ||
        !schema.columns.every((c) => typeof c?.name === 'string')
      ) {
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
      const note = el('p', 'lm-note', 'Select up to 12 numeric scores from the observation table.');
      const validation = el('p', 'lm-note');
      validation.setAttribute('role', 'status');
      scoreSection.append(scoreHead, search, scoreList, note, validation);
      body.replaceChildren(file, designSection, scoreSection, errorBox);

      const mappedRoles = () => roles.map((r) => selectors[r.key].value).filter(Boolean);
      function updateState() {
        const mapped = mappedRoles();
        const overlap = new Set(mapped).size !== mapped.length;
        const problem = !selectors.participant.value
          ? 'Choose the participant ID column.'
          : overlap
            ? 'Use a different column for each study role.'
            : !numeric.length
              ? 'This file has no numeric observation scores.'
              : selected.size === 0
                ? 'Choose at least one outcome score.'
                : '';
        submit.disabled = Boolean(problem);
        validation.textContent = problem;
        validation.hidden = !problem;
        count.textContent = selected.size + ' selected';
        const locked = new Set(mapped);
        scoreList.querySelectorAll('input').forEach((box) => {
          box.disabled = locked.has(box.value) || (!box.checked && selected.size >= scoreLimit);
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
          fragment.append(row);
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
      roleChanged();
      selectors.participant.focus({ preventScroll: true });
      let sending = false;
      form.onsubmit = async (event) => {
        event.preventDefault();
        if (sending || submit.disabled) return;
        sending = true;
        const payload = Object.fromEntries(roles.map((r) => [r.key, selectors[r.key].value || null]));
        payload.scores = [...selected];
        errorBox.hidden = true;
        submit.disabled = true;
        submit.textContent = 'Opening…';
        for (const control of body.querySelectorAll('input,select')) control.disabled = true;
        try {
          const result = await request('/local/study', payload, controller.signal);
          if (!dialog.open || controller.signal.aborted) return;
          if (
            !result?.studyId ||
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
  window.localStudyClient = Object.freeze({ available, openMapping, fit });
})();
