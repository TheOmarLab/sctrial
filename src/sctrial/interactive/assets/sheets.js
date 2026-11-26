(() => {
  'use strict';

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  const icons = {
    sheet: 'M4.5 3.5h10v12h-10z M7 12V9m2.5 3V6.5M12 12V8',
    rename: 'm4 12.5-.6 3.1 3.1-.6L15 6.5 12.5 4z M11 5.5 13.5 8',
    duplicate: 'M7 7h8v8H7z M4.5 12H3.3V3.3H12v1.2',
    remove: 'M3.5 5h11M7 5V3h4v2 M5 5l.7 10h6.6L13 5 M8 7.5v5 M10 7.5v5',
  };

  function glyph(name) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 18 18');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'sheet-icon');
    if (name === 'more') {
      [4, 9, 14].forEach((x) => {
        const dot = document.createElementNS(ns, 'circle');
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', '9');
        dot.setAttribute('r', '1.2');
        dot.setAttribute('fill', 'currentColor');
        svg.append(dot);
      });
    } else {
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', icons[name]);
      path.setAttribute('stroke', 'currentColor');
      path.setAttribute('stroke-width', '1.45');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      svg.append(path);
    }
    return svg;
  }

  function button(className, label, icon) {
    const control = node('button', className);
    control.type = 'button';
    control.setAttribute('aria-label', label);
    if (icon) control.append(glyph(icon));
    return control;
  }

  window.createAnalysisSheets = function ({ getSpec, getStudy, onOpen, onRename = () => {} }) {
    const list = document.getElementById('sheetList');
    const addButton = document.getElementById('newSheet');
    let papers = [];
    let active = null;
    let defaults;
    let nextId = 1;
    let overview = true;
    let removed = null;
    let menu = null;
    let menuTrigger = null;
    let renaming = null;

    function reset(label = 'Primary analysis') {
      dismissMenu();
      renaming = null;
      removed = null;
      defaults = { ...getSpec() };
      nextId = 1;
      active = { id: nextId++, title: label, spec: { ...defaults } };
      papers = [active];
      overview = true;
      render();
    }

    function open(paper) {
      const navigator = list.closest('[inert], #projectSidebar');
      window.SctrialChrome?.closeNavigator();
      if (paper === active && overview) {
        if (navigator?.inert) document.getElementById('main')?.focus({ preventScroll: true });
        return;
      }
      if (active) active.spec = { ...getSpec() };
      active = paper;
      overview = true;
      onOpen({ ...paper.spec });
      render();
      const target = navigator?.inert
        ? document.getElementById('main')
        : list.querySelector(`[data-sheet-id="${paper.id}"]`);
      target?.focus({ preventScroll: true });
    }

    function unique(base, names = new Set(papers.map((paper) => paper.title))) {
      let result = base.slice(0, 80);
      let number = 2;
      while (names.has(result)) {
        const suffix = ' (' + number++ + ')';
        result = base.slice(0, 80 - suffix.length) + suffix;
      }
      return result;
    }

    function create(original) {
      if (!active) return;
      active.spec = { ...getSpec() };
      const paper = {
        id: nextId,
        title: unique(original ? original.title + ' copy' : 'Analysis ' + nextId),
        spec: { ...(original ? original.spec : defaults) },
      };
      nextId++;
      const index = original ? papers.indexOf(original) + 1 : papers.length;
      papers.splice(index, 0, paper);
      open(paper);
      if (!list.closest('[inert]'))
        list.querySelector(`[data-sheet-id="${paper.id}"]`)?.scrollIntoView({ block: 'nearest' });
    }

    function remove(paper) {
      if (papers.length === 1) return;
      const index = papers.indexOf(paper);
      if (active === paper) paper.spec = { ...getSpec() };
      removed = { index, paper };
      papers.splice(index, 1);
      if (active === paper) open(papers[Math.min(index, papers.length - 1)]);
      else render();
      list.querySelector('.sheet-undo')?.focus();
    }

    function undoRemove() {
      if (!removed) return;
      const { index, paper } = removed;
      removed = null;
      papers.splice(Math.min(index, papers.length), 0, paper);
      open(paper);
    }

    function dismissMenu(restoreFocus = false) {
      menu?.remove();
      menu = null;
      if (menuTrigger) {
        menuTrigger.setAttribute('aria-expanded', 'false');
        if (restoreFocus) menuTrigger.focus();
      }
      menuTrigger = null;
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('focusin', focusOutside);
      window.removeEventListener('resize', placeMenu);
      document.removeEventListener('scroll', placeMenu, true);
    }

    function outside(event) {
      if (!menu?.contains(event.target) && !menuTrigger?.contains(event.target)) dismissMenu();
    }

    function focusOutside(event) {
      if (!menu?.contains(event.target) && event.target !== menuTrigger) dismissMenu();
    }

    function placeMenu() {
      if (!menu || !menuTrigger) return;
      const box = menuTrigger.getBoundingClientRect();
      const width = menu.offsetWidth;
      const height = menu.offsetHeight;
      const left = Math.max(6, Math.min(box.right - width, window.innerWidth - width - 6));
      const top =
        box.bottom + height + 6 <= window.innerHeight ? box.bottom + 3 : box.top - height - 3;
      menu.style.left = left + 'px';
      menu.style.top = Math.max(4, top) + 'px';
    }

    function showMenu(paper, trigger) {
      const same = trigger === menuTrigger;
      dismissMenu();
      if (same) return;
      menuTrigger = trigger;
      menu = node('div', 'sheet-menu');
      menu.setAttribute('role', 'menu');
      menu.setAttribute('aria-label', 'Actions for ' + paper.title);
      trigger.setAttribute('aria-expanded', 'true');
      const items = [
        ['rename', 'Rename', () => editName(paper)],
        ['duplicate', 'Duplicate', () => create(paper)],
        ['remove', 'Remove', () => remove(paper)],
      ];
      items.forEach(([icon, label, action]) => {
        const item = button('sheet-menu-item', label, icon);
        item.setAttribute('role', 'menuitem');
        item.tabIndex = -1;
        item.disabled = icon === 'remove' && papers.length === 1;
        item.append(document.createTextNode(label));
        item.addEventListener('click', () => {
          dismissMenu();
          action();
        });
        menu.append(item);
      });
      menu.addEventListener('keydown', (event) => {
        const enabled = [...menu.querySelectorAll('button:enabled')];
        const index = enabled.indexOf(document.activeElement);
        switch (event.key) {
          case 'ArrowDown':
            enabled[(index + 1) % enabled.length].focus();
            break;
          case 'ArrowUp':
            enabled[(index + enabled.length - 1) % enabled.length].focus();
            break;
          case 'Home':
            enabled[0].focus();
            break;
          case 'End':
            enabled[enabled.length - 1].focus();
            break;
          case 'Escape':
            dismissMenu(true);
            break;
          case 'Tab':
            dismissMenu(true);
            return;
          default:
            return;
        }
        event.preventDefault();
        event.stopPropagation();
      });
      document.body.append(menu);
      placeMenu();
      menu.firstElementChild.focus();
      document.addEventListener('pointerdown', outside, true);
      document.addEventListener('focusin', focusOutside);
      window.addEventListener('resize', placeMenu);
      document.addEventListener('scroll', placeMenu, true);
    }

    function editName(paper) {
      renaming = paper;
      render();
      const input = list.querySelector('input');
      input.focus();
      input.select();
    }

    function nameEditor(paper) {
      const form = node('form', 'sheet-name-form');
      const input = node('input', 'sheet-name');
      input.value = paper.title;
      input.maxLength = 80;
      input.autocomplete = 'off';
      input.setAttribute('aria-label', 'Analysis name');
      const finish = (commit, focus) => {
        if (renaming !== paper) return;
        renaming = null;
        const title = input.value
          .trim()
          .replace(/[\u0000-\u001f\u007f]/g, ' ')
          .trim();
        if (commit && title) paper.title = title;
        form.parentElement?.replaceWith(sheetRow(paper));
        markActive();
        onRename();
        if (focus) list.querySelector(`[data-sheet-id="${paper.id}"]`)?.focus();
      };
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        finish(true, true);
      });
      input.addEventListener('blur', () => finish(true, false));
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          finish(false, true);
        }
      });
      form.append(input);
      return form;
    }

    function markActive() {
      for (const item of list.querySelectorAll('[data-sheet-id]')) {
        const current = overview && item.dataset.sheetId === String(active?.id);
        item.toggleAttribute('data-active', current);
        if (current) item.setAttribute('aria-current', 'page');
        else item.removeAttribute('aria-current');
      }
    }

    function sheetRow(paper) {
      const row = node('div', 'sheet-item');
      if (renaming === paper) {
        row.append(nameEditor(paper));
        return row;
      }
      const select = button('sheet-link', paper.title, 'sheet');
      select.dataset.sheetId = paper.id;
      select.title = paper.title;
      select.append(node('span', 'sheet-label', paper.title));
      select.addEventListener('click', () => open(paper));
      const more = button('sheet-more', 'Actions for ' + paper.title, 'more');
      more.title = 'Analysis actions';
      more.setAttribute('aria-haspopup', 'menu');
      more.setAttribute('aria-expanded', 'false');
      more.addEventListener('click', () => showMenu(paper, more));
      row.append(select, more);
      return row;
    }

    function render() {
      dismissMenu();
      list.replaceChildren(...papers.map(sheetRow));
      if (removed) {
        const undoButton = button('sheet-undo', 'Undo removal of ' + removed.paper.title);
        undoButton.textContent = 'Undo removal';
        undoButton.addEventListener('click', undoRemove);
        list.append(undoButton);
      }
      markActive();
    }

    function studySchema(study) {
      const sorted = (values) => [...values].sort();
      const mapping = study.mapping
        ? Object.fromEntries(
            ['participant', 'arm', 'visit', 'population'].map((role) => [
              role,
              study.mapping[role] ?? null,
            ]),
          )
        : null;
      return {
        name: study.name,
        source: study.source,
        arms: sorted(study.arms),
        visits: sorted(study.visits),
        populations: sorted(study.populations),
        features: study.features
          .map(({ id, label }) => ({ id, label }))
          .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)),
        mapping,
      };
    }

    function validatedSpec(settings, schema, study) {
      if (!settings || typeof settings !== 'object') throw Error('Missing comparison settings.');
      const { design, arm, baseline, visit, population, feature, mode, visitOrder } = settings;
      if (!study.designs.includes(design)) throw Error('Unsupported study design.');
      if (!schema.arms.includes(arm)) throw Error('Unknown group.');
      if (!schema.populations.includes(population)) throw Error('Unknown cell population.');
      if (!schema.features.some((f) => f.id === feature)) throw Error('Unknown outcome.');
      if (!schema.visits.includes(baseline) || !schema.visits.includes(visit))
        throw Error('Unknown comparison visit.');
      if (study.fixedBaseline && baseline !== study.fixedBaseline)
        throw Error('The reference visit does not match the fitted study.');
      if (['did', 'paired'].includes(design) && baseline === visit)
        throw Error('Select two distinct visits for a paired comparison.');
      if (!['comparison', 'trajectory', 'changes'].includes(mode))
        throw Error('Unknown analysis view.');
      if (
        !Array.isArray(visitOrder) ||
        visitOrder.length !== schema.visits.length ||
        !visitOrder.every((v) => schema.visits.includes(v)) ||
        new Set(visitOrder).size !== visitOrder.length
      )
        throw Error('Visit order must include each study visit exactly once.');
      if (study.fixedBaseline && !visitOrder.every((v, i) => v === study.visits[i]))
        throw Error('The visit order does not match the fitted study.');
      return {
        design,
        arm,
        baseline,
        visit,
        population,
        feature,
        visitOrder: [...visitOrder],
        mode,
      };
    }

    function exportWorkbook() {
      const study = getStudy();
      const schema = studySchema(study);
      const contents = papers.map((paper) => {
        const spec = paper === active ? getSpec() : paper.spec;
        return {
          title: paper.title,
          comparison: validatedSpec(spec, schema, study),
        };
      });
      return (
        JSON.stringify(
          {
            format: 'sctrial-workbook',
            version: 1,
            study: schema,
            sheets: contents,
            activeSheet: papers.indexOf(active) + 1,
          },
          null,
          2,
        ) + '\n'
      );
    }

    function importWorkbook(text) {
      let file;
      try {
        file = JSON.parse(text.replace(/^\uFEFF/, ''));
      } catch {
        throw Error('Choose a valid workbook JSON file.');
      }
      if (file?.format !== 'sctrial-workbook' || file.version !== 1)
        throw Error('Choose a version 1 sctrial analysis workbook.');
      const study = getStudy();
      const schema = studySchema(study);
      const saved = file.study;
      if (!saved || saved.name !== schema.name || saved.source !== schema.source)
        throw Error('Open the study that was used to save this workbook.');
      const sameList = (a, b) =>
        Array.isArray(a) && a.length === b.length && a.every((value, index) => value === b[index]);
      if (
        !sameList(saved.arms, schema.arms) ||
        !sameList(saved.populations, schema.populations) ||
        !sameList(saved.visits, schema.visits)
      )
        throw Error('The groups, visits, or cell populations differ from the saved study.');
      if (
        !Array.isArray(saved.features) ||
        saved.features.length !== schema.features.length ||
        !schema.features.every(
          (f, i) => saved.features[i]?.id === f.id && saved.features[i]?.label === f.label,
        )
      )
        throw Error(
          'The outcome columns differ from the saved study. Check their names and mapping.',
        );
      if (
        schema.mapping === null
          ? saved.mapping !== null
          : !saved.mapping ||
            !Object.keys(schema.mapping).every((key) => saved.mapping[key] === schema.mapping[key])
      )
        throw Error('The H5AD column mapping differs from the saved study.');
      if (
        !Array.isArray(file.sheets) ||
        !file.sheets.length ||
        !Number.isInteger(file.activeSheet) ||
        file.activeSheet < 1 ||
        file.activeSheet > file.sheets.length
      )
        throw Error('The workbook has no valid active sheet.');

      const names = new Set(papers.map((paper) => paper.title));
      const imported = file.sheets.map((sheet, i) => {
        if (
          !sheet ||
          typeof sheet.title !== 'string' ||
          !sheet.title.trim() ||
          sheet.title.length > 80 ||
          /[\u0000-\u001f\u007f]/.test(sheet.title)
        )
          throw Error('Sheet ' + (i + 1) + ' has an invalid name.');
        let spec;
        try {
          spec = validatedSpec(sheet.comparison, schema, study);
        } catch (error) {
          throw Error('Sheet ' + (i + 1) + ': ' + error.message);
        }
        const title = unique(sheet.title.trim(), names);
        names.add(title);
        return { id: nextId + i, title, spec };
      });
      nextId += imported.length;
      papers = papers.concat(imported);
      open(imported[file.activeSheet - 1]);
      return imported.length;
    }

    addButton.addEventListener('click', () => create());
    return Object.freeze({
      reset,
      exportWorkbook,
      importWorkbook,
      title: () => active?.title || '',
      setVisible(value) {
        if (overview === value) return;
        overview = value;
        markActive();
      },
    });
  };
})();
