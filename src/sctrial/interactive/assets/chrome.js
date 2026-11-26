(() => {
  'use strict';
  const get = (id) => document.getElementById(id);
  const sidebar = get('projectSidebar');
  const trigger = get('showSidebar');
  const shade = get('navigatorShade');
  const compact = matchMedia('(max-width: 850px)');
  let lastStudy;
  let lastOrder;

  function toggleSidebar(open, returnFocus = true) {
    const wasOpen = trigger.getAttribute('aria-expanded') === 'true';
    trigger.setAttribute('aria-expanded', String(open));
    sidebar.classList.toggle('is-open', open);
    shade.hidden = !open;
    sidebar.inert = compact.matches && !open;
    document.querySelector('.workspace-body').inert = open;
    if (open) get('closeSidebar').focus();
    else if (wasOpen && returnFocus && compact.matches) trigger.focus();
  }
  trigger.onclick = () => toggleSidebar(true);
  get('closeSidebar').onclick = () => toggleSidebar(false);
  shade.onclick = () => toggleSidebar(false);
  sidebar.addEventListener(
    'click',
    (event) => {
      if (!compact.matches) return;
      if (event.target.closest('[data-view], [data-action], [data-sheet-id]')) toggleSidebar(false);
    },
    true,
  );
  document.addEventListener('keydown', (event) => {
    if (
      event.key === 'Escape' &&
      !document.querySelector('dialog[open]') &&
      sidebar.classList.contains('is-open')
    )
      toggleSidebar(false);
  });
  compact.addEventListener('change', () => toggleSidebar(false));
  toggleSidebar(false);

  // Count a participant once per visit across all recorded populations.
  function setStudy(study) {
    const order = JSON.stringify(study.visits);
    if (lastStudy === study && lastOrder === order) return;
    lastStudy = study;
    lastOrder = order;
    get('mobileSource').textContent =
      study.source === 'h5ad' ? 'Local H5AD' : study.study ? 'Synthetic study' : 'Local CSV';
    const arms = [...new Set(study.participants.map((person) => person.arm))];
    const counts = new Map(
      arms.map((arm) => [arm, new Map(study.visits.map((visit) => [visit, new Set()]))]),
    );
    const totals = new Map(arms.map((arm) => [arm, 0]));
    for (const row of study.participants) totals.set(row.arm, totals.get(row.arm) + 1);
    for (const row of study.observations) {
      if (row.cells > 0) counts.get(row.arm)?.get(row.visit)?.add(row.id);
    }
    const stage = get('studyMap');
    stage.replaceChildren();
    const format = (tag, label) => {
      const el = document.createElement(tag);
      el.textContent = label;
      return el;
    };
    const table = document.createElement('table');
    table.setAttribute(
      'aria-label',
      'Participants sampled per group and visit across all populations',
    );
    const header = table.createTHead().insertRow();
    header.append(format('th', 'Visit'));
    for (const arm of arms) {
      const th = format('th', arm);
      th.scope = 'col';
      th.title = arm;
      header.append(th);
    }
    const body = table.createTBody();
    for (const visit of study.visits) {
      const tr = body.insertRow();
      const th = format('th', visit);
      th.scope = 'row';
      th.title = visit;
      tr.append(th);
      for (const arm of arms) {
        const td = tr.insertCell();
        const n = counts.get(arm).get(visit).size;
        const bar = document.createElement('i');
        td.append(format('span', n), bar);
        td.style.setProperty('--coverage', (100 * n) / totals.get(arm) + '%');
      }
    }
    stage.append(table);
  }
  window.SctrialChrome = { setStudy, closeNavigator: () => toggleSidebar(false, false) };
})();
