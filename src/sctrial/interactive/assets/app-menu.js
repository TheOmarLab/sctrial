(() => {
  'use strict';

  const menus = {
    File: [
      { label: 'Open study', action: 'open' },
      { label: 'Save workbook', action: 'saveWorkbook' },
      { label: 'Export results…', action: 'export' },
      { label: 'Study information', action: 'info', divider: true },
    ],
    Analysis: [
      { label: 'Configure comparison…', action: 'configure' },
      { label: 'New analysis', newSheet: true },
      { label: 'Pin contrast', action: 'pin', divider: true },
      { label: 'Pinned contrasts', view: 'bench' },
    ],
    View: [
      { label: 'Cell populations', view: 'overview', mode: 'cells' },
      { label: 'Genes', view: 'overview', mode: 'genes' },
      { label: 'Pathways', view: 'overview', mode: 'pathways' },
      { label: 'Modules', view: 'overview', mode: 'modules' },
      { label: 'Time course', view: 'overview', mode: 'trajectory' },
      { label: 'Analysis plot', view: 'overview', mode: 'comparison', divider: true },
      { label: 'Participant measurements', view: 'overview', mode: 'measurements' },
      { label: 'Changes', view: 'overview', mode: 'changes' },
      { label: 'Participant register', view: 'participants', divider: true },
      { label: 'Results table', view: 'results' },
      { label: 'Find in study', action: 'search', shortcut: '/', divider: true },
    ],
  };
  const host = document.getElementById('appMenus');
  if (!host) return;
  const compact = window.matchMedia('(max-width: 639px)');
  const bar = document.createElement('div');
  bar.className = 'am-bar';
  bar.setAttribute('role', 'menubar');
  bar.setAttribute('aria-label', 'Application menus');
  const popup = document.createElement('div');
  popup.className = 'am-popup';
  popup.id = 'appMenuPopup';
  popup.setAttribute('role', 'menu');
  popup.hidden = true;
  host.replaceChildren(bar, popup);

  const categories = Object.keys(menus);
  const triggers = new Map();
  const state = { view: 'overview', mode: 'cells', longitudinal: true, canPin: true };
  let openName = '';
  let roving = compact.matches ? 'Menu' : categories[0];
  let items = [];
  let search = '';
  let lastKeyTime = 0;
  let hoveredTrigger = null;

  function visibleMenus() {
    return compact.matches ? ['Menu'] : categories;
  }

  function syncTriggers() {
    for (const [name, trigger] of triggers) {
      trigger.hidden = !visibleMenus().includes(name);
      trigger.tabIndex = name === roving ? 0 : -1;
      trigger.setAttribute('aria-expanded', String(openName === name));
    }
  }

  function close(restore = false) {
    if (!openName) return;
    const trigger = triggers.get(openName);
    openName = '';
    hoveredTrigger = null;
    items = [];
    popup.hidden = true;
    popup.replaceChildren();
    syncTriggers();
    if (restore) trigger.focus();
  }

  function activate(spec) {
    close(true);
    if (spec.newSheet) {
      document.getElementById('newSheet')?.click();
      return;
    }
    const detail = spec.action ? { action: spec.action } : { view: spec.view };
    if (spec.mode) detail.mode = spec.mode;
    document.dispatchEvent(new CustomEvent('sctrial:menu-command', { detail }));
  }

  function labelFor(spec) {
    return spec.mode === 'changes' && !state.longitudinal ? 'Distribution' : spec.label;
  }

  function updateItem({ control, label, spec }) {
    label.textContent = labelFor(spec);
    if (spec.view) {
      const current = spec.view === state.view && (!spec.mode || spec.mode === state.mode);
      control.setAttribute('aria-checked', String(current));
    }
    if (spec.action === 'pin') {
      control.setAttribute('aria-disabled', String(!state.canPin));
      control.title = state.canPin ? '' : 'A fitted contrast is required';
    }
  }

  function addGroup(category, mobile) {
    const group = document.createElement('div');
    group.className = 'am-group';
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', category);
    if (mobile) {
      const title = document.createElement('div');
      title.className = 'am-category';
      title.setAttribute('aria-hidden', 'true');
      title.textContent = category;
      group.append(title);
    }
    for (const spec of menus[category]) {
      if (spec.divider) {
        const divider = document.createElement('div');
        divider.className = 'am-divider';
        divider.setAttribute('role', 'separator');
        group.append(divider);
      }
      const control = document.createElement('button');
      control.type = 'button';
      control.tabIndex = -1;
      control.className = 'am-item';
      control.setAttribute('role', spec.view ? 'menuitemradio' : 'menuitem');
      // Keep the check in its own column so labels never move as the view changes.
      const mark = document.createElement('span');
      mark.className = 'am-check';
      mark.setAttribute('aria-hidden', 'true');
      const label = document.createElement('span');
      label.className = 'am-item-label';
      control.append(mark, label);
      if (spec.shortcut) {
        const key = document.createElement('kbd');
        key.setAttribute('aria-hidden', 'true');
        key.textContent = spec.shortcut;
        control.setAttribute('aria-keyshortcuts', spec.shortcut);
        control.append(key);
      }
      const item = { control, label, spec };
      updateItem(item);
      control.addEventListener('pointermove', (event) => {
        if (event.pointerType === 'mouse' && document.activeElement !== control) {
          control.focus({ preventScroll: true });
        }
      });
      control.addEventListener('click', () => {
        if (control.getAttribute('aria-disabled') !== 'true') activate(spec);
      });
      group.append(control);
      items.push(item);
    }
    popup.append(group);
  }

  function placePopup() {
    if (!openName) return;
    const anchor = triggers.get(openName).getBoundingClientRect();
    const gap = 5;
    const minLeft = 5;
    const width = popup.offsetWidth;
    const left = Math.max(minLeft, Math.min(anchor.left, window.innerWidth - width - minLeft));
    const top = Math.min(anchor.bottom + gap, window.innerHeight - gap);
    popup.style.left = left + 'px';
    popup.style.top = top + 'px';
    popup.style.maxHeight = Math.max(72, window.innerHeight - top - gap) + 'px';
  }

  function open(name, last = false) {
    if (!visibleMenus().includes(name)) return;
    openName = roving = name;
    hoveredTrigger = null;
    search = '';
    lastKeyTime = 0;
    items = [];
    popup.replaceChildren();
    const trigger = triggers.get(name);
    popup.setAttribute('aria-labelledby', trigger.id);
    if (name === 'Menu') categories.forEach((category) => addGroup(category, true));
    else addGroup(name, false);
    popup.hidden = false;
    syncTriggers();
    placePopup();
    items[last ? items.length - 1 : 0]?.control.focus({ preventScroll: true });
    popup.scrollTop = last ? popup.scrollHeight : 0;
  }

  function topKey(event, name) {
    if (event.altKey || event.metaKey || event.ctrlKey) return;
    const names = visibleMenus();
    let index = names.indexOf(name);
    switch (event.key) {
      case 'ArrowLeft':
        index = (index + names.length - 1) % names.length;
        break;
      case 'ArrowRight':
        index = (index + 1) % names.length;
        break;
      case 'Home':
        index = 0;
        break;
      case 'End':
        index = names.length - 1;
        break;
      case 'ArrowDown':
      case 'ArrowUp':
        event.preventDefault();
        open(name, event.key === 'ArrowUp');
        return;
      case 'Escape':
        close(true);
        return;
      default:
        return;
    }
    event.preventDefault();
    roving = names[index];
    syncTriggers();
    triggers.get(roving).focus();
    if (openName) open(roving);
  }

  for (const name of [...categories, 'Menu']) {
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'am-trigger';
    trigger.id = 'appMenu' + name;
    trigger.setAttribute('role', 'menuitem');
    trigger.setAttribute('aria-controls', popup.id);
    trigger.setAttribute('aria-haspopup', 'menu');
    trigger.textContent = name;
    trigger.addEventListener('click', (event) => {
      if (event.detail && hoveredTrigger === name) {
        hoveredTrigger = null;
        items[0]?.control.focus();
        return;
      }
      if (openName === name) close(true);
      else open(name);
    });
    trigger.addEventListener('pointerenter', (event) => {
      if (event.pointerType === 'mouse' && openName && openName !== name) {
        open(name);
        hoveredTrigger = name;
      }
    });
    trigger.addEventListener('keydown', (event) => topKey(event, name));
    triggers.set(name, trigger);
    bar.append(trigger);
  }

  popup.addEventListener('keydown', (event) => {
    if (!openName || event.altKey || event.metaKey || event.ctrlKey) return;
    const index = items.findIndex((item) => item.control === document.activeElement);
    let next = index;
    switch (event.key) {
      case 'ArrowDown':
        next = (index + 1) % items.length;
        break;
      case 'ArrowUp':
        next = (index - 1 + items.length) % items.length;
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = items.length - 1;
        break;
      case 'ArrowLeft':
      case 'ArrowRight': {
        event.preventDefault();
        const delta = event.key === 'ArrowRight' ? 1 : -1;
        if (compact.matches) {
          if (delta < 0) close(true);
          return;
        }
        open(
          categories[
            (categories.indexOf(openName) + delta + categories.length) % categories.length
          ],
        );
        return;
      }
      case 'Escape':
        event.preventDefault();
        close(true);
        return;
      case 'Tab':
        close(true);
        return;
      case '/':
        // The workbook has one global Find shortcut.
        close(true);
        return;
      default: {
        if (event.key.length !== 1 || event.key === ' ') return;
        const now = performance.now();
        const letter = event.key.toLocaleLowerCase();
        search = now - lastKeyTime > 600 || search === letter ? letter : search + letter;
        lastKeyTime = now;
        for (let i = 1; i <= items.length; i++) {
          const candidate = (index + i) % items.length;
          if (labelFor(items[candidate].spec).toLocaleLowerCase().startsWith(search)) {
            next = candidate;
            break;
          }
        }
      }
    }
    event.preventDefault();
    items[next]?.control.focus();
  });
  document.addEventListener('pointerdown', (event) => {
    if (openName && !host.contains(event.target)) close();
  });
  document.addEventListener('focusin', (event) => {
    if (openName && !host.contains(event.target)) close();
  });
  window.addEventListener('resize', placePopup);
  compact.addEventListener('change', () => {
    const ownedFocus = host.contains(document.activeElement);
    close();
    roving = visibleMenus()[0];
    syncTriggers();
    if (ownedFocus) triggers.get(roving).focus();
  });
  syncTriggers();
  window.SctrialAppMenu = {
    setState(next) {
      Object.assign(state, next);
      items.forEach(updateItem);
    },
  };
})();
