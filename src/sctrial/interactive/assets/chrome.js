(() => {
  'use strict';
  const control = document.getElementById('screenToggle');
  const fullScreenSupported =
    !!document.documentElement.requestFullscreen && document.fullscreenEnabled;
  control.hidden = !fullScreenSupported;
  control.addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      control.hidden = true;
    }
  });
  document.addEventListener('fullscreenchange', () => {
    const active = !!document.fullscreenElement;
    control.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
    control.setAttribute('aria-pressed', String(active));
  });
})();

// A link selects a workspace view. Study names, participant IDs and outcomes stay in the document.
(() => {
  'use strict';
  const fragments = new Map(
    Object.entries({
      cells: 'cells',
      genes: 'expression',
      pathways: 'pathways',
      modules: 'scores',
      trajectory: 'time',
      comparison: 'plot',
      changes: 'changes',
      measurements: 'measurements',
      participants: 'participants',
      results: 'results',
      bench: 'pins',
    }),
  );
  const modes = new Map(Array.from(fragments, ([mode, hash]) => [hash, mode]));
  let listening = false;
  let lastMode = 'cells';
  const followFragment = () => {
    const fragment = location.hash.slice(1);
    const mode = fragment ? modes.get(fragment) : 'cells';
    if (!mode || mode === lastMode) return;
    lastMode = mode;
    const detail = ['participants', 'results', 'bench'].includes(mode)
      ? { view: mode }
      : { view: 'overview', mode };
    document.dispatchEvent(new CustomEvent('sctrial:menu-command', { detail }));
  };
  window.SctrialNavigation = {
    sync(mode) {
      if (!listening || !fragments.has(mode) || mode === lastMode) return;
      lastMode = mode;
      const hash = '#' + fragments.get(mode);
      if (location.hash !== hash) {
        try {
          history.pushState(null, '', hash);
        } catch {
          /* Embedded and local viewers can restrict browser history. */
        }
      }
    },
  };
  window.addEventListener('DOMContentLoaded', () => {
    listening = true;
    followFragment();
  });
  window.addEventListener('popstate', followFragment);
  window.addEventListener('hashchange', followFragment);
})();
