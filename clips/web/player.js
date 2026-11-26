/* sctrial feature clips: the player.
 *
 * Loads the scenes listed below, then either mounts the feature list beside one player
 * (Studio.mount) or, with ?capture=1 in the address, exposes window.__clip so that clips/render.py
 * can ask for single frames of one scene (?scene=<id>) or of a reel (?reel=square or ?reel=wide,
 * with ?only=id,id to pick scenes).
 */
(function (S) {
  "use strict";
  const SCENES = ["01_design", "02_participants", "03_two_arm", "04_single_arm", "05_cross_sectional",
                  "06_repeated_visits", "07_small_cohorts", "08_incomplete", "09_cell_types", "10_power"];
  const REEL_WORDS = { name: "sctrial", line: "Single-cell trials, analyzed per participant",
                       install: "pip install sctrial", link: "omar-lab.com/sctrial" };
  const base = new URL(".", document.currentScript.src).href;      // .../web/
  const params = new URLSearchParams(location.search);
  const states = {};

  function script(name) {
    return new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = `${base}scenes/${name}.js`; el.onload = resolve;
      el.onerror = () => reject(new Error(`scene script ${name} did not load`));
      document.head.appendChild(el);
    });
  }
  // a scene script that does not load is reported and skipped, so the other scenes still play
  const broken = [];
  const ready = SCENES.reduce((p, name) => p.then(() => script(name)).catch((error) => {
    broken.push(name); console.error("sctrial features:", error);
  }), Promise.resolve());

  const state = (scene) => states[scene.id] || (states[scene.id] = scene.prepare ? scene.prepare() : {});
  const length = (scene) => scene.duration || 10;

  function canvas() {
    const el = document.createElement("canvas");
    el.width = S.W; el.height = S.H;
    return el;
  }

  /* The feature list beside one player. */
  async function mount(container) {
    await ready;
    const make = (tag, cls, parent) => {
      const el = document.createElement(tag);
      el.className = cls; parent.appendChild(el);
      return el;
    };
    container.classList.add("sct-features"); container.replaceChildren();
    if (!S.scenes.length) {
      make("p", "sct-notice", container).textContent = "The feature clips could not be loaded.";
      return;
    }
    const list = make("nav", "sct-list", container);
    list.setAttribute("aria-label", "Features");
    const player = make("div", "sct-player", container), stage = make("div", "sct-stage", player);
    const view = canvas(), ctx = view.getContext("2d");
    view.setAttribute("role", "img");
    stage.appendChild(view);
    const pause = make("button", "sct-pause", stage), notice = make("p", "sct-notice", player);
    pause.type = "button"; notice.setAttribute("role", "status");
    // someone who asked for reduced motion gets the last frame of a clip and starts it themselves
    const calm = Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    const failed = {};
    let current = S.scenes.find((s) => s.id === params.get("scene")) || S.scenes[0];
    let playing = !calm, origin = performance.now(), held = 0, visible = true, dirty = true;

    const button = () => {
      pause.textContent = playing ? "II" : "▶";
      pause.setAttribute("aria-label", playing ? "Pause" : "Play");
    };
    const fail = (scene, error) => {
      failed[scene.id] = true;
      console.error(`sctrial features: ${scene.id}:`, error);
      if (scene === current) notice.textContent = `The clip "${scene.title}" could not be shown.`;
    };
    function select(scene) {
      current = scene; notice.textContent = ""; delete failed[scene.id];
      list.querySelectorAll(".sct-item").forEach((el) => {
        el.classList.toggle("on", el.dataset.id === scene.id);
        if (el.dataset.id === scene.id) el.setAttribute("aria-current", "true"); else el.removeAttribute("aria-current");
      });
      view.setAttribute("aria-label", `${scene.title}. ${scene.caption}`);
      try { state(scene); } catch (error) { fail(scene, error); }
      origin = performance.now(); held = playing ? 0 : length(scene) - 0.7; dirty = true;
    }
    S.scenes.forEach((scene) => {
      const el = make("button", "sct-item", list);
      el.type = "button"; el.dataset.id = scene.id;
      make("h3", "", el).textContent = scene.title; make("p", "", el).textContent = scene.caption;
      el.addEventListener("click", () => select(scene));
    });
    pause.addEventListener("click", () => {
      playing = !playing; button();
      if (playing) { if (held >= length(current) - 0.7) held = 0; origin = performance.now() - held * 1000; }
    });
    button();
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => { visible = entries[0].isIntersecting; }).observe(stage);
    }
    // the loop survives a scene that throws: that scene is reported once and left alone
    function tick(now) {
      requestAnimationFrame(tick);
      if (failed[current.id] || !visible || document.hidden || !(playing || dirty)) return;
      if (playing) held = ((now - origin) / 1000) % length(current);
      try {
        S.frame(ctx, current, held, state(current)); dirty = false;
      } catch (error) {
        fail(current, error);
      }
    }
    select(current);
    requestAnimationFrame(tick);
  }

  /* One scene on a bare 1920 x 1080 canvas, or a reel, drawn frame by frame on request. */
  function capture() {
    const layout = S.REELS[params.get("reel")], reel = Boolean(layout), view = canvas(), ctx = view.getContext("2d");
    if (params.has("reel") && !reel) throw new Error(`unknown reel layout ${params.get("reel")}`);
    if (reel) { view.width = layout.W; view.height = layout.H; }
    view.id = "canvas"; view.style.cssText = `display:block;width:${view.width}px;height:${view.height}px`;
    document.body.style.cssText = "margin:0;background:#000";
    document.body.replaceChildren(view);
    const opened = ready.then(() => {
      if (broken.length) throw new Error(`scene scripts did not load: ${broken.join(", ")}`);
      if (reel) {
        const only = params.get("only") ? params.get("only").split(",") : null;
        const picked = only ? only.map((id) => S.scenes.find((s) => s.id === id)) : S.scenes;
        if (picked.some((s) => !s)) throw new Error(`unknown scene in ${params.get("only")}`);
        picked.forEach(state);
        window.__clip.duration = S.reelDuration(picked);
        window.__clip.render = (t) => S.reelFrame(ctx, t, states, REEL_WORDS, layout, picked);
        return true;
      }
      const scene = params.has("scene") ? S.scenes.find((s) => s.id === params.get("scene")) : S.scenes[0];
      if (!scene) throw new Error(`unknown scene ${params.get("scene")}`);
      const own = state(scene);
      window.__clip.duration = length(scene);
      window.__clip.render = (t) => S.frame(ctx, scene, t, own);
      window.__clip.caption = scene.caption;
      window.__clip.title = scene.title;
      return true;
    });
    window.__clip = { ready: opened, scenes: ready.then(() => S.scenes.map((s) => s.id)) };
  }

  S.mount = mount;
  if (params.has("capture")) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", capture);
    else capture();
  }
})(window.Studio);
