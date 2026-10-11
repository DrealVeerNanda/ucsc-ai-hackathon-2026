// Deck controller: slides and builds, keyboard/click navigation, #slide.step in the URL,
// a speaker-notes window (N) kept in sync over BroadcastChannel, fullscreen (F), and an
// auto-play mode (?auto) that advances on each slide's data-dur for recording a video.
import { createStage } from "./stage.js";

const $ = (id) => document.getElementById(id);
const slides = [...document.querySelectorAll(".slide")];
const stage = createStage($("stage"), {
  word: $("lab-word"), mug: $("lab-mug"), you: $("lab-you"), cloud: $("lab-cloud"),
  ax: $("lab-ax"), ay: $("lab-ay"), az: $("lab-az"),
  gx10: $("lab-gx10"), gxx: $("lab-gxx"), gxy: $("lab-gxy"), gxz: $("lab-gxz"),
  q: $("lab-q"), tip: $("lab-tip"), hand: $("lab-hand"),
  motors: [...document.querySelectorAll(".lab-motor")],
});
const channel = "BroadcastChannel" in window ? new BroadcastChannel("deck") : null;
const params = new URLSearchParams(location.search);
const auto = params.has("auto");
// The live dashboard: this same server when main.py serves the deck, or ?dash=http://<gx10>:8642
// when the deck runs elsewhere. Without one, the demo slide plays media/demo-run.mp4 if present.
const DASH = (params.get("dash") || "").replace(/\/$/, "");
const RECORDED = "media/demo-run.mp4";
if (new URLSearchParams(location.search).has("still")) document.documentElement.classList.add("is-still");

let index = 0;
let step = 0;
let shownIndex = -1;
let started = null;

const stepsOf = (s) => Number(s.dataset.steps || 0);
const shotsOf = (s) => (s.dataset.shots || "title").split(",");

function parseHash() {
  const m = location.hash.match(/^#(\d+)(?:\.(\d+))?$/);
  if (!m) return [0, 0];
  const i = Math.min(slides.length - 1, Math.max(0, Number(m[1]) - 1));
  return [i, Math.min(stepsOf(slides[i]), Number(m[2] || 0))];
}

function show(i, s) {
  index = i;
  step = s;
  slides.forEach((el, k) => el.classList.toggle("is-on", k === i));
  const slide = slides[i];
  slide.querySelectorAll("[data-step]").forEach((el) => el.classList.toggle("is-later", Number(el.dataset.step) > s));
  // data-only: shown at exactly that build (0-based), so builds replace each other in place
  slide.querySelectorAll("[data-only]").forEach((el) => {
    const n = Number(el.dataset.only);
    el.classList.toggle("is-off", n !== s);
    el.classList.toggle("is-past", n < s);
  });
  if (i !== shownIndex) {
    transition(slide, i > shownIndex || shownIndex < 0);
    shownIndex = i;
  }
  effects(slide);
  const shots = shotsOf(slide);
  stage.setShot(shots[Math.min(shots.length - 1, s)] || shots[0]);
  if (slide.dataset.id === "live") loadLive();
  history.replaceState(null, "", `#${i + 1}${s ? `.${s}` : ""}${location.search ? "" : ""}`);
  $("counter").textContent = `${i + 1} / ${slides.length}`;
  publish();
}

function next() {
  if (started === null) started = Date.now();
  if (step < stepsOf(slides[index])) return show(index, step + 1);
  if (index < slides.length - 1) show(index + 1, 0);
}
function prev() {
  if (step > 0) return show(index, step - 1);
  if (index > 0) show(index - 1, stepsOf(slides[index - 1]));
}

// --- notes window ---------------------------------------------------------------------------
function noteOf(s) {
  const n = s.querySelector(".notes");
  return n ? n.textContent.trim() : "";
}
function titleOf(s) {
  const h = s.querySelector("h1, h2");
  return h ? h.textContent.trim() : s.dataset.id;
}
function publish() {
  if (!channel) return;
  const s = slides[index];
  const n = slides[index + 1];
  channel.postMessage({
    index, step, steps: stepsOf(s), total: slides.length, title: titleOf(s), notes: noteOf(s),
    next: n ? titleOf(n) : "End", started, dur: slides.reduce((a, el) => a + Number(el.dataset.dur || 0), 0),
  });
}
channel?.addEventListener("message", (e) => {
  if (e.data === "hello") publish();
  else if (e.data === "next") next();
  else if (e.data === "prev") prev();
});

// --- live pieces ----------------------------------------------------------------------------
let dashboard = null; // base URL of a reachable dashboard, or false
async function findDashboard() {
  if (dashboard !== null) return dashboard;
  try {
    const r = await fetch(`${DASH}/trials.json`, { cache: "no-store" });
    dashboard = r.ok && (r.headers.get("content-type") || "").includes("json") ? DASH : false;
  } catch {
    dashboard = false;
  }
  return dashboard;
}

async function loadLive() {
  const f = $("live");
  if (f.dataset.started) return;
  f.dataset.started = "1";
  const note = $("live-note");
  const base = await findDashboard();
  if (base !== false) {
    f.addEventListener("load", () => f.classList.add("is-loaded"), { once: true });
    f.src = `${base}/`;
    return;
  }
  const recorded = await fetch(RECORDED, { method: "HEAD" }).then((r) => r.ok).catch(() => false);
  if (recorded) {
    const v = document.createElement("video");
    Object.assign(v, { src: RECORDED, autoplay: true, muted: true, loop: true, playsInline: true });
    v.className = "live is-loaded";
    f.replaceWith(v);
    note.hidden = true;
  } else {
    note.textContent = "Live demo runs here. Start main.py, or add media/demo-run.mp4 for the recorded run.";
  }
}

// --- transitions: each slide arrives its own way --------------------------------------------
const curtain = $("curtain");
const still = params.has("still");
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const EASE = "cubic-bezier(0.65, 0, 0.35, 1)";

function transition(slide, forward) {
  curtain.getAnimations().forEach((a) => a.cancel());
  curtain.replaceChildren();
  curtain.style.cssText = "";
  const kind = slide.dataset.trans || "none";
  if (still || kind === "none" || shownIndex < 0) return;
  if (reduced.matches) {
    curtain.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 400, easing: "linear" });
    return;
  }
  const dir = forward ? 1 : -1;
  const run = (kf, opt) => curtain.animate(kf, { duration: 850, easing: EASE, fill: "both", ...opt });
  switch (kind) {
    case "lights": // the room goes dark, then the points fade up
      run([{ opacity: 1 }, { opacity: 1, offset: 0.35 }, { opacity: 0 }], { duration: 1600, easing: "ease-in" });
      break;
    case "iris": { // a circle opens on the scene
      const t0 = performance.now(), dur = 950;
      const tick = (now) => {
        const k = Math.min(1, (now - t0) / dur), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        const r = `${e * 160}vmax`;
        const g = `radial-gradient(circle at 70% 55%, transparent ${r}, var(--ground) calc(${r} + 1px))`;
        curtain.style.maskImage = curtain.style.webkitMaskImage = g;
        if (k < 1) requestAnimationFrame(tick); else curtain.style.cssText = "opacity:0";
      };
      requestAnimationFrame(tick);
      break;
    }
    case "wipe":
      run([{ transform: "translateX(0)" }, { transform: `translateX(${100 * dir}%)` }]);
      break;
    case "blinds": { // horizontal slats close away one after another
      curtain.style.background = "transparent";
      for (let k = 0; k < 8; k++) {
        const slat = document.createElement("div");
        slat.className = "slat";
        curtain.append(slat);
        slat.animate([{ transform: "scaleY(1)" }, { transform: "scaleY(0)" }], { duration: 520, delay: k * 70, easing: EASE, fill: "both" });
      }
      break;
    }
    case "split": { // the screen opens like doors, top and bottom
      curtain.style.background = "transparent";
      for (const sign of [-1, 1]) {
        const half = document.createElement("div");
        half.className = sign < 0 ? "half top" : "half bottom";
        curtain.append(half);
        half.animate([{ transform: "translateY(0)" }, { transform: `translateY(${sign * 100}%)` }], { duration: 900, easing: EASE, fill: "both" });
      }
      break;
    }
    case "screen": // the live view grows out of a small screen
      curtain.style.opacity = "0";
      $("live").animate([{ transform: "scale(0.28)", opacity: 0.4 }, { transform: "scale(1)", opacity: 1 }],
        { duration: 1000, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
      break;
    case "columns": { // vertical shutters fold away left to right
      curtain.style.background = "transparent";
      curtain.style.display = "flex";
      for (let k = 0; k < 8; k++) {
        const col = document.createElement("div");
        col.className = "col";
        curtain.append(col);
        col.animate([{ transform: "scaleX(1)" }, { transform: "scaleX(0)" }], { duration: 520, delay: k * 60, easing: EASE, fill: "both" });
      }
      break;
    }
    case "box": // the screen opens from its centre outward
      run([{ clipPath: "inset(0% 0% 0% 0%)" }, { clipPath: "inset(0% 50% 0% 50%)" }], { duration: 900 });
      break;
    case "lift":
      run([{ transform: "translateY(0)" }, { transform: "translateY(-100%)" }], { duration: 800 });
      break;
    case "diagonal":
      run([{ clipPath: "polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%)" }, { clipPath: "polygon(100% 0%, 100% 0%, 100% 100%, 135% 100%)" }], { duration: 950 });
      break;
  }
}

// --- text effects: count-ups, typing, decoding ------------------------------------------------
const fmt = new Intl.NumberFormat("en-US");
const playing = new WeakMap();

function visible(el, slide) {
  if (!slide.classList.contains("is-on")) return false;
  const off = el.closest("[data-only]");
  const later = el.closest("[data-step]");
  return !(off && off.classList.contains("is-off")) && !(later && later.classList.contains("is-later"));
}

function animateText(el, render, dur) {
  const t0 = performance.now();
  const tick = (now) => {
    if (playing.get(el) !== t0) return;
    const k = Math.min(1, (now - t0) / dur);
    el.textContent = render(k);
    if (k < 1) requestAnimationFrame(tick);
  };
  playing.set(el, t0);
  requestAnimationFrame(tick);
}

function effects(slide) {
  const instant = still || reduced.matches;
  slide.querySelectorAll("[data-count], [data-type], [data-scramble]").forEach((el) => {
    const on = visible(el, slide);
    if (!on) { el.dataset.played = ""; return; }
    if (el.dataset.played) return;
    el.dataset.played = "1";
    const final = el.dataset.final || (el.dataset.final = el.textContent);
    if (instant) { el.textContent = final; return; }
    if (el.hasAttribute("data-count")) {
      const n = Number(el.dataset.count), pre = el.dataset.prefix || "", suf = el.dataset.suffix || "";
      const show = (v) => pre + (el.hasAttribute("data-group") ? fmt.format(v) : String(v)) + suf;
      animateText(el, (k) => show(Math.round(n * (1 - Math.pow(1 - k, 3)))), 1200);
    } else if (el.hasAttribute("data-type")) {
      animateText(el, (k) => final.slice(0, Math.round(final.length * k)) || "\u200b", final.length * 70);
    } else {
      const glyphs = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789#%&";
      animateText(el, (k) => [...final].map((c, i) => (c === " " || i / final.length < k ? c : glyphs[(Math.random() * glyphs.length) | 0])).join(""), 800);
    }
  });
}

// --- input ------------------------------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (["ArrowRight", "ArrowDown", "PageDown", " ", "Enter"].includes(k)) { e.preventDefault(); next(); }
  else if (["ArrowLeft", "ArrowUp", "PageUp", "Backspace"].includes(k)) { e.preventDefault(); prev(); }
  else if (k === "Home") show(0, 0);
  else if (k === "End") show(slides.length - 1, stepsOf(slides[slides.length - 1]));
  else if (k === "f" || k === "F") {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  } else if (k === "n" || k === "N") window.open("notes.html", "deck-notes", "width=760,height=560");
});
document.addEventListener("click", (e) => { if (e.button === 0) next(); });
document.addEventListener("contextmenu", (e) => { e.preventDefault(); prev(); });
window.addEventListener("hashchange", () => show(...parseHash()));

let idle = 0;
document.addEventListener("mousemove", () => {
  document.body.classList.remove("is-idle");
  clearTimeout(idle);
  idle = setTimeout(() => document.body.classList.add("is-idle"), 1400);
});

// --- auto-play for recording ------------------------------------------------------------------
function autoplay() {
  const s = slides[index];
  const per = (Number(s.dataset.dur || 6) * 1000) / (stepsOf(s) + 1);
  setTimeout(() => {
    if (index === slides.length - 1 && step === stepsOf(s)) return;
    next();
    autoplay();
  }, per);
}

// typeset the core-loop equations (KaTeX is vendored and loaded before this module)
document.querySelectorAll("[data-tex]").forEach((el) => {
  if (window.katex) window.katex.render(el.dataset.tex, el, { throwOnError: false });
  else el.textContent = el.dataset.tex;
});

show(...parseHash());
if (auto) { started = Date.now(); autoplay(); }
