// Deck controller: slides and builds, keyboard/click navigation, #slide.step in the URL,
// a speaker-notes window (N) kept in sync over BroadcastChannel, fullscreen (F), and an
// auto-play mode (?auto) that advances on each slide's data-dur for recording a video.
import { createStage } from "./stage.js";

const $ = (id) => document.getElementById(id);
const slides = [...document.querySelectorAll(".slide")];
const stage = createStage($("stage"), {
  word: $("lab-word"), mug: $("lab-mug"), you: $("lab-you"), cloud: $("lab-cloud"),
  ax: $("lab-ax"), ay: $("lab-ay"), az: $("lab-az"),
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
  // data-only: shown at exactly that build, so builds can replace each other in place
  slide.querySelectorAll("[data-only]").forEach((el) => el.classList.toggle("is-off", Number(el.dataset.only) !== Math.max(1, s)));
  const shots = shotsOf(slide);
  stage.setShot(shots[Math.min(shots.length - 1, Math.max(0, s - 1))] || shots[0]);
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

show(...parseHash());
if (auto) { started = Date.now(); autoplay(); }
