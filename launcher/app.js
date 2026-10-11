// The launcher's page. It asks the keeper (launcher.py) for its state once a second and redraws
// in place: nothing here is remembered, so a refresh or a second window changes nothing.
"use strict";

const $ = (id) => document.getElementById(id);
const GROUP = {
  pipeline: "run",
  phone_usb: "phone", phone_trust: "phone", phone_stream: "phone", phone_cable: "phone",
  airpods: "airpods", airpods_output: "airpods", airpods_mic: "airpods", bluetooth: "airpods",
  sound_check: "airpods", mic_check: "airpods",
};
const MARK = {
  green: '<svg viewBox="0 0 24 24"><path d="M4 13l5 5L20 6"/></svg>',
  amber: '<svg viewBox="0 0 24 24"><path d="M12 4v10M12 19.5v.5"/></svg>',
  red: '<svg viewBox="0 0 24 24"><path d="M5 5l14 14M19 5L5 19"/></svg>',
  none: '<svg viewBox="0 0 24 24"><path d="M6 12h12"/></svg>',
};
const SAYS = { green: "ready", amber: "look at this", red: "needs fixing", none: "not tested" };

let state = null;
let skew = 0;            // the keeper's clock minus this browser's, so "running for" is right on a laptop too
let failures = 0;
const open = new Set();  // lists of files the person has unfolded

async function act(action, more) {
  try {
    const r = await fetch("api/do", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Proprio": "1" },
      body: JSON.stringify({ action, ...(more || {}) }),
    });
    const got = await r.json();
    if (!got.ok) banner(got.message, false);
  } catch (e) {
    banner("The keeper did not answer. Double-click the Proprio icon again.", false);
  }
  poll();
}

// The options card: guidance sound, hand tracking, when to say Grab. One click switches the
// running pipeline and is what it starts with from then on. Redrawn only when something changed,
// so a button is never replaced under the pointer.
let choicesShown = "";
function drawChoices(s) {
  const families = s.options || [];
  $("choices").hidden = families.length === 0;
  if (!families.length) return;
  const running = s.mode !== "checklist";
  const key = JSON.stringify([families, running, s.can_control]);
  if (key === choicesShown) return;
  choicesShown = key;
  $("choices-k").textContent = running
    ? "Click one to switch now. The wearer hears its name. By voice: \"option 3\", \"hand option 5\", \"grab option 2\"."
    : "Click one to start with it. It can be changed while it runs, here or by voice.";
  $("choices-reset").hidden = !families.some((f) => f.changed) || !s.can_control;
  const rows = $("choices-rows");
  rows.textContent = "";
  for (const f of families) {
    const row = document.createElement("div");
    row.className = "choice";
    const head = document.createElement("h3");
    head.textContent = f.heading;
    const list = document.createElement("div");
    list.className = "choice-list";
    for (const o of f.options) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice-b";
      b.title = o.about || o.title;
      b.disabled = !s.can_control;
      b.setAttribute("aria-pressed", String(o.n === f.current));
      const n = document.createElement("span");
      n.className = "choice-n";
      n.textContent = o.n;
      const t = document.createElement("span");
      t.textContent = o.title + (o.n === f.default ? " (usual)" : "");
      b.append(n, t);
      b.addEventListener("click", () => act("option", { family: f.family, n: o.n }));
      list.append(b);
    }
    const now = f.options.find((o) => o.n === f.current);
    const about = document.createElement("p");
    about.className = "choice-about";
    about.textContent = now ? now.about : "";
    row.append(head, list, about);
    rows.append(row);
  }
}
$("choices-reset").addEventListener("click", () => act("default_options"));

let bannerUntil = 0;
function banner(text, info) {
  const el = $("banner");
  el.textContent = text;
  el.classList.toggle("is-info", !!info);
  el.hidden = !text;
  bannerUntil = Date.now() + 8000;
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function drawLine(line, canControl) {
  let li = document.querySelector(`[data-id="${line.id}"]`);
  // while it runs, where the session is being saved belongs with the run, not down among the checks
  const group = line.id === "record" && state.mode !== "checklist" ? "run" : GROUP[line.id] || "computer";
  const home = $("g-" + group);
  if (!li) {
    li = document.createElement("li");
    li.dataset.id = line.id;
    li.innerHTML = '<span class="mark"></span><h3 class="line-title"></h3><p class="line-msg"></p>' +
      '<p class="line-note" hidden></p><details class="line-detail" hidden><summary></summary><ol></ol></details>' +
      '<button class="btn line-fix" type="button" hidden></button>';
    li.querySelector(".line-fix").addEventListener("click", (e) => act(e.currentTarget.dataset.action));
    li.querySelector("details").addEventListener("toggle", (e) => {
      if (e.currentTarget.open) open.add(line.id); else open.delete(line.id);
    });
  }
  if (li.parentElement !== home) home.appendChild(li);
  const level = line.blocked ? "blocked" : line.level;
  const cls = `line is-${line.level}${line.blocked ? " is-blocked" : ""}${line.minor ? " is-minor" : ""}`;
  if (li.className !== cls) {
    li.className = cls;
    li.querySelector(".mark").innerHTML = MARK[line.level] || MARK.none;
    li.querySelector(".mark").title = level === "blocked" ? "waiting for a line above" : SAYS[line.level] || "";
  }
  const set = (sel, text) => { const el = li.querySelector(sel); if (el.textContent !== text) el.textContent = text; return el; };
  set(".line-title", line.title);
  set(".line-msg", line.message);
  set(".line-note", line.note || "").hidden = !line.note;
  const fix = li.querySelector(".line-fix");
  fix.hidden = !line.fix;
  if (line.fix) {
    set(".line-fix", line.fix.label);
    fix.dataset.action = line.fix.action;
    fix.disabled = !canControl || (state.busy || []).includes(line.fix.action);
  }
  const detail = li.querySelector("details");
  detail.hidden = !line.detail;
  if (line.detail) {
    const bad = line.detail.filter((d) => !d.ok).length;
    set("summary", bad ? `${bad} of ${line.detail.length} not right: show the list` : `Show all ${line.detail.length}`);
    detail.open = open.has(line.id);
    if (detail.open) {
      detail.querySelector("ol").innerHTML = "";
      for (const d of line.detail) {
        const row = document.createElement("li");
        row.className = d.ok ? "" : "is-bad";
        row.textContent = `${d.ok ? "✓" : "✕"} ${d.name}: ${d.note}`;
        detail.querySelector("ol").appendChild(row);
      }
    }
  }
  return line.id;
}

function draw() {
  const s = state;
  const running = s.mode !== "checklist";
  const now = Date.now() / 1000 + skew;
  const lines = s.lines;
  const seen = new Set(lines.map((l) => drawLine(l, s.can_control)));
  document.querySelectorAll(".line").forEach((li) => { if (!seen.has(li.dataset.id)) li.remove(); });
  // Keep each column in the keeper's order. Only when it differs: moving a line while it is
  // being clicked would swallow the click.
  document.querySelectorAll(".lines").forEach((list) => {
    const want = lines.map((l) => l.id).filter((id) => list.querySelector(`:scope > [data-id="${id}"]`));
    const have = [...list.children].map((li) => li.dataset.id);
    if (want.join() !== have.join()) want.forEach((id) => list.appendChild(list.querySelector(`:scope > [data-id="${id}"]`)));
  });

  drawChoices(s);
  const waiting = lines.filter((l) => l.required && l.level !== "green");
  const reds = waiting.filter((l) => l.level === "red" && !l.blocked);
  const head = $("state");
  let word, tone;
  if (s.mode === "stopping") { word = "Stopping"; tone = "amber"; }
  else if (running) {
    const p = lines.find((l) => l.id === "pipeline") || { level: "amber" };
    word = p.level === "green" ? "Running" : p.level === "amber" ? "Starting" : "Recovering";
    tone = p.level;
  } else if (s.ready) { word = "Ready"; tone = "green"; }
  else { word = "Not ready"; tone = reds.length ? "red" : "amber"; }
  head.textContent = word;
  head.className = "bar-state is-" + tone;
  document.title = `Proprio, start: ${word.toLowerCase()}`;

  $("k1").textContent = running ? "Running for" : "Before the demo";
  $("v1").textContent = running ? clock(now - (s.since || now))
    : s.ready ? "Everything is green" : `${waiting.length} line${waiting.length === 1 ? "" : "s"} to fix`;
  $("i2").hidden = $("i3").hidden = !running;
  $("v2").textContent = String(s.restarts);
  $("v3").textContent = s.target || "nothing yet";

  $("start").hidden = running;
  $("start").disabled = !s.ready || !s.can_control;
  $("anyway").hidden = running || s.ready;
  $("anyway").disabled = !s.can_control;
  $("stop").hidden = s.mode !== "running";
  $("stop").disabled = !s.can_control;
  $("live").hidden = !running;
  $("live").href = `http://${location.hostname}:${s.dash_port}/#camera`;
  $("typing").hidden = !running;
  $("typing").href = `https://${location.hostname}:${s.voice_port}/`;

  $("run").hidden = !running;
  for (const [id, text] of [["heard", s.heard], ["said", s.said]]) {
    $(id).textContent = text || "nothing yet";
    $(id).classList.toggle("is-empty", !text);
  }

  const events = $("events");
  const rows = s.events.slice().reverse().slice(0, 8);
  const key = rows.map((e) => e.t).join();
  if (events.dataset.key !== key) {
    events.dataset.key = key;
    events.innerHTML = "";
    for (const e of rows) {
      const li = document.createElement("li");
      const t = document.createElement("time");
      t.textContent = new Date(e.t * 1000).toLocaleTimeString("en-GB");
      const what = document.createElement("span");
      what.textContent = e.text.split("\n")[0];
      li.append(t, what);
      events.appendChild(li);
    }
  }
  if (!$("log").open) $("log-text").textContent = (s.log || []).join("\n") || "Nothing yet.";

  const bits = [`${s.host}`];
  if (s.replay) bits.push(`camera: replaying ${s.replay}`);
  for (const [k, v] of Object.entries(s.settings || {})) bits.push(`${k}=${v}`);
  bits.push(s.offline ? "models: offline" : "models: network allowed");
  if (!s.can_control) bits.push("the buttons work on the GX10's own screen");
  $("foot").textContent = bits.join("  ·  ");
  if (Date.now() > bannerUntil) $("banner").hidden = true;
}

async function poll() {
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 3000);
    const r = await fetch("api/state", { signal: ctl.signal, cache: "no-store" });
    clearTimeout(timer);
    state = await r.json();
    skew = state.t - Date.now() / 1000;
    failures = 0;
    draw();
  } catch (e) {
    failures += 1;
    if (failures >= 3) {
      $("state").textContent = "No keeper";
      $("state").className = "bar-state is-red";
      $("start").disabled = true;
      banner("The keeper is not answering, so this page is out of date. Double-click the Proprio icon on the desktop. " +
             "A demo that was running carries on by itself.", false);
    }
  }
  if ($("log").open) {
    try {
      const r = await fetch("api/log", { cache: "no-store" });
      const pre = $("log-text");
      const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
      pre.textContent = (await r.json()).lines.join("\n") || "Nothing yet.";
      if (atEnd) pre.scrollTop = pre.scrollHeight;
    } catch (e) { /* the banner above already says so */ }
  }
}

$("start").addEventListener("click", () => act("start"));
$("anyway").addEventListener("click", () => {
  const waiting = state.lines.filter((l) => l.required && l.level !== "green").map((l) => l.title);
  if (confirm(`Start although these are not green?\n\n${waiting.join("\n")}`)) act("start_anyway");
});
$("stop").addEventListener("click", () => {
  if (confirm("Stop the demo?\n\nThe pipeline stops. The phone bridge and the model servers stay as they are.")) act("stop");
});

setInterval(poll, 1000);
poll();
