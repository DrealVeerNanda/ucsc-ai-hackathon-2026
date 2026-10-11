/* Logs-only view: a switch in the bar that hides the camera and the 3D scene and shows what the
   pipeline heard, what each model answered and what was said. Kept apart from app.js so the two
   views do not depend on each other. Lines come from /log, fed by the pipeline's own trace. */
(() => {
  "use strict";
  const button = document.getElementById("mode");
  const panel = document.getElementById("logs");
  const list = document.getElementById("logs-list");
  const KIND = { SPOKEN: "is-spoken", STATE: "is-state", NEMOTRON: "is-model", COSMOS: "is-model", WHISPER: "is-model", DETECTOR: "is-model" };
  const MAX = 300;
  let after = 0;

  function remembered() {
    try { return localStorage.getItem("blindfold-view") === "logs"; } catch (e) { return false; }
  }
  function setMode(logs) {
    document.body.classList.toggle("logs-only", logs);
    panel.hidden = !logs;
    button.setAttribute("aria-pressed", String(logs));
    button.textContent = logs ? "Back to camera view" : "Logs only";
    try { localStorage.setItem("blindfold-view", logs ? "logs" : "normal"); } catch (e) {}
    if (logs) list.scrollTop = list.scrollHeight;
    window.dispatchEvent(new Event("resize"));     // the camera and 3D view lay themselves out again
  }

  function cell(cls, text) {
    const el = document.createElement("span");
    el.className = cls;
    el.textContent = text;
    return el;
  }
  function add(line) {
    const stuck = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
    const li = document.createElement("li");
    if (KIND[line.who]) li.className = KIND[line.who];
    const t = line.t ? new Date(line.t * 1000) : new Date();
    li.append(cell("logs-t", t.toLocaleTimeString("en-GB")), cell("logs-who", line.who),
              cell("logs-took", line.took || ""), cell("logs-what", line.what));
    list.append(li);
    while (list.children.length > MAX) list.firstChild.remove();
    if (stuck) list.scrollTop = list.scrollHeight;  // follow the newest line unless the reader scrolled up
  }
  async function poll() {
    try {
      const r = await fetch("/log?after=" + after, { cache: "no-store" });
      const d = await r.json();
      if (d.next < after) list.textContent = "";    // main.py was restarted: start a fresh log
      after = d.next;
      d.lines.forEach(add);
    } catch (e) { /* main.py is not running: keep what is shown and try again */ }
    setTimeout(poll, 400);
  }

  button.addEventListener("click", () => setMode(!document.body.classList.contains("logs-only")));
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === "INPUT") return;
    if (e.key === "l" || e.key === "L") setMode(!document.body.classList.contains("logs-only"));
  });
  setMode(location.hash === "#logs" || (location.hash !== "#camera" && remembered()));
  poll();
})();
