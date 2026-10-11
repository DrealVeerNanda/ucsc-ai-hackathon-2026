/* The options panel: every way of guiding, of tracking the hand and of deciding "grab" that the
   pipeline offers, with the one in use marked. A click switches while it runs. The list comes from
   /options, so a new option in the Python code shows up here with no change to this file. Kept
   apart from app.js, like logs.js. */
(() => {
  "use strict";
  const button = document.getElementById("options-btn");
  const panel = document.getElementById("options");
  const list = document.getElementById("options-list");
  const note = document.getElementById("options-note");
  let shown = "";
  let version = null;     // the server's stamp of these files when this tab loaded them

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  async function choose(family, n) {
    note.textContent = "";
    try {
      const r = await fetch("/option", { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify({ family, n }) });
      const d = await r.json();
      note.textContent = d.ok ? "Switched to: " + d.title : "Could not switch: " + d.error;
    } catch (e) {
      note.textContent = "Could not switch: the pipeline is not answering.";
    }
    refresh();
  }
  function draw(families) {
    const key = JSON.stringify(families);
    if (key === shown) return;                      // nothing changed: leave focus and hover alone
    shown = key;
    list.textContent = "";
    for (const f of families) {
      const group = el("section", "options-group");
      group.append(el("h2", "options-h", f.heading));
      const row = el("div", "options-row");
      for (const o of f.options) {
        const b = el("button", "options-b");
        b.type = "button";
        b.title = o.about || o.title;
        b.setAttribute("aria-pressed", String(o.n === f.current));
        b.append(el("span", "options-n", String(o.n)), el("span", "options-t", o.title));
        b.addEventListener("click", () => choose(f.family, o.n));
        row.append(b);
      }
      group.append(row);
      const now = f.options.find((o) => o.n === f.current);
      group.append(el("p", "options-about", now ? now.about : "Starts with the first request."));
      list.append(group);
    }
  }
  async function refresh() {
    try {
      const r = await fetch("/options", { cache: "no-store" });
      const d = await r.json();
      if (version === null) version = d.version;
      else if (d.version && d.version !== version) return location.reload();      // the page was updated
      button.hidden = d.families.length === 0;
      draw(d.families);
    } catch (e) { /* main.py is not running: keep what is shown */ }
  }
  function toggle(open) {
    panel.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (open) refresh();
  }

  button.addEventListener("click", () => toggle(panel.hidden));
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === "INPUT") return;
    if (e.key === "o" || e.key === "O") toggle(panel.hidden);
    if (e.key === "Escape" && !panel.hidden) toggle(false);
  });
  refresh();
  if (location.hash === "#options") toggle(true);
  setInterval(refresh, 3000);     // also how a tab left open notices that the page has been updated
})();
