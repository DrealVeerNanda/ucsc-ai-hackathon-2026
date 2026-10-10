"use strict";

(() => {
  const $ = (id) => document.getElementById(id);

  const STATE_WORDS = {
    idle: "Waiting",
    listening: "Listening",
    searching: "Looking",
    waiting_hand: "Show me your hand",
    guiding: "Guiding",
    grab: "Grab",
  };
  const QUIET_STATES = new Set(["idle", "listening", "searching"]);
  const LANE_NAMES = { stream: "camera", perception: "detection", guidance: "guidance", voice: "voice input" };
  const DEAD_M = 0.03; // per-axis dead zone in the team plan
  const GRAB_M = 0.05;
  const STALE_MS = 300;
  const MAP_Z0 = 90; // map y of the wearer's chest, in cm
  const KEYS = {
    ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down",
    w: "w", s: "s", o: "o", h: "h", n: "n", r: "r",
  };

  const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const tenth = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  const video = $("video");
  const overlay = $("overlay");
  let latest = null;
  let scheduled = false;
  let shownImg = null;
  let shownDepth = null;
  let aspect = 0;
  let captionKey = "";
  let freshTimer = 0;
  let connected = false;

  const cm = (m) => `${whole.format(Math.abs(m) * 100)} cm`;
  const ms = (v) => (v == null ? "" : `${whole.format(v)} ms`);

  function clock(s) {
    const t = Math.max(0, Math.floor(s * 10) / 10);
    const m = Math.floor(t / 60);
    const r = t - m * 60;
    return `${m}:${r < 10 ? "0" : ""}${tenth.format(r)}`;
  }

  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  function toggle(el, cls, on) {
    el.classList.toggle(cls, Boolean(on));
  }

  // SVG elements ignore the hidden property, so set the attribute for both kinds.
  function vis(el, on) {
    el.toggleAttribute("hidden", !on);
  }

  // camera overlay ----------------------------------------------------------

  function vb(u, v) {
    return [u * 1000, (v * 1000) / aspect];
  }

  function placeTag(el, u, v, dx, dy) {
    const w = video.clientWidth;
    const h = video.clientHeight;
    const tw = el.offsetWidth;
    const th = el.offsetHeight;
    const x = Math.min(Math.max(u * w + dx, 8), w - tw - 8);
    const y = Math.min(Math.max(v * h + dy - th, 8), h - th - 8);
    el.style.setProperty("--x", `${x}px`);
    el.style.setProperty("--y", `${y}px`);
  }

  // Off to the side of the leg, never over the fingertip ring, the arrowhead or the object.
  function placeLegTag(el, hc, oc, box, dist) {
    const w = video.clientWidth;
    const h = video.clientHeight;
    const tw = el.offsetWidth;
    const th = el.offsetHeight;
    const hx = hc[0] * w, hy = hc[1] * h, ox = oc[0] * w, oy = oc[1] * h;
    const len = Math.hypot(ox - hx, oy - hy);
    let x, y;
    if (dist < 0.08 && box) {
      const right = box[2] * w + 14;
      x = right + tw < w - 8 ? right : box[0] * w - tw - 14;
      y = (box[1] + box[3]) / 2 * h - th / 2;
    } else {
      let nx = -(oy - hy) / (len || 1), ny = (ox - hx) / (len || 1);
      if (ny < 0) { nx = -nx; ny = -ny; }
      const mx = hx + (ox - hx) * 0.4, my = hy + (oy - hy) * 0.4;
      x = mx + nx * (th + 14) - tw / 2;
      y = my + ny * (th + 14) - th / 2;
    }
    el.style.setProperty("--x", `${Math.min(Math.max(x, 8), w - tw - 8)}px`);
    el.style.setProperty("--y", `${Math.min(Math.max(y, 8), h - th - 8)}px`);
  }

  function brackets(x1, y1, x2, y2) {
    const l = Math.min(x2 - x1, y2 - y1) * 0.28;
    return [
      `M${x1} ${y1 + l}V${y1}H${x1 + l}`,
      `M${x2 - l} ${y1}H${x2}V${y1 + l}`,
      `M${x2} ${y2 - l}V${y2}H${x2 - l}`,
      `M${x1 + l} ${y2}H${x1}V${y2 - l}`,
    ].join("");
  }

  function renderOverlay(d) {
    const obj = d.object;
    const hand = d.hand;
    const objG = $("obj");
    const objLabel = $("obj-label");
    const target = d.target || "object";

    let oc = null;
    if (obj.box || obj.px) {
      let [x1, y1, x2, y2] = obj.box || [obj.px[0] - 0.05, obj.px[1] - 0.04, obj.px[0] + 0.05, obj.px[1] + 0.04];
      oc = [(x1 + x2) / 2, (y1 + y2) / 2];
      const [a, b] = vb(x1, y1);
      const [c, e] = vb(x2, y2);
      $("obj-brackets").setAttribute("d", brackets(a, b, c, e));
      vis(objG, !(false));
      toggle(objG, "is-remembered", !obj.visible);
      toggle(objLabel, "is-remembered", !obj.visible);
      setText(objLabel, obj.visible ? target : `${target}, last seen here`);
      vis(objLabel, !(false));
      placeTag(objLabel, x1, y1, 0, -8);
    } else {
      vis(objG, !(true));
      vis(objLabel, !(true));
    }

    const handG = $("hand");
    const handLabel = $("hand-label");
    const hc = hand.px;
    if (hc) {
      const [x, y] = vb(hc[0], hc[1]);
      handG.setAttribute("transform", `translate(${x} ${y})`);
      vis(handG, !(false));
      const noDepth = !hand.xyz;
      toggle(handG, "no-depth", noDepth);
      vis(handLabel, !(!noDepth));
      if (noDepth) {
        setText(handLabel, "Too close for lidar");
        placeTag(handLabel, hc[0], hc[1], 22, 40);
      }
    } else {
      vis(handG, !(true));
      vis(handLabel, !(true));
    }

    const leg = $("leg");
    const legLabel = $("leg-label");
    if (oc && hc && d.dist != null) {
      const [x1, y1] = vb(hc[0], hc[1]);
      const [x2, y2] = vb(oc[0], oc[1]);
      const len = Math.hypot(x2 - x1, y2 - y1) || 1;
      const pull = Math.min(26, len * 0.3);
      leg.setAttribute("x1", x1 + ((x2 - x1) / len) * 30);
      leg.setAttribute("y1", y1 + ((y2 - y1) / len) * 30);
      leg.setAttribute("x2", x2 - ((x2 - x1) / len) * pull);
      leg.setAttribute("y2", y2 - ((y2 - y1) / len) * pull);
      vis(leg, !(false));
      const close = d.dist < GRAB_M;
      toggle(leg, "is-close", close);
      leg.setAttribute("marker-end", close ? "url(#tip-close)" : "url(#tip)");
      toggle(legLabel, "is-close", close);
      setText(legLabel, cm(d.dist));
      vis(legLabel, !(false));
      placeLegTag(legLabel, hc, oc, obj.box, d.dist);
    } else {
      vis(leg, !(true));
      vis(legLabel, !(true));
    }
  }

  // video chrome ------------------------------------------------------------

  function renderClock(d) {
    const run = d.run || { status: "idle" };
    const base = d.trials && d.trials.baseline;
    const median = base && base.median_s;
    const bug = $("clock");
    const sub = $("clock-sub");
    toggle(bug, "is-done", run.status === "done");
    setText($("clock-time"), clock(run.elapsed_s || 0));
    if (run.status === "idle") {
      setText(sub, "Waiting for a request");
    } else if (run.status === "done") {
      setText(sub, median ? `Grabbed. Unaided median ${clock(median)}` : "Grabbed");
    } else {
      setText(sub, median ? `Unaided median ${clock(median)}` : `Finding the ${d.target || "object"}`);
    }
    const budget = $("budget");
    vis(budget, !(!median || run.status === "idle"));
    if (median) $("budget-fill").style.setProperty("--p", Math.min(1, (run.elapsed_s || 0) / median));
  }

  function renderCaptions(d) {
    const phrases = d.phrases || [];
    const now = $("cap-now");
    const prev = $("cap-prev");
    const last = phrases[phrases.length - 1];
    const before = phrases[phrases.length - 2];
    const key = last ? `${last.t}|${last.text}` : "";
    if (key === captionKey) return;
    const firstRender = captionKey === "";
    captionKey = key;
    toggle(now, "is-quiet", !last);
    setText($("cap-text"), last ? `“${last.text}”` : "Nothing said yet");
    setText(prev, before ? `“${before.text}”` : "");
    if (last && !firstRender) {
      now.classList.add("is-fresh");
      clearTimeout(freshTimer);
      freshTimer = setTimeout(() => now.classList.remove("is-fresh"), 1400);
    }
  }

  function renderBanners(d) {
    const fake = Object.entries(d.modes || {}).filter(([, v]) => v === "fake").map(([k]) => LANE_NAMES[k] || k);
    const sim = $("sim");
    vis(sim, !(fake.length === 0));
    if (fake.length) {
      const keys = d.modes.perception === "fake" ? ". Arrow keys, W and S move the hand." : "";
      setText(sim, `Simulated ${fake.join(", ")}${keys}`);
    }
    const stale = $("stale");
    const age = d.age_ms || 0;
    vis(stale, !(age < STALE_MS));
    if (age >= STALE_MS) setText(stale, `No new camera frames for ${tenth.format(age / 1000)} s`);
  }

  // guidance column -----------------------------------------------------------

  const MOVES = [
    { pos: "Right", neg: "Left" },
    { pos: "Down", neg: "Up" },
    { pos: "Forward", neg: "Back" },
  ];
  // Reach gets its own glyphs (toward or away from a far edge) so it never reads as Height.
  const ARROWS = {
    Right: "M5 12h14M13 6l6 6-6 6",
    Left: "M19 12H5M11 6l-6 6 6 6",
    Up: "M12 19V5M6 11l6-6 6 6",
    Down: "M12 5v14M6 13l6 6 6-6",
    Forward: "M5 4h14M12 21V9M7 14l5-5 5 5",
    Back: "M5 20h14M12 3v12M7 10l5 5 5-5",
  };

  function renderGuide(d) {
    const state = d.state || "idle";
    const stateEl = $("state");
    setText(stateEl, STATE_WORDS[state] || state);
    toggle(stateEl, "is-grab", state === "grab");
    toggle(stateEl, "is-quiet", QUIET_STATES.has(state));

    const wrap = $("dist-wrap");
    const hasDist = d.dist != null;
    setText($("dist"), hasDist ? cm(d.dist) : "–");
    toggle(wrap, "is-empty", !hasDist);
    toggle(wrap, "is-close", hasDist && d.dist < GRAB_M);
    const thing = d.target || "object";
    setText(wrap.querySelector(".dist-label"), hasDist ? `between fingertip and ${thing}` : `Needs both the fingertip and the ${thing}`);

    document.querySelectorAll(".move").forEach((row, i) => {
      const text = row.querySelector(".move-text");
      const m = MOVES[i];
      const v = d.err ? d.err[i] : null;
      toggle(row, "is-empty", v == null);
      toggle(row, "is-ok", v != null && Math.abs(v) <= DEAD_M);
      if (v == null) {
        setText(text, "–");
      } else if (Math.abs(v) <= DEAD_M) {
        setText(text, "On target");
      } else {
        const word = v > 0 ? m.pos : m.neg;
        setText(text, `${word} ${cm(v)}`);
        const path = row.querySelector(".arrow path");
        if (path.getAttribute("d") !== ARROWS[word]) path.setAttribute("d", ARROWS[word]);
      }
    });

    const a = d.audio || {};
    const on = Boolean(a.tone);
    for (const [id, val] of [["pan", a.pan], ["pitch", a.pitch]]) {
      const meter = $(id);
      toggle(meter, "is-off", !on);
      $(`${id}-mark`).style.setProperty("--pos", on ? (Math.max(-1, Math.min(1, val || 0)) + 1) / 2 : 0.5);
    }
    const beeps = $("beeps");
    const beepOn = on && !a.steady && a.beep_hz > 0;
    toggle(beeps, "is-off", !on);
    toggle(beeps, "is-steady", on && a.steady);
    toggle(beeps, "is-beeping", beepOn);
    if (beepOn) beeps.style.setProperty("--period", `${(1 / a.beep_hz).toFixed(3)}s`);
    setText($("beep-text"), !on ? "Silent" : a.steady ? "Steady tone, grab now" : `Beeping ${tenth.format(a.beep_hz)} times a second`);
    vis($("audio-note"), !(d.audio_source !== "derived"));
  }

  // system rail ---------------------------------------------------------------

  function step(name, status, msText, cls) {
    const li = document.querySelector(`[data-step="${name}"]`);
    setText(li.querySelector(".step-status"), status);
    setText(li.querySelector(".step-ms"), msText);
    li.className = cls || "";
  }

  function renderSystem(d) {
    const st = d.stages || {};
    const modes = d.modes || {};
    const fake = (lane) => modes[lane] === "fake";
    const timing = (lane, v) => (fake(lane) ? "Simulated" : ms(v));
    const anyFake = Object.values(modes).includes("fake");
    const target = d.target || "object";
    const age = d.age_ms;
    step("camera", age >= STALE_MS ? "No new frames" : "Color and depth, in sync", timing("stream", age), age >= STALE_MS ? "is-warn" : "is-ok");

    const obj = d.object;
    if (obj.visible && obj.xyz) step("object", `Found the ${target}`, "", "is-ok");
    else if (obj.xyz) step("object", "Covered by the arm, using its last position", "", "is-warn");
    else step("object", `Looking for the ${target}`, "", "is-idle");

    const hand = d.hand;
    if (hand.xyz) step("hand", "Found", "", "is-ok");
    else if (hand.visible) step("hand", "In view, too close for lidar", "", "is-warn");
    else step("hand", "Not in view", "", "is-idle");

    const models = d.models;
    if (models) {
      const names = [models.object, models.hand].filter(Boolean).join(" and ");
      if (names) setText(document.querySelector('[data-step="object"] .step-name'), `Find the object (${names})`);
    }
    step("depth", d.dist != null ? `${cm(d.dist)} apart; steps 2 to 4 run as one pass` : "Waiting for both points", timing("perception", st.perception), d.dist != null ? "is-ok" : "is-idle");
    step("guide", STATE_WORDS[d.state] || d.state || "Waiting", timing("guidance", st.guidance), QUIET_STATES.has(d.state) ? "is-idle" : "is-ok");

    const a = d.audio || {};
    const sound = a.tone ? (a.steady ? "Steady tone" : "Panned tone with beeps") : "Silent between phrases";
    step("sound", sound, "", a.tone ? "is-ok" : "is-idle");

    setText($("e2e"), anyFake ? "" : ms(d.e2e_ms) || "–");
    toggle($("e2e").parentElement, "is-pending", anyFake);
    setText($("e2e-label"), anyFake
      ? "end to end, needs real modules"
      : d.e2e_label === "proc" ? "processing time; frames carry no arrival stamp" : "from frame arrival to sound");
    setText($("fps"), d.fps ? `Loop running at ${tenth.format(d.fps)} frames a second` : "");

    if (d.host) setText($("host"), d.host);
    const net = $("net");
    toggle(net, "is-offline", d.online === false);
    setText(net, d.online === false ? "Internet unplugged, still running" : d.online ? "Internet connected" : "Checking the internet connection…");
    vis($("offline-bug"), d.online === false);

    renderTrials(d.trials);
    renderMap(d);
  }

  function renderTrials(t) {
    const el = $("trials");
    if (!t || (t.baseline.n === 0 && t.guided.n === 0)) {
      if (!el.dataset.empty) {
        el.dataset.empty = "1";
        el.innerHTML = "No timed trials yet. Run <code>python trials.py baseline</code> to record unaided times.";
      }
      return;
    }
    delete el.dataset.empty;
    const line = (name, r) => {
      const p = document.createElement("span");
      p.style.display = "block";
      const b = document.createElement("strong");
      b.textContent = name;
      p.append(b, r.n === 0 ? " no trials yet" : ` ${r.successes} of ${r.n} found${r.median_s != null ? `, median ${clock(r.median_s)}` : ""}`);
      return p;
    };
    el.replaceChildren(line("Unaided:", t.baseline), line("Guided:", t.guided));
  }

  function mapPoint(xyz) {
    const x = Math.max(-48, Math.min(48, xyz[0] * 100));
    const y = Math.max(26, Math.min(MAP_Z0, MAP_Z0 - xyz[2] * 100));
    return [x, y];
  }

  function renderMap(d) {
    const o = $("map-obj");
    const h = $("map-hand");
    const l = $("map-leg");
    let op = null;
    let hp = null;
    if (d.object.xyz) {
      op = mapPoint(d.object.xyz);
      o.setAttribute("x", op[0] - 3);
      o.setAttribute("y", op[1] - 3);
      toggle(o, "is-remembered", !d.object.visible);
    }
    vis(o, !(!op));
    if (d.hand.xyz) {
      hp = mapPoint(d.hand.xyz);
      h.setAttribute("cx", hp[0]);
      h.setAttribute("cy", hp[1]);
    }
    vis(h, !(!hp));
    vis(l, !(!(op && hp)));
    if (op && hp) {
      l.setAttribute("x1", hp[0]);
      l.setAttribute("y1", hp[1]);
      l.setAttribute("x2", op[0]);
      l.setAttribute("y2", op[1]);
      toggle(l, "is-close", d.dist != null && d.dist < GRAB_M);
    }
  }

  // frames ----------------------------------------------------------------------

  // Images exist only once the stream has a frame for them, so none ever renders empty.
  function image(id, alt, parent, before, w, h) {
    let img = $(id);
    if (!img) {
      img = document.createElement("img");
      img.id = id;
      img.alt = alt;
      img.decoding = "sync";
      img.width = w;
      img.height = h;
      parent.insertBefore(img, before);
    }
    return img;
  }

  function renderImages(d) {
    if (d.aspect && Math.abs(d.aspect - aspect) > 1e-3) {
      aspect = d.aspect;
      video.style.setProperty("--aspect", String(aspect));
      overlay.setAttribute("viewBox", `0 0 1000 ${Math.round(1000 / aspect)}`);
    }
    if (d.img && d.img !== shownImg) {
      image("frame", "Live view from the wearer’s chest camera", video, video.firstChild, 480, 640).src = d.img;
      shownImg = d.img;
    }
    if (d.depth && d.depth !== shownDepth) {
      const pip = $("pip");
      image("depth", "Lidar depth from the same frame, near is pale, far is dark", pip, pip.firstChild, 180, 240).src = d.depth;
      shownDepth = d.depth;
      vis(pip, true);
    }
  }

  function render() {
    scheduled = false;
    const d = latest;
    if (!d) return;
    vis($("notice"), !(true));
    renderImages(d);
    toggle(video, "is-grab", d.state === "grab");
    renderOverlay(d);
    renderClock(d);
    renderCaptions(d);
    renderBanners(d);
    renderGuide(d);
    renderSystem(d);
  }

  function schedule() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(render);
    }
  }

  // wiring ------------------------------------------------------------------------

  function connect() {
    const notice = $("notice");
    const es = new EventSource("/events");
    es.onopen = () => {
      connected = true;
      if (!latest) setText(notice, "Waiting for the first frame from main.py…");
    };
    es.onmessage = (e) => {
      latest = JSON.parse(e.data);
      schedule();
    };
    es.onerror = () => {
      if (connected || !latest) {
        vis(notice, !(false));
        setText(notice, "Lost the connection to main.py. Retrying every few seconds.");
      }
      connected = false;
    };
  }

  function sendKey(key) {
    fetch("/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    }).catch(() => {});
  }

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "t") return sendKey("t");
    const key = KEYS[e.key] || KEYS[e.key.toLowerCase()];
    if (!key || !latest || !latest.modes || latest.modes.perception !== "fake") return;
    e.preventDefault();
    sendKey(key);
  });

  new ResizeObserver(schedule).observe(video);

  connect();
})();
