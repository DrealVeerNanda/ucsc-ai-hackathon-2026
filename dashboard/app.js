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
    scene3d.update(d);
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

  // 3D scene --------------------------------------------------------------------
  // The latest lidar points with the fingertip, the object and the leg in real 3D. Points
  // are splatted into an ImageData with a depth buffer: no WebGL and no library, so it
  // runs with the network unplugged. It sways slowly so depth reads at a glance; dragging
  // takes over, and reduced motion holds a still three-quarter view.
  const scene3d = (() => {
    const canvas = $("scene");
    const note = $("scene-note");
    const ctx = canvas.getContext("2d");
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    const OBJ_HALF = [0.04, 0.05, 0.04];
    const FRUSTUM = [0.04, 0.053, 0.07]; // half width, half height, depth of the drawn camera
    const SWAY = 0.55; // radians either side
    const SWAY_PERIOD = 14000;
    const DIST = 0.62; // virtual camera distance from the pivot, metres
    const count = new Intl.NumberFormat("en-US");

    const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    function rgbOf(cssVar) {
      probe.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
      probe.fillRect(0, 0, 1, 1);
      return Array.from(probe.getImageData(0, 0, 1, 1).data.slice(0, 3));
    }
    const C = {};
    function readColors() {
      for (const k of ["plate", "text", "muted", "leg", "grab", "ground"]) C[k] = `rgb(${rgbOf(`--${k}`).join(",")})`;
      const [r, g, b] = rgbOf("--plate");
      C.bg = (255 << 24) | (b << 16) | (g << 8) | r;
    }

    let w = 0, h = 0, dpr = 1, image = null, u32 = null, zbuf = null;
    let pts = null, cols = null, n = 0;
    let obj = null, objSeen = true, hand = null, close = false;
    let yaw = 0.4, pitch = 0.5;
    const pivot = [0, 0.1, 0.55];
    let dragging = false, lastX = 0, lastY = 0, lastTouch = -1e9, lastDraw = 0;

    function size() {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      const nw = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const nh = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (nw === w && nh === h) return;
      w = canvas.width = nw;
      h = canvas.height = nh;
      image = ctx.createImageData(w, h);
      u32 = new Uint32Array(image.data.buffer);
      zbuf = new Float32Array(w * h);
    }

    function decode(cloud) {
      const bin = atob(cloud.b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const m = cloud.n;
      const mm = new Int16Array(bytes.buffer, 0, m * 3);
      pts = new Float32Array(m * 3);
      for (let i = 0; i < m * 3; i++) pts[i] = mm[i] / 1000;
      cols = new Uint8Array(bytes.buffer, m * 6, m * 3);
      n = m;
    }

    // camera frame (x right, y down, z forward) -> canvas pixels and depth
    let cyw, syw, cp, sp, F, cx, cy;
    function setView() {
      cyw = Math.cos(yaw); syw = Math.sin(yaw); cp = Math.cos(pitch); sp = Math.sin(pitch);
      F = 1.05 * Math.min(w, h * 1.4);
      cx = w / 2;
      cy = h * 0.52;
    }
    function project(x, y, z, out) {
      x -= pivot[0]; y -= pivot[1]; z -= pivot[2];
      const x1 = cyw * x + syw * z;
      const z1 = -syw * x + cyw * z;
      const y2 = cp * y + sp * z1;
      const z2 = -sp * y + cp * z1 + DIST;
      if (z2 < 0.05) return false;
      out[0] = cx + (F * x1) / z2;
      out[1] = cy + (F * y2) / z2;
      out[2] = z2;
      return true;
    }

    const tmp = [0, 0, 0];
    function at(p) {
      return project(p[0], p[1], p[2], tmp) ? [tmp[0], tmp[1], tmp[2]] : null;
    }
    function line(a, b) {
      const p = at(a), q = at(b);
      if (!p || !q) return;
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(q[0], q[1]);
    }

    function draw() {
      if (!w || !n) return;
      setView();
      u32.fill(C.bg);
      zbuf.fill(Infinity);
      const out = tmp;
      for (let i = 0; i < n; i++) {
        if (!project(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], out)) continue;
        const s = Math.max(1, Math.round((1.9 * dpr * DIST) / out[2]));
        const x0 = Math.round(out[0] - s / 2), y0 = Math.round(out[1] - s / 2);
        const c = (255 << 24) | ((cols[i * 3 + 2] * 0.85) << 16) | ((cols[i * 3 + 1] * 0.85) << 8) | (cols[i * 3] * 0.85);
        for (let dy = 0; dy < s; dy++) {
          const yy = y0 + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = 0; dx < s; dx++) {
            const xx = x0 + dx;
            if (xx < 0 || xx >= w) continue;
            const k = yy * w + xx;
            if (out[2] < zbuf[k]) { zbuf[k] = out[2]; u32[k] = c; }
          }
        }
      }
      ctx.putImageData(image, 0, 0);

      // the phone on the chest, as a small frustum at the origin
      const [fx, fy, fz] = FRUSTUM;
      const corners = [[-fx, -fy, fz], [fx, -fy, fz], [fx, fy, fz], [-fx, fy, fz]];
      ctx.beginPath();
      for (const c of corners) line([0, 0, 0], c);
      for (let i = 0; i < 4; i++) line(corners[i], corners[(i + 1) % 4]);
      ctx.strokeStyle = C.muted;
      ctx.lineWidth = dpr;
      ctx.setLineDash([]);
      ctx.stroke();
      const o = at([0, 0, 0]);
      if (o) {
        ctx.fillStyle = C.muted;
        ctx.font = `500 ${12 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
        ctx.fillText("Chest camera", o[0] + 8 * dpr, o[1] - 6 * dpr);
      }

      if (obj) {
        const [hx, hy, hz] = OBJ_HALF;
        const v = [];
        for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) v.push([obj[0] + sx * hx, obj[1] + sy * hy, obj[2] + sz * hz]);
        ctx.beginPath();
        for (const [a, b] of [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]) line(v[a], v[b]);
        ctx.strokeStyle = objSeen ? C.text : C.muted;
        ctx.lineWidth = 2 * dpr;
        ctx.setLineDash(objSeen ? [] : [5 * dpr, 4 * dpr]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (obj && hand) {
        ctx.beginPath();
        line(hand, obj);
        ctx.strokeStyle = close ? C.grab : C.leg;
        ctx.lineWidth = 3 * dpr;
        ctx.lineCap = "round";
        ctx.stroke();
      }
      if (hand) {
        const p = at(hand);
        if (p) {
          const r = Math.max(4 * dpr, (F * 0.012) / p[2]);
          ctx.beginPath();
          ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
          ctx.fillStyle = C.text;
          ctx.fill();
          ctx.lineWidth = 2 * dpr;
          ctx.strokeStyle = C.ground;
          ctx.stroke();
        }
      }
    }

    function tick(now) {
      requestAnimationFrame(tick);
      if (now - lastDraw < 33) return;
      lastDraw = now;
      if (!dragging) {
        const free = now - lastTouch > 5000;
        const target = reduced.matches ? 0.4 : free ? SWAY * Math.sin((now / SWAY_PERIOD) * Math.PI * 2) : yaw;
        yaw += (target - yaw) * (reduced.matches ? 1 : 0.05);
      }
      if (obj) for (let i = 0; i < 3; i++) pivot[i] += ((hand ? (obj[i] + hand[i]) / 2 : obj[i]) - pivot[i]) * 0.05;
      draw();
    }

    canvas.addEventListener("pointerdown", (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      yaw -= (e.clientX - lastX) * 0.008;
      pitch = Math.max(0.05, Math.min(1.3, pitch + (e.clientY - lastY) * 0.006));
      lastX = e.clientX;
      lastY = e.clientY;
    });
    const release = () => { dragging = false; lastTouch = performance.now(); };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);

    readColors();
    new ResizeObserver(size).observe(canvas);
    size();
    requestAnimationFrame(tick);

    return {
      cloud(c) {
        if (c && c.n) decode(c);
      },
      update(d) {
        obj = d.object.xyz;
        objSeen = d.object.visible;
        hand = d.hand.xyz;
        close = d.dist != null && d.dist < GRAB_M;
        setText(note, n ? `${count.format(n)} lidar points from the latest frame. Drag to turn it.` : "Waiting for lidar points…");
      },
    };
  })();
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
    es.addEventListener("cloud", (e) => scene3d.cloud(JSON.parse(e.data)));
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
