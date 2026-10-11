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
  let aspect = 0;
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
    toggle($("clock"), "is-done", run.status === "done");
    setText($("clock-time"), clock(run.elapsed_s || 0));
    setText($("clock-sub"), run.status === "done" ? "Grabbed in" : "Time");
  }

  function renderBanners(d) {
    const fake = Object.entries(d.modes || {}).filter(([, v]) => v === "fake").map(([k]) => LANE_NAMES[k] || k);
    const sim = $("sim");
    vis(sim, !(fake.length === 0));
    if (fake.length) {
      setText(sim, fake.length === 4 ? "Simulated input" : `Simulated ${fake.join(", ")}`);
      sim.title = d.modes.perception === "fake" ? "Arrow keys, W and S move the simulated hand" : "";
    }
    const stale = $("stale");
    const age = d.age_ms || 0;
    vis(stale, !(age < STALE_MS));
    if (age >= STALE_MS) setText(stale, `No new camera frames for ${tenth.format(age / 1000)} s`);
  }

  // guidance column -----------------------------------------------------------

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
    const thing = d.target || "item";
    setText(wrap.querySelector(".dist-label"), hasDist ? `to the ${thing}` : `Needs the fingertip and the ${thing}`);
    setText($("target"), d.heard ? `“${d.heard}”` : d.target || "nothing yet");

    renderSpoken(d);
  }

  // spoken guidance -------------------------------------------------------------

  const DIRECTION_WORDS = new Set(["Left", "Right", "Up", "Down", "Forward", "Back"]);
  const FRESH_MS = 2500;

  // Why the latest word was said, from the measured positions, so a judge can check it.
  function whyFor(text, d) {
    const thing = d.target || "item";
    const e = d.err;
    const off = (axis) => (e ? cm(e[axis]) : "");
    switch (text) {
      case "Right": return e ? `Hand is ${off(0)} left of the ${thing}` : "";
      case "Left": return e ? `Hand is ${off(0)} right of the ${thing}` : "";
      case "Down": return e ? `Hand is ${off(1)} above the ${thing}` : "";
      case "Up": return e ? `Hand is ${off(1)} below the ${thing}` : "";
      case "Forward": return e ? `Hand is ${off(2)} short of the ${thing}` : "";
      case "Back": return e ? `Hand is ${off(2)} past the ${thing}` : "";
      case "Looking": return `Searching for the ${thing}`;
      case "Found it": return `The ${thing} is in view`;
      case "Show me your hand": return "The fingertip is not in view";
      case "Grab": return "Fingertip within 5 cm";
      case "I lost it": return `The ${thing} left the view`;
      case "I can't see that": return `No ${thing} found yet`;
      default: return "";
    }
  }

  function renderSpoken(d) {
    const phrases = d.phrases || [];
    const last = phrases[phrases.length - 1];
    const fresh = last && (d.sent - last.t) * 1000 < FRESH_MS;
    const now = $("say-now");
    const word = fresh ? last.text : "Quiet";
    setText($("say-word"), word);
    toggle(now, "is-quiet", !fresh);
    toggle(now, "is-phrase", fresh && !DIRECTION_WORDS.has(word));
    toggle(now, "is-grab", fresh && word === "Grab");
    if (fresh && DIRECTION_WORDS.has(word)) {
      const path = now.querySelector(".arrow path");
      if (path.getAttribute("d") !== ARROWS[word]) path.setAttribute("d", ARROWS[word]);
    }
    setText($("say-why"), fresh ? whyFor(word, d) : last ? "Between instructions" : "Nothing said yet");

    // earlier instructions, newest first, repeats collapsed ("Right ×5")
    const runs = [];
    for (const p of phrases.slice(0, -1).reverse()) {
      const r = runs[runs.length - 1];
      if (r && r.text === p.text) r.n += 1;
      else runs.push({ text: p.text, n: 1 });
    }
    const items = runs.slice(0, 5).map((r) => (r.n > 1 ? `${r.text} ×${r.n}` : r.text));
    const list = $("said");
    const key = items.join("|");
    if (list.dataset.key !== key) {
      list.dataset.key = key;
      list.replaceChildren(...items.map((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        return li;
      }));
    }
  }

  // system rail ---------------------------------------------------------------

  function renderSystem(d) {
    const anyFake = Object.values(d.modes || {}).includes("fake");
    setText($("e2e"), anyFake ? "Simulated" : ms(d.e2e_ms) || "–");
    setText($("e2e-label"), d.e2e_label === "proc" && !anyFake ? "Processing only" : "Camera to earbuds");
    toggle($("net-wrap"), "is-offline", d.online === false);
    setText($("net"), d.online === false ? "Unplugged, still running" : d.online ? "Connected" : "Checking…");
    vis($("offline-bug"), d.online === false);
    toggle($("hero"), "is-grab", d.state === "grab");
    scene3d.update(d);
  }

  // 3D scene --------------------------------------------------------------------
  // The latest lidar points, levelled on the tabletop (the server fits the plane) and seen
  // from behind and above the wearer, like a third-person game camera. A 10 cm grid on the
  // table gives scale; drop lines and a shadow show how high the hand is; labels name what
  // each mark is. Points are splatted into an offscreen canvas with a depth buffer: no
  // WebGL and no library, so it runs with the network unplugged. Drag to look around,
  // double-click to go back to the default view.
  const HAND_OPEN = ['M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2', 'M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2', 'M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8', 'M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15'].map((d) => new Path2D(d));
  const HAND_GRAB = ['M18 11.5V9a2 2 0 0 0-2-2a2 2 0 0 0-2 2v1.4', 'M14 10V8a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2', 'M10 9.9V9a2 2 0 0 0-2-2a2 2 0 0 0-2 2v5', 'M6 14a2 2 0 0 0-2-2a2 2 0 0 0-2 2', 'M18 11a2 2 0 1 1 4 0v3a8 8 0 0 1-8 8h-4a8 8 0 0 1-8-8 2 2 0 1 1 4 0'].map((d) => new Path2D(d));

  const scene3d = (() => {
    const canvas = $("scene");
    const note = $("scene-note");
    const ctx = canvas.getContext("2d");
    const layer = document.createElement("canvas");
    const lctx = layer.getContext("2d");
    const OBJ_HALF = [0.045, 0.05, 0.045];
    const GRID = { x0: -0.4, x1: 0.4, z0: 0.1, z1: 0.9, step: 0.1 };
    const VIEW = { az: 0.38, el: 0.6, dist: 0.92 }; // behind-right of the wearer, looking down ~34 degrees
    // near to far: pale blue to deep navy (sRGB, packed little-endian for the Uint32 view)
    const pack = (r, g, b) => (255 << 24) | (b << 16) | (g << 8) | r;
    const RAMP = Array.from({ length: 64 }, (_, i) => {
      const t = i / 63;
      return pack(Math.round(196 + (22 - 196) * t), Math.round(228 + (46 - 228) * t), Math.round(255 + (84 - 255) * t));
    });
    const OBJ_PX = pack(236, 246, 255);

    const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    function rgbOf(cssVar) {
      probe.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(cssVar).trim();
      probe.fillRect(0, 0, 1, 1);
      return Array.from(probe.getImageData(0, 0, 1, 1).data.slice(0, 3));
    }
    const C = {};
    for (const k of ["deep", "text", "muted", "faint", "rule", "leg", "grab", "ground", "on-leg", "on-grab"]) {
      C[k] = `rgb(${rgbOf(`--${k}`).join(",")})`;
    }
    const font = (px, weight = 600) => `${weight} ${px * dpr}px ${getComputedStyle(document.body).fontFamily}`;

    let w = 0, h = 0, dpr = 1, image = null, u32 = null, zbuf = null;
    let cam = null, cols = null, n = 0;        // camera-frame points and colours
    let world = null, height = null;           // levelled points and their height above the table
    let R = null, tableH = 0.45, tableFound = false;
    let obj = null, objSeen = true, hand = null, close = false, dist = null, target = "object";
    let az = VIEW.az, el = VIEW.el;
    const focus = [0, -0.4, 0.45];
    let dragging = false, lastX = 0, lastY = 0, dirty = true, lastDraw = 0;

    function size() {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      const nw = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const nh = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (nw === w && nh === h) return;
      w = canvas.width = layer.width = nw;
      h = canvas.height = layer.height = nh;
      image = lctx.createImageData(w, h);
      u32 = new Uint32Array(image.data.buffer);
      zbuf = new Float32Array(w * h);
      dirty = true;
    }

    // camera axes (x right, y down, z forward) -> level axes (x right, y up from the lens, z forward)
    function level(p) {
      return [
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
      ];
    }

    function relevel() {
      if (!cam || !R) return;
      world = new Float32Array(n * 3);
      height = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const q = level([cam[i * 3], cam[i * 3 + 1], cam[i * 3 + 2]]);
        world[i * 3] = q[0]; world[i * 3 + 1] = q[1]; world[i * 3 + 2] = q[2];
        height[i] = q[1] + tableH;
      }
    }

    function decode(c) {
      const bin = atob(c.b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      n = c.n;
      const mm = new Int16Array(bytes.buffer, 0, n * 3);
      cam = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) cam[i] = mm[i] / 1000;
      cols = new Uint8Array(bytes.buffer, n * 6, n * 3);
      R = c.R;
      tableH = c.h;
      tableFound = c.table;
      relevel();
      dirty = true;
    }

    // orbit camera looking at the focus point; world y is up
    let eye, fx, fy, fz, rx, rz, ux, uy, uz, F, cx, cy;
    function setView() {
      const ce = Math.cos(el);
      eye = [focus[0] + VIEW.dist * Math.sin(az) * ce, focus[1] + VIEW.dist * Math.sin(el), focus[2] - VIEW.dist * Math.cos(az) * ce];
      fx = focus[0] - eye[0]; fy = focus[1] - eye[1]; fz = focus[2] - eye[2];
      const fl = Math.hypot(fx, fy, fz);
      fx /= fl; fy /= fl; fz /= fl;
      const rl = Math.hypot(fz, fx);
      rx = fz / rl; rz = -fx / rl;                    // right: horizontal, perpendicular to view
      ux = -fy * fx; uy = 1 - fy * fy; uz = -fy * fz; // world up, minus its part along the view
      const ul = Math.hypot(ux, uy, uz);
      ux /= ul; uy /= ul; uz /= ul;
      F = 1.1 * Math.min(w, h * 1.5);
      cx = w * 0.6;   // the spoken word sits top-left, so the scene sits right of centre
      cy = h * 0.56;
    }
    function project(x, y, z, out) {
      const dx = x - eye[0], dy = y - eye[1], dz = z - eye[2];
      const zc = dx * fx + dy * fy + dz * fz;
      if (zc < 0.05) return false;
      out[0] = cx + (F * (dx * rx + dz * rz)) / zc;
      out[1] = cy - (F * (dx * ux + dy * uy + dz * uz)) / zc;
      out[2] = zc;
      return true;
    }
    const tmp = [0, 0, 0];
    const at = (p) => (project(p[0], p[1], p[2], tmp) ? [tmp[0], tmp[1], tmp[2]] : null);
    function seg(a, b) {
      const p = at(a), q = at(b);
      if (!p || !q) return;
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(q[0], q[1]);
    }
    function stroke(color, width, dash = []) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width * dpr;
      ctx.setLineDash(dash.map((v) => v * dpr));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    function tag(text, p, bg, fg, dx = 0, dy = 0) {
      if (!p) return;
      ctx.font = font(15);
      const tw = ctx.measureText(text).width;
      const pad = 6 * dpr, th = 22 * dpr;
      const x = Math.min(Math.max(p[0] + dx * dpr - tw / 2 - pad, 2 * dpr), w - tw - 2 * pad - 2 * dpr);
      const y = Math.min(Math.max(p[1] + dy * dpr - th / 2, 2 * dpr), h - th - 2 * dpr);
      ctx.fillStyle = bg;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, tw + 2 * pad, th, 3 * dpr);
      else ctx.rect(x, y, tw + 2 * pad, th);
      ctx.fill();
      ctx.fillStyle = fg;
      ctx.textBaseline = "middle";
      ctx.fillText(text, x + pad, y + th / 2 + dpr / 2);
    }

    function drawPoints() {
      u32.fill(0);
      zbuf.fill(Infinity);
      const out = tmp;
      for (let i = 0; i < n; i++) {
        if (!project(world[i * 3], world[i * 3 + 1], world[i * 3 + 2], out)) continue;
        // a perception view, not a photo: the tabletop is a blue ramp that darkens with
        // distance, and anything standing on the table is drawn near-white so it pops
        const raised = height[i] >= 0.02;
        const f = Math.min(1, Math.max(0, (out[2] - VIEW.dist + 0.45) / 1.1));
        const c = raised ? OBJ_PX : RAMP[Math.round(f * 63)];
        const s = Math.max(1, Math.round(((raised ? 2.4 : 1.8) * dpr * VIEW.dist) / out[2]));
        const x0 = Math.round(out[0] - s / 2), y0 = Math.round(out[1] - s / 2);
        for (let dy = 0; dy < s; dy++) {
          const yy = y0 + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = 0; dx < s; dx++) {
            const xx = x0 + dx;
            if (xx < 0 || xx >= w) continue;
            const j = yy * w + xx;
            if (out[2] < zbuf[j]) { zbuf[j] = out[2]; u32[j] = c; }
          }
        }
      }
      lctx.putImageData(image, 0, 0);
    }

    function draw() {
      if (!w) return;
      setView();
      ctx.fillStyle = C.deep;
      ctx.fillRect(0, 0, w, h);
      const T = -tableH;

      // 10 cm grid on the tabletop
      ctx.beginPath();
      for (let x = GRID.x0; x <= GRID.x1 + 1e-6; x += GRID.step) seg([x, T, GRID.z0], [x, T, GRID.z1]);
      for (let z = GRID.z0; z <= GRID.z1 + 1e-6; z += GRID.step) seg([GRID.x0, T, z], [GRID.x1, T, z]);
      stroke(C.rule, 1);

      if (world) {
        drawPoints();
        ctx.drawImage(layer, 0, 0);
      }

      const o = obj && level(obj);
      const hp = hand && level(hand);

      if (o) {
        const [hx, hy, hz] = OBJ_HALF;
        const bottom = Math.max(T, o[1] - hy);
        const v = [];
        for (const sx of [-1, 1]) for (const sy of [bottom, o[1] + hy]) for (const sz of [-1, 1]) v.push([o[0] + sx * hx, sy, o[2] + sz * hz]);
        ctx.beginPath();
        for (const [a, b] of [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]) seg(v[a], v[b]);
        stroke(objSeen ? C.text : C.muted, 2, objSeen ? [] : [5, 4]);
        tag(objSeen ? target : `${target}, last seen`, at([o[0], o[1] + hy, o[2]]), objSeen ? C.text : C.muted, C.ground, 0, -16);
      }

      if (hp) {
        // shadow on the table and a drop line, so the height reads
        const ring = [];
        for (let i = 0; i <= 24; i++) {
          const a = (i / 24) * Math.PI * 2;
          ring.push(at([hp[0] + 0.025 * Math.cos(a), T, hp[2] + 0.025 * Math.sin(a)]));
        }
        if (ring.every(Boolean)) {
          ctx.beginPath();
          ring.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          ctx.fillStyle = C.ground;
          ctx.fill();
        }
        ctx.beginPath();
        seg(hp, [hp[0], T, hp[2]]);
        stroke(C.text, 1.5, [3, 3]);
      }

      if (o && hp) {
        ctx.beginPath();
        seg(hp, o);
        ctx.lineCap = "round";
        stroke(close ? C.grab : C.leg, 3.5);
        if (dist != null) {
          const mid = at([(hp[0] + o[0]) / 2, (hp[1] + o[1]) / 2, (hp[2] + o[2]) / 2]);
          tag(cm(dist), mid, close ? C.grab : C.leg, close ? C["on-grab"] : C["on-leg"], 0, 16);
        }
      }

      if (hp) {
        const p = at(hp);
        if (p) {
          // the Proprio hand (same as the deck), anchored so its reticle sits on the tracked point
          const img = close ? handGrabImg : handOpenImg;
          if (img.complete && img.naturalWidth) {
            const size = Math.max(56 * dpr, (F * 0.1) / p[2]);
            ctx.drawImage(img, p[0] - size * 0.6022, p[1] - size * 0.2400, size, size);
          }
        }
      }
    }

    const handTemplate = ["<svg class=\"h\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-30 -30 316 316\" width=\"316\" height=\"316\"><g transform=\"rotate(26 128 128)\"><path d=\"M216,64v90.93c0,46.2-36.85,84.55-83,85.06A83.71,83.71,0,0,1,72.6,215.4C50.79,192.33,26.15,136,26.15,136a16,16,0,0,1,6.53-22.23c7.66-4,17.1-.84,21.4,6.62l21,36.44a6.09,6.09,0,0,0,6,3.09l.12,0A8.19,8.19,0,0,0,88,151.74V48a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V112a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V32a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V120a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V64.45c0-8.61,6.62-16,15.23-16.43A16,16,0,0,1,216,64Z\" transform=\"translate(14 18)\" fill=\"__SHADE__\"/><path d=\"M216,64v90.93c0,46.2-36.85,84.55-83,85.06A83.71,83.71,0,0,1,72.6,215.4C50.79,192.33,26.15,136,26.15,136a16,16,0,0,1,6.53-22.23c7.66-4,17.1-.84,21.4,6.62l21,36.44a6.09,6.09,0,0,0,6,3.09l.12,0A8.19,8.19,0,0,0,88,151.74V48a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V112a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V32a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V120a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V64.45c0-8.61,6.62-16,15.23-16.43A16,16,0,0,1,216,64Z\" fill=\"__BODY__\" stroke=\"__GROUND__\" stroke-width=\"12\" paint-order=\"stroke\"/><g transform=\"translate(121 40) rotate(-26)\" fill=\"none\" stroke=\"__RET__\" stroke-width=\"4\" stroke-linecap=\"round\"><circle r=\"17\"/><path d=\"M-23 0h-12M23 0h12M0 -23v-12M0 23v12\"/></g></g></svg>", "<svg class=\"h\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-30 -30 316 316\" width=\"316\" height=\"316\"><g transform=\"rotate(26 128 128)\"><path d=\"M216,64v90.93c0,46.2-36.85,84.55-83,85.06A83.71,83.71,0,0,1,72.6,215.4C50.79,192.33,26.15,136,26.15,136a16,16,0,0,1,6.53-22.23c7.66-4,17.1-.84,21.4,6.62l21,36.44a6.09,6.09,0,0,0,6,3.09l.12,0A8.19,8.19,0,0,0,88,151.74V48a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V112a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V32a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V120a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V64.45c0-8.61,6.62-16,15.23-16.43A16,16,0,0,1,216,64Z\" transform=\"translate(14 18)\" fill=\"__GSHADE__\"/><path d=\"M216,64v90.93c0,46.2-36.85,84.55-83,85.06A83.71,83.71,0,0,1,72.6,215.4C50.79,192.33,26.15,136,26.15,136a16,16,0,0,1,6.53-22.23c7.66-4,17.1-.84,21.4,6.62l21,36.44a6.09,6.09,0,0,0,6,3.09l.12,0A8.19,8.19,0,0,0,88,151.74V48a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V112a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V32a16,16,0,0,1,16.77-16c8.61.4,15.23,7.82,15.23,16.43V120a8,8,0,0,0,8.53,8,8.17,8.17,0,0,0,7.47-8.25V64.45c0-8.61,6.62-16,15.23-16.43A16,16,0,0,1,216,64Z\" fill=\"__GRAB__\" stroke=\"__GROUND__\" stroke-width=\"12\" paint-order=\"stroke\"/><g transform=\"translate(121 40) rotate(-26)\" fill=\"none\" stroke=\"__RET__\" stroke-width=\"4\" stroke-linecap=\"round\"><circle r=\"9\"/><circle r=\"3.5\" fill=\"__RET__\" stroke=\"none\"/></g></g></svg>"];
    const handImage = (t) => {
      const svg = t.replaceAll("__BODY__", C.text).replaceAll("__GRAB__", C.grab).replaceAll("__GROUND__", C.ground)
        .replaceAll("__SHADE__", "rgb(36,52,78)").replaceAll("__GSHADE__", "rgb(30,82,52)").replaceAll("__RET__", "rgb(205,226,245)");
      const im = new Image();
      im.onload = () => { dirty = true; };
      im.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
      return im;
    };
    const handOpenImg = handImage(handTemplate[0]);
    const handGrabImg = handImage(handTemplate[1]);

    function tick(now) {
      requestAnimationFrame(tick);
      if (now - lastDraw < 33) return;
      // keep the reach area centred: between the wearer's spot on the table and the object
      const goal = obj ? level(obj) : null;
      const want = goal ? [goal[0] * 0.5, -tableH + 0.05, goal[2] * 0.6] : [0, -tableH + 0.05, 0.4];
      for (let i = 0; i < 3; i++) {
        const step = (want[i] - focus[i]) * 0.15;
        if (Math.abs(step) > 1e-4) { focus[i] += step; dirty = true; }
      }
      if (!dirty) return;
      lastDraw = now;
      dirty = false;
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
      az -= (e.clientX - lastX) * 0.008; // the scene turns with the drag, like grabbing it
      el = Math.max(0.12, Math.min(1.45, el + (e.clientY - lastY) * 0.006));
      lastX = e.clientX;
      lastY = e.clientY;
      dirty = true;
    });
    const release = () => { dragging = false; };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("dblclick", () => { az = VIEW.az; el = VIEW.el; dirty = true; });

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
        dist = d.dist;
        close = d.dist != null && d.dist < GRAB_M;
        target = d.target || "object";
        dirty = true;
        setText(note, !n
          ? "Waiting for lidar points…"
          : `View from behind you, 10 cm grid.${tableFound ? "" : " No table found; assuming a 30° tilt."}`);
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
  }

  // demo mode for judges: full-bleed 3D, camera picture-in-picture, poster word, distance meter
  let startDist = 0;
  let runKey = "";
  function renderDemo(d) {
    const run = d.run || {};
    const key = `${run.target}|${run.status === "running"}`;
    if (key !== runKey) { runKey = key; startDist = 0; }
    if (d.dist != null) startDist = Math.max(startDist, d.dist);
    const k = d.dist != null && startDist > 0 ? Math.max(0, Math.min(1, d.dist / startDist)) : 1;
    $("meter-fill").style.setProperty("--k", String(k));
    toggle($("meter"), "is-close", d.dist != null && d.dist < GRAB_M);
    toggle(document.body, "is-grabbed", d.state === "grab");
  }
  function setDemo(on) {
    document.body.classList.toggle("demo", on);
    $("demo-btn").setAttribute("aria-pressed", String(on));
    try { localStorage.setItem("blindfold-demo", on ? "1" : "0"); } catch (e) {}
    window.dispatchEvent(new Event("resize"));
  }
  $("demo-btn").addEventListener("click", () => setDemo(!document.body.classList.contains("demo")));
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === "INPUT") return;
    if (e.key === "d" || e.key === "D") setDemo(!document.body.classList.contains("demo"));
  });
  let demoSaved = false;
  try { demoSaved = localStorage.getItem("blindfold-demo") === "1"; } catch (e) {}
  setDemo(location.hash === "#demo" || (location.hash !== "#camera" && demoSaved));

  function render() {
    scheduled = false;
    const d = latest;
    if (!d) return;
    renderDemo(d);
    vis($("notice"), !(true));
    renderImages(d);
    toggle(video, "is-grab", d.state === "grab");
    renderOverlay(d);
    renderClock(d);
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

  $("new-btn").addEventListener("click", () => sendKey("t"));

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
