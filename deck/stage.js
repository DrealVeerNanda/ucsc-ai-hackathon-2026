// One persistent 3D scene behind every slide: a tabletop seen as lidar points, a mug, the
// wearer's chest camera, a fingertip, and the yellow line from fingertip to mug. Slides pick
// a "shot" (camera + a few levels); every level eases toward its target each frame, so moves
// between slides are glides, never cuts. three.js is vendored, so this runs offline.
import * as THREE from "./vendor/three.module.min.js";

// three.js cannot parse oklch(), so resolve each CSS colour to sRGB through a 1px canvas.
const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
function color(name) {
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
  return new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
}

// World: metres, y up, table top at y = 0, the wearer's chest camera at the origin's
// edge of the table, looking forward (+z).
const CHEST = new THREE.Vector3(0, 0.45, 0);
const MUG = new THREE.Vector3(0.09, 0, 0.62);       // base centre of the mug
const MUG_TOP = new THREE.Vector3(0.09, 0.1, 0.62);
const GRAB_AT = new THREE.Vector3(0.09, 0.06, 0.57); // where the fingertip ends: the mug's side
const START = new THREE.Vector3(-0.2, 0.2, 0.32);    // where a reach starts

// Camera shots: position, look-at target, and how far to push the subject right (fraction of
// width) so the slide's text has the left side. Levels are 0..1 and eased.
const SHOTS = {
  title:     { pos: [-0.38, 0.75, -0.35], at: [0.05, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1 } },
  dark:      { pos: [-0.32, 0.55, -0.25], at: [0.0, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, dim: 1, hand: 1, wander: 1 } },
  provide:   { pos: [-0.34, 0.6, -0.3], at: [0.05, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, words: 1 } },
  wide:      { pos: [-0.63, 1.1, -0.9], at: [0.0, 0.1, 0.45], shift: 0.2, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, frustum: 1 } },
  recede:    { pos: [-0.72, 1.3, -1.2], at: [0.0, 0.0, 0.5], shift: 0.0, lv: { reveal: 1, dim: 0.75 } },
  differ:    { pos: [-0.36, 0.65, -0.35], at: [0.05, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, words: 1, frustum: 1 } },
  sweep:     { pos: [-0.43, 0.85, -0.55], at: [0.0, 0.1, 0.45], shift: 0.16, lv: { sweep: 1, frustum: 1 } },
  found:     { pos: [-0.38, 0.75, -0.45], at: [0.05, 0.05, 0.5], shift: 0.16, lv: { reveal: 1, frustum: 1, box: 1, hand: 1, still: 1 } },
  vector:    { pos: [-0.02, 1.0, -0.2], at: [0.0, 0.0, 0.48], shift: 0.2, lv: { reveal: 1, box: 1, hand: 1, still: 1, leg: 1, axes: 1 } },
  speak:     { pos: [-0.34, 0.6, -0.3], at: [0.05, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, words: 1 } },
  off:       { pos: [-0.38, 0.75, -0.35], at: [0.05, 0.05, 0.55], shift: 0.0, lv: { reveal: 1, dim: 1 } },
  local:     { pos: [-0.5, 0.95, -0.7], at: [0.0, 0.15, 0.45], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, link: 1 } },
  cut:       { pos: [-0.5, 0.95, -0.7], at: [0.0, 0.15, 0.45], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, link: 1, cut: 1 } },
  grab:      { pos: [-0.25, 0.42, 0.12], at: [0.08, 0.06, 0.58], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, still: 1, grabbed: 1 } },
};
const LEVELS = ["reveal", "dim", "hand", "leg", "guide", "wander", "words", "frustum", "person", "sweep",
  "box", "still", "axes", "link", "cut", "grabbed"];

export function createStage(canvas, labels) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const scene = new THREE.Scene();
  scene.background = color("--ground");
  // Data is in the wearer's frame (x = wearer's right). A camera behind the wearer looking
  // forward has +x on its left, so mirror the scene once to keep "right" on screen right.
  scene.scale.x = -1;
  const DS = THREE.DoubleSide;
  const camera = new THREE.PerspectiveCamera(34, 16 / 9, 0.01, 20);

  const C = {
    text: color("--text"), muted: color("--muted"), faint: color("--faint"), rule: color("--rule"),
    leg: color("--leg"), grab: color("--grab"), far: new THREE.Color(0x0c1628), near: new THREE.Color(0xc4e4ff),
  };

  // --- lidar points: tabletop and mug ---------------------------------------------------
  const table = [];
  for (let i = 0; i < 9000; i++) {
    const x = (Math.random() - 0.5) * 1.1, z = 0.12 + Math.random() * 1.0;
    table.push([x, 0, z]);
  }
  const mug = [];
  for (let i = 0; i < 1400; i++) {
    const a = Math.random() * Math.PI * 2, y = Math.random() * 0.1;
    mug.push([MUG.x + 0.045 * Math.cos(a), y, MUG.z + 0.045 * Math.sin(a)]);
  }
  for (let i = 0; i < 260; i++) { // handle
    const a = -Math.PI / 2 + Math.random() * Math.PI, r = 0.028;
    mug.push([MUG.x + 0.045 + r * Math.cos(a) * 0.9, 0.05 + r * Math.sin(a), MUG.z + (Math.random() - 0.5) * 0.008]);
  }
  const all = table.concat(mug);
  const n = all.length, nTable = table.length;
  const home = new Float32Array(n * 3), scatter = new Float32Array(n * 3), pos = new Float32Array(n * 3);
  const base = new Float32Array(n * 3), col = new Float32Array(n * 3);
  all.forEach((p, i) => {
    home.set(p, i * 3);
    scatter.set([p[0] + (Math.random() - 0.5) * 0.9, p[1] + Math.random() * 0.8, p[2] + (Math.random() - 0.5) * 0.9], i * 3);
    const c = i < nTable ? C.far.clone().lerp(C.near, 0.15 + 0.5 * (1 - Math.min(1, p[2] / 1.2))) : C.text.clone();
    base.set([c.r, c.g, c.b], i * 3);
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.0065, vertexColors: true, sizeAttenuation: true }));
  scene.add(points);

  // --- 10 cm grid on the tabletop -----------------------------------------------------------
  const gridPts = [];
  for (let x = -0.5; x <= 0.501; x += 0.1) gridPts.push(x, 0, 0.1, x, 0, 1.1);
  for (let z = 0.1; z <= 1.101; z += 0.1) gridPts.push(-0.5, 0, z, 0.5, 0, z);
  const grid = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(gridPts, 3)),
    new THREE.LineBasicMaterial({ color: C.rule, transparent: true, opacity: 0.55 }));
  scene.add(grid);

  // --- chest camera frustum and the wearer --------------------------------------------------
  const fr = [];
  const corners = [[-0.06, -0.1, 0.12], [0.06, -0.1, 0.12], [0.06, -0.01, 0.12], [-0.06, -0.01, 0.12]].map((c) => [c[0], CHEST.y + c[1], c[2]]);
  for (const c of corners) fr.push(CHEST.x, CHEST.y, CHEST.z, ...c);
  for (let i = 0; i < 4; i++) fr.push(...corners[i], ...corners[(i + 1) % 4]);
  const frustum = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(fr, 3)),
    new THREE.LineBasicMaterial({ color: C.muted, transparent: true, opacity: 0 }));
  scene.add(frustum);
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.11, 0.42, 6, 16),
    new THREE.MeshBasicMaterial({ color: C.rule, transparent: true, side: DS, opacity: 0, wireframe: true }));
  body.position.set(0, 0.32, -0.16);
  scene.add(body);

  // --- object box, fingertip, leg ----------------------------------------------------------
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.11, 0.11, 0.11)),
    new THREE.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0 }));
  box.position.set(MUG.x, 0.055, MUG.z);
  scene.add(box);

  const hand = new THREE.Mesh(new THREE.SphereGeometry(0.014, 24, 16), new THREE.MeshBasicMaterial({ color: C.text, transparent: true, side: DS }));
  scene.add(hand);
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.022, 32), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, side: DS, opacity: 0.5 }));
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.0015;
  scene.add(shadow);
  const drop = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({ color: C.text, dashSize: 0.012, gapSize: 0.01, transparent: true }));
  scene.add(drop);

  const legMat = new THREE.MeshBasicMaterial({ color: C.leg, transparent: true, side: DS });
  const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.0045, 1, 12), legMat);
  scene.add(leg);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.012, 0.03, 16), legMat);
  scene.add(tip);

  // x / y / z components of the leg, as a staircase from the fingertip
  const axisMat = new THREE.LineDashedMaterial({ color: C.text, dashSize: 0.01, gapSize: 0.007, transparent: true });
  const axes = new THREE.Line(new THREE.BufferGeometry().setFromPoints([0, 1, 2, 3].map(() => new THREE.Vector3())), axisMat);
  scene.add(axes);

  // a link up to "the cloud" that gets cut
  const linkMat = new THREE.LineDashedMaterial({ color: C.muted, dashSize: 0.03, gapSize: 0.02, transparent: true, opacity: 0 });
  const CLOUD = new THREE.Vector3(-0.5, 0.95, 0.95);
  const link = new THREE.Line(new THREE.BufferGeometry().setFromPoints([CHEST.clone(), CLOUD.clone()]), linkMat);
  scene.add(link);
  const cloud = new THREE.Mesh(new THREE.SphereGeometry(0.05, 20, 14), new THREE.MeshBasicMaterial({ color: C.muted, wireframe: true, transparent: true, side: DS, opacity: 0 }));
  cloud.position.copy(CLOUD);
  scene.add(cloud);

  // lidar sweep plane
  const sweep = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.5), new THREE.MeshBasicMaterial({ color: C.near, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
  scene.add(sweep);

  // --- state -------------------------------------------------------------------------------
  const lv = Object.fromEntries(LEVELS.map((k) => [k, 0]));
  const goal = Object.fromEntries(LEVELS.map((k) => [k, 0]));
  let shot = SHOTS.title;
  const camPos = new THREE.Vector3(...shot.pos), camAt = new THREE.Vector3(...shot.at);
  let shift = shot.shift;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  // ?still snaps every level and the camera to the shot (screenshots, PDF export)
  const still = new URLSearchParams(location.search).has("still");
  const snap = () => reduced.matches || still;
  let t0 = performance.now(), last = t0, sweepT = 0, word = "", wordT = 0;
  const handPos = START.clone();

  function setShot(name) {
    shot = SHOTS[name] || SHOTS.title;
    for (const k of LEVELS) goal[k] = shot.lv[k] || 0;
    if (shot.lv.sweep) sweepT = 0;
    t0 = performance.now();
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // reach loop: a guided hand moves to the mug in steps (one axis at a time, like the words)
  function guidedHand(t) {
    const cycle = 7.5, u = (t % cycle) / cycle;
    const err = new THREE.Vector3().subVectors(GRAB_AT, START);
    const legs = [[0, 0.0, 0.22], [1, 0.22, 0.42], [2, 0.42, 0.72]]; // x, then y, then z
    const p = START.clone();
    let say = "";
    for (const [axis, a, b] of legs) {
      const k = Math.min(1, Math.max(0, (u - a) / (b - a)));
      const e = k * k * (3 - 2 * k);
      p.setComponent(axis, START.getComponent(axis) + err.getComponent(axis) * e);
      if (u >= a && u < b) say = ["Right", err.y < 0 ? "Down" : "Up", "Forward"][axis];
    }
    if (u >= 0.72) say = "Grab";
    return { p, say, grabbed: u >= 0.72 };
  }
  function wanderHand(t) {
    return new THREE.Vector3(-0.05 + 0.28 * Math.sin(t * 0.37) * Math.cos(t * 0.13), 0.14 + 0.06 * Math.sin(t * 0.71), 0.5 + 0.22 * Math.sin(t * 0.23 + 1.3));
  }

  const tmpV = new THREE.Vector3();
  function project(v) {
    tmpV.copy(v);
    tmpV.x *= -1; // the scene is mirrored, see above
    tmpV.project(camera);
    return [(tmpV.x * 0.5 + 0.5) * canvas.clientWidth, (-tmpV.y * 0.5 + 0.5) * canvas.clientHeight, tmpV.z < 1];
  }

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = (now - t0) / 1000;
    const fast = snap() ? 1 : 1 - Math.exp(-dt * 3.2);
    for (const k of LEVELS) lv[k] += (goal[k] - lv[k]) * fast;

    camPos.lerp(new THREE.Vector3(...shot.pos), snap() ? 1 : 1 - Math.exp(-dt * 1.6));
    camAt.lerp(new THREE.Vector3(...shot.at), snap() ? 1 : 1 - Math.exp(-dt * 1.6));
    shift += (shot.shift - shift) * (snap() ? 1 : 1 - Math.exp(-dt * 1.6));
    camera.position.copy(camPos);
    camera.lookAt(camAt);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.setViewOffset(w, h, -shift * w, 0, w, h);

    // points: scatter in, dim, or sweep in
    if (goal.sweep) sweepT = still ? 0.62 : Math.min(1.25, sweepT + dt / 3.2);
    const sweepZ = 0.1 + sweepT * 1.0;
    const reveal = lv.reveal, dim = 1 - 0.72 * lv.dim;
    for (let i = 0; i < n; i++) {
      const j = i * 3;
      let k = reveal;
      if (lv.sweep > 0.01) k = Math.max(k, home[j + 2] < sweepZ ? 1 : 0);
      const e = k * k * (3 - 2 * k);
      pos[j] = scatter[j] + (home[j] - scatter[j]) * e;
      pos[j + 1] = scatter[j + 1] + (home[j + 1] - scatter[j + 1]) * e;
      pos[j + 2] = scatter[j + 2] + (home[j + 2] - scatter[j + 2]) * e;
      const b = (0.15 + 0.85 * e) * dim * (i >= nTable && lv.box > 0.5 ? 1.15 : 1);
      col[j] = base[j] * b; col[j + 1] = base[j + 1] * b; col[j + 2] = base[j + 2] * b;
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    grid.material.opacity = 0.5 * Math.max(reveal, lv.sweep) * dim;
    sweep.material.opacity = lv.sweep * (sweepT < 1.05 ? 0.12 : 0);
    sweep.position.set(0, 0.2, sweepZ);

    frustum.material.opacity = 0.9 * lv.frustum;
    body.material.opacity = 0.35 * lv.person;
    box.material.opacity = lv.box;

    // hand
    let say = "", grabbed = lv.grabbed > 0.5;
    if (lv.wander > 0.5) handPos.lerp(wanderHand(now / 1000), 0.08);
    else if (lv.still > 0.5 || reduced.matches) handPos.lerp(lv.grabbed > 0.5 ? GRAB_AT : START, still ? 1 : 0.08);
    else if (lv.guide > 0.5) { const g = guidedHand(still ? 2.3 : t); handPos.copy(g.p); say = g.say; grabbed = grabbed || g.grabbed; }
    hand.position.copy(handPos);
    hand.material.opacity = lv.hand;
    shadow.position.set(handPos.x, 0.0015, handPos.z);
    shadow.material.opacity = 0.5 * lv.hand;
    drop.geometry.setFromPoints([handPos, new THREE.Vector3(handPos.x, 0, handPos.z)]);
    drop.computeLineDistances();
    drop.material.opacity = 0.7 * lv.hand;

    // leg from fingertip to the mug, yellow; green on grab
    const to = lv.box > 0.5 || grabbed ? GRAB_AT : MUG_TOP.clone().add(new THREE.Vector3(0, -0.04, -0.04));
    const d = new THREE.Vector3().subVectors(to, handPos);
    const len = Math.max(0.001, d.length() - 0.03);
    const shown = len * lv.leg;
    const dir = d.clone().normalize();
    leg.scale.set(1, Math.max(0.0001, shown), 1);
    leg.position.copy(handPos).addScaledVector(dir, shown / 2);
    leg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    tip.position.copy(handPos).addScaledVector(dir, shown + 0.012);
    tip.quaternion.copy(leg.quaternion);
    legMat.color.copy(grabbed ? C.grab : C.leg);
    legMat.opacity = lv.leg;
    leg.visible = tip.visible = lv.leg > 0.02 && d.length() > 0.035;

    // components of the leg: across, then down, then forward
    const a0 = handPos.clone(), a1 = a0.clone(), a2 = a0.clone();
    a1.x = to.x; a2.x = to.x; a2.y = to.y;
    axes.geometry.setFromPoints([a0, a1, a2, to]);
    axes.computeLineDistances();
    axisMat.opacity = lv.axes;

    linkMat.opacity = 0.9 * lv.link * (1 - lv.cut);
    cloud.material.opacity = 0.7 * lv.link * (1 - 0.6 * lv.cut);
    link.computeLineDistances();

    renderer.render(scene, camera);

    // labels that follow 3D points (HTML, so they use the page's type)
    const L = labels;
    const place = (el, v, on, dx = 0, dy = 0) => {
      if (!el) return;
      const [x, y, inFront] = project(v);
      el.style.opacity = on && inFront ? on : 0;
      el.style.transform = `translate(${x + dx}px, ${y + dy}px)`;
    };
    if (say && say !== word) { word = say; wordT = now; }
    if (L.word) {
      L.word.textContent = say || word;
      L.word.classList.toggle("is-grab", (say || word) === "Grab");
    }
    place(L.word, handPos, lv.words * (say ? 1 : 0), 18, -46);
    place(L.mug, MUG_TOP, Math.max(lv.box, lv.words * 0.9, lv.axes), -18, -34);
    place(L.you, CHEST, lv.frustum, -14, -30);
    place(L.cloud, CLOUD, lv.link * (1 - lv.cut * 0.4), -30, -40);
    if (L.cloud) L.cloud.classList.toggle("is-cut", lv.cut > 0.5);
    const mid = (p, q) => p.clone().add(q).multiplyScalar(0.5);
    place(L.ax, mid(a0, a1), lv.axes, -20, -28);
    place(L.ay, mid(a1, a2), lv.axes, 10, -10);
    place(L.az, mid(a2, to), lv.axes, 10, -28);
    if (lv.axes > 0.01) {
      const f = (v) => `${Math.round(Math.abs(v) * 100)} cm`;
      L.ax.textContent = `Right ${f(to.x - a0.x)}`;
      L.ay.textContent = `${to.y < a0.y ? "Down" : "Up"} ${f(to.y - a0.y)}`;
      L.az.textContent = `Forward ${f(to.z - a0.z)}`;
    }
  }
  requestAnimationFrame(frame);

  return { setShot };
}
