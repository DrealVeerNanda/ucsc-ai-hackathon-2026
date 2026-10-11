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
  // title: the whole idea, a guided reach
  title:   { pos: [-0.38, 0.75, -0.35], at: [0.05, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1 } },
  // problem: the world as a blind wearer gets it, a fog of points nobody can read
  fog:     { pos: [-0.2, 0.45, -0.15], at: [0.0, 0.1, 0.6], shift: 0.22, lv: { reveal: 0.18, dim: 0.5 } },
  query:   { pos: [-0.22, 0.4, 0.1], at: [-0.06, 0.05, 0.6], shift: 0.22, lv: { reveal: 0.8, dim: 0.4, q: 1 } },
  phone:   { pos: [-0.55, 0.7, -0.7], at: [0.0, 0.22, 0.25], shift: 0.22, lv: { reveal: 1, dim: 0.55, frustum: 1 } },
  // the ask: low and close, the hand moving, a speech bubble with each word
  ask:     { pos: [-0.36, 0.45, -0.1], at: [-0.02, 0.08, 0.5], shift: 0.0, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, words: 1 } },
  // competitors: the scene sinks far back behind the list
  recede:  { pos: [-0.9, 1.6, -1.5], at: [0.0, 0.0, 0.5], shift: 0.25, lv: { reveal: 1, dim: 0.85 } },
  // how it works, one look per word
  sweep:   { pos: [-0.43, 0.85, -0.55], at: [0.0, 0.1, 0.45], shift: 0.16, lv: { sweep: 1, frustum: 1 } },
  found:   { pos: [-0.36, 0.6, -0.3], at: [0.05, 0.08, 0.45], shift: 0.18, lv: { reveal: 1, box: 1, hand: 1, still: 1, tip: 1 } },
  vector:  { speed: 2.6, pos: [-0.5, 0.55, 0.05], at: [0.0, 0.08, 0.48], shift: 0.2, lv: { reveal: 1, box: 1, hand: 1, still: 1, leg: 1, axes: 1 } },
  speak:   { pos: [-0.3, 0.35, 0.05], at: [-0.05, 0.15, 0.4], shift: 0.2, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, words: 1, big: 1 } },
  // core loop: straight overhead, a coordinate gizmo at the fingertip
  plan:    { pos: [-0.02, 1.15, 0.35], at: [0.0, 0.0, 0.5], shift: 0.22, lv: { reveal: 1, box: 1, hand: 1, still: 1, leg: 1, gizmo: 1 } },
  buzz:    { pos: [0.02, 0.5, -0.25], at: [0.2, 0.18, 0.32], shift: 0.24, lv: { reveal: 1, dim: 0.6, hand: 1, still: 1, motors: 1 } },
  overload:{ pos: [0.02, 0.5, -0.25], at: [0.2, 0.18, 0.32], shift: 0.24, lv: { reveal: 1, dim: 0.6, hand: 1, still: 1, motors: 1, all: 1 } },
  finale:  { pos: [-0.5, 0.6, -0.45], at: [-0.05, 0.05, 0.55], shift: 0.2, lv: { reveal: 1, hand: 1, leg: 1, still: 1, grabbed: 1 } },
  off:     { pos: [-0.38, 0.75, -0.35], at: [0.05, 0.05, 0.55], shift: 0.0, lv: { reveal: 1, dim: 1 } },
  // results: overhead on the mug, where reaches ended
  camonly: { pos: [-0.06, 0.55, 0.4], at: [-0.08, 0.0, 0.63], shift: 0.3, lv: { reveal: 1, dim: 0.85, land2: 1, ring: 1 } },
  both:    { pos: [-0.06, 0.55, 0.4], at: [-0.08, 0.0, 0.63], shift: 0.3, lv: { reveal: 1, dim: 0.85, land1: 1, land2: 1, ring: 1 } },
  // why the GX10, one look per reason
  dome:    { pos: [-1.2, 1.1, -1.4], at: [0.0, 0.2, 0.5], shift: 0.2, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, dome: 1, gx: 1 } },
  boxed:   { pos: [-0.8, 0.5, -0.38], at: [-0.32, 0.03, 0.22], shift: 0.2, lv: { reveal: 1, dim: 0.6, gx: 1 } },
  quick:   { pos: [-0.3, 0.55, -0.2], at: [0.05, 0.05, 0.55], shift: 0.16, lv: { reveal: 1, hand: 1, leg: 1, guide: 1, words: 1, fast: 1 } },
  cut:     { pos: [-0.95, 0.8, -0.7], at: [-0.4, 0.16, 0.5], shift: 0.18, lv: { reveal: 1, gx: 1, link: 1, cut: 1, hand: 1, leg: 1, guide: 1 } },
};
const LEVELS = ["reveal", "dim", "hand", "leg", "guide", "wander", "words", "frustum", "sweep", "box",
  "still", "axes", "link", "cut", "grabbed", "big", "gizmo", "land1", "land2", "ring", "dome", "gx", "fast", "q", "motors", "all", "tip"];

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

  const hand = new THREE.Mesh(new THREE.SphereGeometry(0.007, 16, 12), new THREE.MeshBasicMaterial({ color: C.text, transparent: true, side: DS }));
  // the reaching hand as lidar would see it: points on a palm-down right hand, index finger out,
  // the fingertip (the point we track) at the origin. x = wearer's right, z = forward.
  const handPts = [];
  const tube = (x0, y0, z0, x1, y1, z1, r, n) => {
    for (let i = 0; i < n; i++) {
      const k = Math.random(), a = Math.random() * Math.PI * 2;
      handPts.push(x0 + (x1 - x0) * k + r * Math.cos(a), y0 + (y1 - y0) * k + r * Math.sin(a), z0 + (z1 - z0) * k);
    }
  };
  tube(0, 0, -0.004, 0, 0.004, -0.075, 0.0075, 170);            // index
  tube(0.019, -0.006, -0.04, 0.02, 0.004, -0.095, 0.0078, 130);  // middle, curled under
  tube(0.036, -0.008, -0.05, 0.037, 0.003, -0.1, 0.0074, 120);   // ring
  tube(0.051, -0.008, -0.065, 0.052, 0.0, -0.1, 0.0065, 100);    // little
  tube(-0.024, -0.008, -0.06, -0.016, -0.002, -0.105, 0.0085, 110); // thumb
  for (let i = 0; i < 420; i++) {                                   // palm, a flat slab
    handPts.push(-0.012 + Math.random() * 0.072, (Math.random() < 0.5 ? -1 : 1) * 0.011 + (Math.random() - 0.5) * 0.004, -0.1 - Math.random() * 0.085);
  }
  tube(0.02, 0, -0.185, 0.02, 0.005, -0.25, 0.028, 220);           // wrist
  const handCloud = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(handPts, 3)),
    new THREE.PointsMaterial({ size: 0.0045, color: C.text, transparent: true, opacity: 0 }));
  scene.add(handCloud);
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

  // coordinate gizmo at the fingertip (core loop slide)
  const gizmo = new THREE.Group();
  for (const [dir, c] of [[[1, 0, 0], C.text], [[0, 1, 0], C.text], [[0, 0, 1], C.text]]) {
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(...dir), new THREE.Vector3(), 0.11, c, 0.025, 0.014);
    arrow.line.material.transparent = true;
    arrow.cone.material.transparent = true;
    arrow.cone.material.side = DS;
    gizmo.add(arrow);
  }
  scene.add(gizmo);

  // where reaches ended (results slide): an illustration drawn from the simulation's medians,
  // lidar about 3 cm out, camera only about 6 cm with one in ten far off
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const rayleigh = (sigma) => sigma * Math.sqrt(-2 * Math.log(1 - rand() * 0.999));
  function landings(sigma, wide) {
    const pts = [];
    for (let i = 0; i < 100; i++) {
      const r = wide && rand() < 0.1 ? 0.15 + rand() * 0.12 : rayleigh(sigma);
      const a = rand() * Math.PI * 2;
      pts.push(MUG.x + r * Math.cos(a), 0.003, MUG.z + r * Math.sin(a));
    }
    return new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)),
      new THREE.PointsMaterial({ size: 0.011, transparent: true, opacity: 0 }));
  }
  const land1 = landings(0.0263, false);
  land1.material.color.copy(C.grab);
  const land2 = landings(0.045, true);
  land2.material.color.copy(C.muted);
  scene.add(land1, land2);
  const ringPts = [];
  for (let i = 0; i <= 64; i++) { const a = (i / 64) * Math.PI * 2; ringPts.push(new THREE.Vector3(MUG.x + 0.06 * Math.cos(a), 0.004, MUG.z + 0.06 * Math.sin(a))); }
  const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts), new THREE.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0 }));
  scene.add(ring);

  // privacy dome over the whole scene (why the GX10)
  const dome = new THREE.Mesh(new THREE.SphereGeometry(0.85, 28, 10, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: C.near, wireframe: true, transparent: true, opacity: 0, side: DS }));
  dome.position.set(0.05, 0, 0.45);
  scene.add(dome);

  // the GX10 on the table, and its internet cable that gets cut
  const GX10 = new THREE.Vector3(0.36, 0.026, 0.2);
  const gx = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.15, 0.052, 0.15)),
    new THREE.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0 }));
  gx.position.copy(GX10);
  scene.add(gx);
  const NET = new THREE.Vector3(0.62, 0.42, 0.95);
  const cableMat = new THREE.LineDashedMaterial({ color: C.text, dashSize: 0.02, gapSize: 0.012, transparent: true, opacity: 0 });
  const cableA = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), cableMat);
  const cableB = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), cableMat);
  scene.add(cableA, cableB);

  // lidar sweep plane
  const sweep = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.55, 0.003, 0), new THREE.Vector3(0.55, 0.003, 0)]),
    new THREE.LineBasicMaterial({ color: C.near, transparent: true, opacity: 0 }));
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
  let t0 = performance.now(), last = t0, sweepT = 0, word = "", wordT = 0, clear = 0, settle = 1, guideT = 0;
  const mvp = new THREE.Matrix4();
  const handPos = START.clone();

  function setShot(name) {
    shot = SHOTS[name] || SHOTS.title;
    for (const k of LEVELS) goal[k] = shot.lv[k] || 0;
    if (shot.lv.sweep) sweepT = 0;
    t0 = performance.now();
    guideT = 0;
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
    const cycle = 9, u = (t % cycle) / cycle;
    const err = new THREE.Vector3().subVectors(GRAB_AT, START);
    const legs = [[0, 0.0, 0.26], [1, 0.26, 0.46], [2, 0.46, 0.74]]; // x, then y, then z
    const p = START.clone();
    let say = "";
    for (const [axis, a, b] of legs) {
      const k = Math.min(1, Math.max(0, (u - a) / (b - a)));
      const e = k * k * (3 - 2 * k);
      p.setComponent(axis, START.getComponent(axis) + err.getComponent(axis) * e);
      if (u >= a && u < b) say = ["Right", err.y < 0 ? "Down" : "Up", "Forward"][axis];
    }
    if (u >= 0.74) say = "Grab";
    return { p, say, grabbed: u >= 0.74 };
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
    const pace = shot.speed || 1;
    const fast = snap() ? 1 : 1 - Math.exp(-dt * 3.2 * pace);
    if (settle > 0.9) guideT += dt * (lv.fast > 0.5 ? 1.8 : 1);
    for (const k of LEVELS) lv[k] += (goal[k] - lv[k]) * fast;

    const before = camPos.clone();
    camPos.lerp(new THREE.Vector3(...shot.pos), snap() ? 1 : 1 - Math.exp(-dt * 1.6 * pace));
    const speed = before.distanceTo(camPos) / Math.max(dt, 1e-3);
    settle += ((speed < 0.06 ? 1 : 0) - settle) * (snap() ? 1 : 1 - Math.exp(-dt * 8));
    camAt.lerp(new THREE.Vector3(...shot.at), snap() ? 1 : 1 - Math.exp(-dt * 1.6 * pace));
    shift += (shot.shift - shift) * (snap() ? 1 : 1 - Math.exp(-dt * 1.6 * pace));
    camera.position.copy(camPos);
    camera.lookAt(camAt);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.setViewOffset(w, h, -shift * w, 0, w, h);
    camera.updateMatrixWorld();
    scene.updateMatrixWorld();
    mvp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(scene.matrixWorld);
    const M = mvp.elements;
    // slides with text on the left: points fade out across the text column so type stays legible
    clear += ((shot.shift > 0.05 ? 1 : 0) - clear) * (snap() ? 1 : 1 - Math.exp(-dt * 3));

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
      let b = (0.15 + 0.85 * e) * dim * (i >= nTable && lv.box > 0.5 ? 1.15 : 1);
      if (clear > 0.01) {
        const px = pos[j], py = pos[j + 1], pz = pos[j + 2];
        const sx = (M[0] * px + M[4] * py + M[8] * pz + M[12]) / (M[3] * px + M[7] * py + M[11] * pz + M[15]);
        const f = Math.min(1, Math.max(0, ((sx + 1) / 2 - 0.4) / 0.22));
        b *= 1 - clear * (1 - (0.04 + 0.96 * f * f * (3 - 2 * f)));
      }
      col[j] = base[j] * b; col[j + 1] = base[j + 1] * b; col[j + 2] = base[j + 2] * b;
    }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    grid.material.opacity = 0.5 * Math.max(reveal, lv.sweep) * dim;
    sweep.material.opacity = lv.sweep * (sweepT < 1.05 ? 1 : 0);
    sweep.position.set(0, 0, sweepZ);

    frustum.material.opacity = 0.9 * lv.frustum;
    body.material.opacity = 0.35 * lv.person;
    box.material.opacity = lv.box;

    // hand
    let say = "", grabbed = lv.grabbed > 0.5;
    if (lv.wander > 0.5) handPos.lerp(wanderHand(now / 1000), 0.08);
    else if (lv.still > 0.5 || reduced.matches) handPos.lerp(lv.grabbed > 0.5 ? GRAB_AT : START, still ? 1 : 0.08);
    else if (lv.guide > 0.5) { const g = guidedHand(still ? 2.3 : guideT); handPos.copy(g.p); say = g.say; grabbed = grabbed || g.grabbed; }
    hand.position.copy(handPos);
    hand.material.opacity = lv.hand;
    handCloud.position.copy(handPos);
    handCloud.material.opacity = 0.9 * lv.hand;
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

    gizmo.position.copy(handPos);
    gizmo.children.forEach((ar) => { ar.line.material.opacity = lv.gizmo; ar.cone.material.opacity = lv.gizmo; });
    gizmo.visible = lv.gizmo > 0.02;
    land1.material.opacity = lv.land1;
    land2.material.opacity = lv.land2;
    ring.material.opacity = 0.8 * lv.ring;
    dome.material.opacity = 0.28 * lv.dome;
    dome.rotation.y = still ? 0 : now / 9000;
    gx.material.opacity = lv.gx;
    // the cable runs from the box toward the internet; on the cut it parts in the middle
    const top = GX10.clone().add(new THREE.Vector3(0, 0.026, 0));
    const gap = 0.5 * lv.cut;
    const m1 = top.clone().lerp(NET, Math.max(0.05, 0.5 - gap * 0.6));
    const m2 = top.clone().lerp(NET, Math.min(0.95, 0.5 + gap * 0.6));
    cableA.geometry.setFromPoints([top, m1]);
    cableB.geometry.setFromPoints([m2, NET]);
    cableA.computeLineDistances();
    cableB.computeLineDistances();
    cableMat.opacity = 0.95 * lv.link;

    renderer.render(scene, camera);

    // labels that follow 3D points (HTML, so they use the page's type)
    const L = labels;
    const place = (el, v, on, dx = 0, dy = 0) => {
      if (!el) return;
      const [x, y, inFront] = project(v);
      el.style.opacity = on && inFront && el.textContent.trim() ? on * settle : 0;
      // left/top, not transform: the pop and buzz effects scale the label in place
      el.style.left = `${x + dx}px`;
      el.style.top = `${y + dy}px`;
    };
    if (say && say !== word) { word = say; wordT = now; }
    if (L.word) {
      if (L.word.textContent !== (say || word)) {
        L.word.textContent = say || word;
        L.word.classList.remove("pop");
        void L.word.offsetWidth;
        L.word.classList.add("pop");
      }
      L.word.classList.toggle("is-grab", (say || word) === "Grab");
      L.word.classList.toggle("is-big", lv.big > 0.5);
    }
    place(L.word, handPos, lv.words * (say ? 1 : 0), lv.big > 0.5 ? -16 : 26, lv.big > 0.5 ? -150 : -100);
    place(L.mug, MUG_TOP, Math.max(lv.box, lv.words * 0.9, lv.axes), -18, -34);
    place(L.you, CHEST, lv.frustum, -14, -30);
    place(L.q, MUG_TOP, lv.q, -10, -110);
    place(L.tip, handPos, lv.tip, -40, 26);
    // four vibration motors around the wrist, buzzing one at a time (why not vibration)
    (L.motors || []).forEach((el, k) => {
      const [dx, dy] = [[-18, -96], [-18, 60], [60, -18], [-96, -18]][k];
      place(el, handPos, lv.motors, dx, dy);
      // one buzz at a time, round the wrist; on the overload build, all four at once
      el.classList.toggle("is-on", lv.all > 0.5 ? true : still ? k === 0 : Math.floor(now / 280) % 4 === k);
    });
    place(L.cloud, NET, lv.link, -40, -40);
    if (L.cloud) L.cloud.classList.toggle("is-cut", lv.cut > 0.5);
    place(L.gx10, GX10.clone().add(new THREE.Vector3(0, 0.06, 0)), lv.gx, -26, -34);
    place(L.gxx, handPos.clone().add(new THREE.Vector3(0.12, 0, 0)), lv.gizmo, 8, -10);
    place(L.gxy, handPos.clone().add(new THREE.Vector3(0, 0.12, 0)), lv.gizmo, 8, -20);
    place(L.gxz, handPos.clone().add(new THREE.Vector3(0, 0, 0.12)), lv.gizmo, 8, -10);
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
