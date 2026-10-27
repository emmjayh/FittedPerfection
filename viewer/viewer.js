// FittedPerfection review viewer. Picks an outfit, a fit version (optionally a second one to compare
// side by side) and one or all four bodies; shows the body as skin, X-ray or hidden; and
// offers the four review cameras (front, 3/4, side, back), with the source renders alongside.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

const GAP = 0.9;                       // metres between bodies when several are shown
const SKIN_TONE = 0xd2a083;
const CAMERAS = {                      // glTF axes: +Y up, bodies face +Z, their left is +X
  front: new THREE.Vector3(0, 0, 1),
  'three-quarter': new THREE.Vector3(1, 0, 1).normalize(),
  side: new THREE.Vector3(1, 0, 0),
  back: new THREE.Vector3(0, 0, -1),
};

const el = (id) => document.getElementById(id);
const stage = el('stage');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setScissorTest(true);
stage.prepend(renderer.domElement);

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.addEventListener('change', draw);

const materials = {
  skin: new THREE.MeshStandardMaterial({ color: SKIN_TONE, roughness: 0.85 }),
  xray: new THREE.MeshBasicMaterial({ color: 0xff6a4d, transparent: true, opacity: 0.3, depthTest: false, depthWrite: false }),
  plain: new THREE.MeshStandardMaterial({ color: 0xa3adb8, roughness: 0.9, side: THREE.DoubleSide }),
};

const view = { body: 'skin', outfit: 'texture', camera: 'front', aimed: false };
let catalog = { outfits: [], bodies: {}, body_order: [] };
let request = 0;

// ---- scenes ---------------------------------------------------------------

function makePane() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1f25);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3f4a, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(2, 4, 3);
  const rim = new THREE.DirectionalLight(0xffffff, 0.5);
  rim.position.set(-3, 2, -2);
  scene.add(key, rim);
  const content = new THREE.Group();
  scene.add(content);
  return { scene, content };
}

const panes = { a: makePane(), b: makePane() };

// ---- loading --------------------------------------------------------------

const loader = new GLTFLoader();
const loaded = new Map();      // url -> Promise<THREE.Object3D>, loaded once and cloned per use

function instance(url) {
  if (!loaded.has(url)) loaded.set(url, loader.loadAsync(url).then((gltf) => gltf.scene));
  return loaded.get(url).then((original) => {
    const copy = SkeletonUtils.clone(original);
    copy.traverse((node) => {
      if (node.isMesh) {
        node.frustumCulled = false;          // skinned bounds are not reliable
        node.userData.ownMaterial = node.material;
      }
    });
    return copy;
  });
}

function applyLooks() {
  for (const pane of Object.values(panes)) {
    pane.content.traverse((node) => {
      if (node.userData.role === 'body') {
        node.visible = view.body !== 'off';
        node.traverse((mesh) => {
          if (!mesh.isMesh) return;
          mesh.material = view.body === 'xray' ? materials.xray : materials.skin;
          mesh.renderOrder = view.body === 'xray' ? 10 : 0;
        });
      } else if (node.userData.role === 'outfit') {
        node.visible = view.outfit !== 'off';
        node.traverse((mesh) => {
          if (mesh.isMesh) mesh.material = view.outfit === 'plain' ? materials.plain : mesh.userData.ownMaterial;
        });
      }
    });
  }
  draw();
}

// ---- selection ------------------------------------------------------------

function current() {
  const outfit = catalog.outfits[el('pick-outfit').value] || null;
  const version = (id) => outfit?.versions.find((v) => v.id === id) || null;
  const choice = el('pick-body').value;
  const bodies = choice === 'all' ? catalog.body_order.filter((b) => catalog.bodies[b]) : [choice];
  return { outfit, a: version(el('pick-version').value), b: version(el('pick-compare').value), bodies };
}

async function rebuild() {
  const mine = ++request;
  const { outfit, a, b, bodies } = current();
  const wanted = b ? { a, b } : { a };
  say('Loading…');
  try {
    const built = {};
    for (const [key, version] of Object.entries(wanted)) {
      built[key] = await Promise.all(bodies.map(async (body, i) => {
        const group = new THREE.Group();
        group.position.x = (i - (bodies.length - 1) / 2) * GAP;
        // the deliverable when the version has it: the full-resolution outfit on the shown body
        const bodyUrl = version?.shown_bodies?.[body] || catalog.bodies[body]?.url;
        const draftUrl = version?.outfits?.[body] || version?.drafts[body];
        const [bodyModel, draftModel] = await Promise.all([
          bodyUrl ? instance(bodyUrl) : null,
          draftUrl ? instance(draftUrl) : null,
        ]);
        if (bodyModel) { bodyModel.userData.role = 'body'; group.add(bodyModel); }
        if (draftModel) { draftModel.userData.role = 'outfit'; group.add(draftModel); }
        return group;
      }));
    }
    if (mine !== request) return;        // a newer selection superseded this one
    for (const key of ['a', 'b']) {
      panes[key].content.clear();
      (built[key] || []).forEach((group) => panes[key].content.add(group));
    }
    const names = bodies.map((body) => `${body.replace(/^body_\d+_/, '').replace('_', ' ')} (${catalog.bodies[body]?.sex[0] ?? '?'})`);
    el('label-a').textContent = `${a ? a.id : 'bodies only'} · ${names.join(' · ')}`;
    el('label-b').textContent = b ? b.id : '';
    showSource(outfit);
    showReport(a, bodies[0]);
    applyLooks();
    aim(view.aimed);
    say(describe(bodies, a));
  } catch (error) {
    say(`Could not load: ${error.message}`);
    console.error(error);
  }
}

function describe(bodies, version) {
  if (!bodies.length) return 'No bodies yet (they are made in build step 2).';
  return `${bodies.length} bod${bodies.length === 1 ? 'y' : 'ies'}${version ? '' : ', no fit selected'}`;
}

// ---- cameras and drawing --------------------------------------------------

function panesShown() {
  const { width, height } = stage.getBoundingClientRect();
  if (!current().b) return [{ pane: panes.a, x: 0, width, height }];
  const half = Math.floor(width / 2);
  return [{ pane: panes.a, x: 0, width: half, height }, { pane: panes.b, x: half, width: width - half, height }];
}

function aim(keepDirection) {
  const box = new THREE.Box3();
  for (const { pane } of panesShown()) box.expandByObject(pane.content);
  if (box.isEmpty()) box.set(new THREE.Vector3(-0.5, 0, -0.3), new THREE.Vector3(0.5, 1.8, 0.3));
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const reach = size.length() + 1;
  const direction = keepDirection
    ? camera.position.clone().sub(controls.target).normalize()
    : CAMERAS[view.camera].clone();
  controls.target.copy(centre);
  camera.position.copy(centre).addScaledVector(direction, reach * 2);
  camera.up.set(0, 1, 0);
  camera.far = reach * 4;
  camera.zoom = 1;
  const { width, height } = panesShown()[0];
  const aspect = width / Math.max(height, 1);
  camera.userData.half = Math.max(size.y, size.x / aspect, size.z / aspect) * 0.54;
  view.aimed = true;
  resize();
}

function resize() {
  const { width, height } = stage.getBoundingClientRect();
  renderer.setSize(width, height);
  const first = panesShown()[0];
  const aspect = first.width / Math.max(first.height, 1);
  const half = camera.userData.half || 1;
  camera.left = -half * aspect;
  camera.right = half * aspect;
  camera.top = half;
  camera.bottom = -half;
  camera.updateProjectionMatrix();
  controls.update();
  draw();
}

function draw() {
  for (const { pane, x, width, height } of panesShown()) {
    renderer.setViewport(x, 0, width, height);
    renderer.setScissor(x, 0, width, height);
    renderer.render(pane.scene, camera);
  }
}

// ---- side panel -----------------------------------------------------------

function showSource(outfit) {
  const views = Object.entries(outfit?.source || {});
  el('source').innerHTML = views.length
    ? views.map(([name, url]) => `<figure><img src="${url}" alt="${name} view"><figcaption>${name}</figcaption></figure>`).join('')
    : '<p class="empty">No source renders yet.</p>';
}

async function showReport(version, body) {
  const url = version?.reports[body];
  if (!url) {
    el('report').textContent = version ? 'No fit.report.json for this body.' : '–';
    return;
  }
  try {
    const report = await (await fetch(url)).json();
    const failed = Object.entries(report.gates || {}).filter(([, passed]) => !passed).map(([gate]) => gate);
    el('report').textContent = JSON.stringify({
      body, status: report.status, fit_reviewed: report.fit_reviewed, game_ready: report.game_ready,
      root_offset_m: report.root_offset_m, failed_gates: failed,
      stretch_cage: report.stretch_vs_source, stretch_full: report.full_resolution?.stretch_vs_source,
    }, null, 1);
  } catch (error) {
    el('report').textContent = `Could not read the report: ${error.message}`;
  }
  showReview(version, body);
}

async function showReview(version, body) {
  // spec §5.7: the audit and Muse Wonder's findings next to the fit; a checklist, never a verdict
  const parts = [];
  try {
    if (version?.audits?.[body]) {
      const audit = await (await fetch(version.audits[body])).json();
      parts.push(JSON.stringify({
        inside_body_rest: audit.rest?.inside_share,
        inside_body_poses: Object.fromEntries(Object.entries(audit.poses || {}).map(([k, v]) => [k, v.inside_share])),
        lost_coverage: Object.fromEntries(Object.entries(audit.coverage || {}).map(([k, v]) => [k, v.lost_share])),
        silhouette: audit.silhouette?.mean_width_change,
      }, null, 1));
    }
    if (version?.wonder) {
      const wonder = await (await fetch(version.wonder)).json();
      parts.push(`Muse Wonder: ${wonder.status}\n` + JSON.stringify(wonder.findings ?? {}, null, 1));
    }
  } catch (error) {
    parts.push(`Could not read the review: ${error.message}`);
  }
  el('review').textContent = parts.length ? parts.join('\n\n') : 'No review yet (run the review stage).';
}

function say(text) { el('status').textContent = text; }

// ---- controls -------------------------------------------------------------

function options(select, entries, placeholder) {
  select.replaceChildren(...[placeholder, ...entries].filter(Boolean).map(([value, text]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    return option;
  }));
}

function fillVersions() {
  const versions = catalog.outfits[el('pick-outfit').value]?.versions || [];
  const entries = versions.map((v) => [v.id, `${v.id}  (${new Date(v.time * 1000).toLocaleString()})`]);
  options(el('pick-version'), entries, ['', versions.length ? '– none –' : 'no fits yet']);
  options(el('pick-compare'), entries, ['', '– none –']);
  if (versions.length) el('pick-version').value = versions[0].id;
}

async function start() {
  catalog = await (await fetch('/api/catalog')).json();
  options(el('pick-outfit'), catalog.outfits.map((o, i) => [i, o.runs.join(' + ')]),
    catalog.outfits.length ? null : ['', 'no runs yet']);
  options(el('pick-body'), catalog.body_order.filter((b) => catalog.bodies[b])
    .map((b) => [b, `${b} (${catalog.bodies[b].sex})`]), ['all', 'All four']);
  fillVersions();

  el('pick-outfit').addEventListener('change', () => { fillVersions(); rebuild(); });
  for (const id of ['pick-version', 'pick-compare', 'pick-body']) el(id).addEventListener('change', rebuild);
  for (const group of document.querySelectorAll('.toggle')) {
    group.addEventListener('click', (event) => {
      const button = event.target.closest('button');
      if (!button) return;
      group.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === button));
      view[group.dataset.group] = button.dataset.value;
      applyLooks();
    });
  }
  document.querySelectorAll('[data-camera]').forEach((button) => button.addEventListener('click', () => {
    view.camera = button.dataset.camera;
    aim(false);
  }));
  window.addEventListener('resize', resize);
  rebuild();
}

start().catch((error) => say(`Could not start: ${error.message}`));
