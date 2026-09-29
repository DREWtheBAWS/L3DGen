import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const $ = (id) => document.getElementById(id);
const CLIENT_ID = 'comfy3d-' + Math.random().toString(36).slice(2, 10);
const VIEWS = ['front', 'left', 'back', 'right'];

const state = {
  files: {},        // view -> uploaded filename on the ComfyUI side
  running: false,
  promptId: null,
  saveNode: null,
  titles: {},
  cached: new Set(),
  lastGlb: null,      // the rigged output of the last run, if there was one
  lastPlainGlb: null, // the unrigged one: what Mixamo wants, and what holds the atlas
  shownGlb: null,     // the ComfyUI output currently in the viewer, for FBX export
  fbxUrl: null,       // set when the server already wrote an FBX (Mixamo round-trip)
  zipUrl: null,       // fbx + .fbm texture folder, the reliable Unity import
  fbxName: null,
  jobId: null,        // set while a multi-stage run is in flight
};

// ------------------------------------------------------------------ log ----
function log(msg, cls) {
  const el = $('log');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
function stage(text, cls) {
  const el = $('stage');
  el.textContent = text;
  el.className = 'stage' + (cls ? ' ' + cls : '');
}
function bar(pct) { $('barFill').style.width = Math.max(0, Math.min(100, pct)) + '%'; }

// -------------------------------------------------------------- viewer ----
const renderer = new THREE.WebGLRenderer({ canvas: $('canvas'), antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
camera.position.set(0, 0.4, 2.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotateSpeed = 1.6;

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x20242c, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.7);
key.position.set(2.5, 3.5, 2.2);
scene.add(key);

let current = null;

// Animation playback. `mixer` is rebuilt per model; a null mixer means the
// loaded GLB carries no clips and the animation bar stays hidden.
let mixer = null;
let action = null;
const clock = new THREE.Clock();

function resize() {
  const vp = $('viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($('viewport'));
resize();

(function tick() {
  requestAnimationFrame(tick);
  // getDelta must be read every frame, not only while a clip is playing, or the
  // first frame after pressing play advances by however long the pause lasted.
  const dt = clock.getDelta();
  if (mixer) {
    mixer.update(dt);
    if (action && !action.paused && $('animScrub')) {
      const clip = action.getClip();
      $('animScrub').value = String((action.time / clip.duration) * 1000);
    }
  }
  controls.autoRotate = $('spin').checked;
  controls.update();
  renderer.render(scene, camera);
})();

$('wire').addEventListener('change', (e) => {
  if (!current) return;
  current.traverse((o) => { if (o.isMesh) o.material.wireframe = e.target.checked; });
});

/** Loads a GLB, frames it, and reports its real triangle count. */
function showModel(url) {
  new GLTFLoader().load(url, (gltf) => {
    if (mixer) { mixer.stopAllAction(); mixer.uncacheRoot(mixer.getRoot()); }
    mixer = null;
    action = null;
    if (current) {
      scene.remove(current);
      current.traverse((o) => {
        if (!o.isMesh) return;
        o.geometry.dispose();
        for (const m of [].concat(o.material)) {
          for (const k of ['map', 'normalMap', 'aoMap', 'roughnessMap', 'metalnessMap']) {
            if (m[k]) m[k].dispose();
          }
          m.dispose();
        }
      });
    }
    current = gltf.scene;

    let tris = 0, verts = 0, textured = false;
    current.traverse((o) => {
      if (!o.isMesh) return;
      const g = o.geometry;
      tris += g.index ? g.index.count / 3 : g.attributes.position.count / 3;
      verts += g.attributes.position.count;
      o.material.wireframe = $('wire').checked;
      o.material.side = THREE.DoubleSide;
      if (o.material.map) textured = true;
    });

    // Normalise into a unit-ish box so framing is consistent run to run.
    const box = new THREE.Box3().setFromObject(current);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const scale = 1.6 / Math.max(size.x, size.y, size.z, 1e-6);
    current.position.sub(centre);
    current.scale.setScalar(scale);
    current.position.multiplyScalar(scale);

    scene.add(current);
    controls.target.set(0, 0, 0);
    camera.position.set(0, 0.35, 2.5);
    controls.update();

    // Bones are what a skinned GLB adds; report them, since "did the rig
    // survive the round trip" is the question this viewer is being asked.
    let bones = 0;
    current.traverse((o) => { if (o.isSkinnedMesh && o.skeleton) bones = o.skeleton.bones.length; });

    if (gltf.animations && gltf.animations.length) {
      mixer = new THREE.AnimationMixer(current);
      const sel = $('animClip');
      sel.innerHTML = '';
      gltf.animations.forEach((c, i) => {
        const o = document.createElement('option');
        o.value = String(i);
        o.textContent = c.name + '  (' + c.duration.toFixed(1) + 's)';
        sel.appendChild(o);
      });
      sel.dataset.clips = '1';
      window.__clips = gltf.animations;
      playClip(0);
    }
    $('animBar').hidden = !mixer;

    $('viewportEmpty').hidden = true;
    $('meshStats').textContent =
      Math.round(tris).toLocaleString() + ' tris · ' +
      verts.toLocaleString() + ' verts · ' +
      (textured ? 'textured' : 'untextured') +
      (bones ? ' · ' + bones + ' bones' : '');
    log('loaded ' + Math.round(tris).toLocaleString() + ' triangles'
        + (bones ? ', ' + bones + ' bones' : '')
        + (mixer ? ', ' + gltf.animations.length + ' animation(s)' : ''), 'k');
  }, undefined, (err) => {
    log('viewer could not load the GLB: ' + err.message, 'e');
  });
}

// -------------------------------------------------------------- mixamo ----
/**
 * Hybrid rigging: export the mesh, rig it on Mixamo, bring the texture back.
 *
 * Mixamo strips the material but preserves the UV layout, so reattaching the
 * pipeline's atlas afterwards is a material assignment rather than a re-bake.
 * The unrigged model is the one that goes out — Mixamo wants a bare mesh, and
 * it is also the file carrying the atlas we reattach on the way back.
 */
function mixamoSource() {
  return state.lastPlainGlb || null;
}

function syncMixamo() {
  const src = mixamoSource();
  $('mixamoPanel').hidden = false;
  $('mixamoHint').hidden = false;
  $('mixamoExport').disabled = !src;
  $('mixamoApply').disabled = !src;
  $('mixamoNote').textContent = src
    ? src.filename.split('/').pop()
    : 'Generate a textured model first.';
}

/**
 * Offer the shown model as FBX, which is what Unity wants.
 *
 * Two sources: after a Mixamo round-trip the server already wrote one, so it is
 * a direct download. Otherwise the shown model is a ComfyUI output and Blender
 * converts it on demand — including its skeleton, if it has one.
 */
function syncFbxButton() {
  $('downloadFbx').hidden = !(state.fbxUrl || state.shownGlb);
  // The zip only exists after a Mixamo reattach; it is the FBX beside its .fbm
  // texture folder, which is the form that imports textured without any clicks.
  $('downloadZip').hidden = !state.zipUrl;
}

$('downloadZip').addEventListener('click', () => {
  if (state.zipUrl) download(state.zipUrl, (state.fbxName || 'model') + '.zip');
});

$('downloadFbx').addEventListener('click', async () => {
  if (state.fbxUrl) {
    download(state.fbxUrl, (state.fbxName || 'model') + '.fbx');
    return;
  }
  const src = state.shownGlb;
  if (!src) return;
  const btn = $('downloadFbx');
  btn.disabled = true;
  const was = btn.textContent;
  btn.textContent = 'converting…';
  try {
    const r = await fetch('/api/export-fbx', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: src.filename, subfolder: src.subfolder, type: src.type }),
    });
    const j = await r.json();
    if (j.error) { log(j.error, 'e'); return; }
    (j.warnings || []).forEach((w) => log(w, 'e'));
    log('exported ' + j.name, 'k');
    download(j.url, j.name);
  } catch (e) {
    log('FBX export failed: ' + e.message, 'e');
  } finally {
    btn.disabled = false;
    btn.textContent = was;
  }
});

/** Kick off a browser download for a URL the server produced. */
function download(url, name) {
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

$('mixamoExport').addEventListener('click', async () => {
  const src = mixamoSource();
  if (!src) return;
  const btn = $('mixamoExport');
  btn.disabled = true;
  const was = btn.textContent;
  btn.textContent = 'converting…';
  try {
    const r = await fetch('/api/export-fbx', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: src.filename, subfolder: src.subfolder, type: src.type }),
    });
    const j = await r.json();
    if (j.error) { log(j.error, 'e'); return; }
    (j.warnings || []).forEach((w) => log(w, 'e'));
    log('exported ' + j.name + ' — upload it to mixamo.com, rig it, then download '
        + 'with skin as FBX and load it back with step 3', 'k');
    download(j.url, j.name);
  } catch (e) {
    log('FBX export failed: ' + e.message, 'e');
  } finally {
    btn.disabled = false;
    btn.textContent = was;
  }
});

$('mixamoOpen').addEventListener('click', () => {
  window.open('https://www.mixamo.com/#/?page=1&type=Character', '_blank', 'noopener');
});

$('mixamoApply').addEventListener('click', () => $('mixamoFile').click());

$('mixamoFile').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = '';                       // let the same file be picked twice
  const src = mixamoSource();
  if (!file || !src) return;

  const btn = $('mixamoApply');
  btn.disabled = true;
  const was = btn.textContent;
  btn.textContent = 'reattaching…';
  log('reattaching the atlas to ' + file.name + ' (Blender)', 'k');
  try {
    const q = new URLSearchParams({
      filename: src.filename, subfolder: src.subfolder || '', type: src.type || 'output',
    });
    const r = await fetch('/api/apply-textures?' + q.toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file,
    });
    const j = await r.json();
    if (j.error) { log(j.error, 'e'); return; }
    // Surface the UV check: if Mixamo ever stops preserving the layout, this is
    // the line that says the reattachment is no longer exact.
    (j.report || []).forEach((line) => log(line));
    $('download').href = j.url;
    $('download').download = 'rigged-textured.glb';
    $('download').hidden = false;
    // The FBX was written alongside; it carries the armature and the atlas
    // packed in, which is the form Unity wants.
    state.shownGlb = null;
    state.fbxUrl = j.fbxUrl;
    state.zipUrl = j.zipUrl || null;
    state.fbxName = 'rigged-textured';
    syncFbxButton();
    showModel(j.url);
    log('rigged + textured — use Download .fbx for Unity (armature + embedded texture)', 'k');
  } catch (err) {
    log('reattaching failed: ' + err.message, 'e');
  } finally {
    btn.disabled = false;
    btn.textContent = was;
  }
});

// ----------------------------------------------------------- animation ----
/** Start a clip by index, replacing whatever was playing. */
function playClip(i) {
  if (!mixer || !window.__clips || !window.__clips[i]) return;
  if (action) action.stop();
  action = mixer.clipAction(window.__clips[i]);
  action.reset();
  action.setLoop(THREE.LoopRepeat, Infinity);
  action.play();
  action.paused = false;
  $('animPlay').textContent = 'pause';
}

$('animPlay').addEventListener('click', () => {
  if (!action) return;
  action.paused = !action.paused;
  $('animPlay').textContent = action.paused ? 'play' : 'pause';
});
$('animClip').addEventListener('change', (e) => playClip(Number(e.target.value)));
$('animScrub').addEventListener('input', (e) => {
  if (!action) return;
  action.paused = true;
  $('animPlay').textContent = 'play';
  action.time = (Number(e.target.value) / 1000) * action.getClip().duration;
  mixer.update(0);
});

/** Which output the animation should be applied to: only a rigged GLB will do. */
function riggedOutput() {
  return state.lastGlb && /rigged/i.test(state.lastGlb.filename || '') ? state.lastGlb : null;
}

/** Enable the animation controls only when there is something to animate. */
function syncAnimTarget() {
  const have = !!riggedOutput();
  $('animPanel').hidden = false;
  $('animNote').textContent = have
    ? 'Applies to ' + riggedOutput().filename.split('/').pop()
    : 'Generate with Auto-rig enabled to get a model an animation can drive.';
  const sel = $('animSource');
  $('animApply').disabled = !have || !sel.value;
}

async function loadAnimationList() {
  try {
    const j = await (await fetch('/api/animations')).json();
    const sel = $('animSource');
    sel.innerHTML = '';
    if (!j.blender) {
      sel.innerHTML = '<option value="">Blender not found &mdash; needed to read FBX</option>';
      $('animApply').disabled = true;
      return;
    }
    if (!j.animations.length) {
      sel.innerHTML = '<option value="">No .fbx in animations/</option>';
      $('animApply').disabled = true;
      return;
    }
    for (const a of j.animations) {
      const o = document.createElement('option');
      o.value = a.file; o.textContent = a.label;
      sel.appendChild(o);
    }
    syncAnimTarget();
  } catch (e) { /* panel still works without animation */ }
}

$('animApply').addEventListener('click', async () => {
  const src = $('animSource').value;
  const target = riggedOutput();
  if (!src) return;
  if (!target) {
    log('no rigged model in the last run — enable Auto-rig and generate first', 'e');
    return;
  }
  const btn = $('animApply');
  btn.disabled = true;
  const was = btn.textContent;
  btn.textContent = 'retargeting…';
  log('retargeting ' + src + ' onto ' + target.filename + ' (Blender)', 'k');
  try {
    const r = await fetch('/api/animate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: target.filename, subfolder: target.subfolder, type: target.type, fbx: src,
      }),
    });
    const j = await r.json();
    if (j.error) { log(j.error, 'e'); return; }
    log('retargeted ' + j.matched + '/' + j.total + ' bones, ' + j.frames + ' frames', 'k');
    showModel(j.url);
  } catch (e) {
    log('retarget failed: ' + e.message, 'e');
  } finally {
    btn.disabled = false;
    btn.textContent = was;
  }
});

// ------------------------------------------------------------- controls ----
function bindSlider(id, outId, fmt) {
  const el = $(id), out = $(outId);
  const sync = () => { out.textContent = fmt ? fmt(el.value) : el.value; };
  el.addEventListener('input', sync); sync();
}
const thousands = (v) => Number(v).toLocaleString('en-US').replace(/,/g, ' ');

bindSlider('tris', 'trisOut', thousands);
bindSlider('remeshRes', 'remeshResOut');
bindSlider('crease', 'creaseOut', (v) => v + '°');
bindSlider('steps', 'stepsOut');
bindSlider('cfg', 'cfgOut');
bindSlider('octree', 'octreeOut');
bindSlider('latentRes', 'latentResOut');
bindSlider('facing', 'facingOut', (v) => Number(v).toFixed(1));
bindSlider('viewScale', 'viewScaleOut', (v) => Number(v).toFixed(2));
bindSlider('viewPitch', 'viewPitchOut', (v) => v + '°');
bindSlider('minFacing', 'minFacingOut', (v) => Number(v).toFixed(2));
bindSlider('maskErode', 'maskErodeOut', (v) => v + ' px');
bindSlider('synthDenoise', 'synthDenoiseOut', (v) => Number(v).toFixed(2));
bindSlider('synthSteps', 'synthStepsOut');
bindSlider('synthCfg', 'synthCfgOut', (v) => Number(v).toFixed(1));
bindSlider('gapGrow', 'gapGrowOut', (v) => v + ' px');
bindSlider('synthWeight', 'synthWeightOut', (v) => Number(v).toFixed(2));

/**
 * One control for two genuinely different stages, because having both under the
 * heading "missing views" read as a duplicate.
 *
 *   before - invent the turnaround from your image, then build the mesh from it.
 *            The generated views condition the shape model, so this improves the
 *            geometry as well as the texture.
 *   after  - render the finished mesh at each uncovered angle and repaint it.
 *            Cannot change the shape; only fills what projection missed.
 */
const STRATEGY_NOTE = {
  none: 'Uncovered surfaces are blended from the art you supplied. No image model runs — fastest, but a side you never drew is guessed from colour alone.',
  before: 'Invents a turnaround to condition the shape model. These images are NOT used for texture: the bake needs an exact yaw and an orthographic camera, and a generative model gives neither — it returns a perspective view at an arbitrary angle, and left/right prompts often come back as the same rotation. Experimental.',
  after: 'Renders the finished mesh at each uncovered angle and repaints it. The viewpoint comes from the geometry, so it is orthographic and exactly on-axis — the only path whose output can be projected back accurately. Texture only; the shape is already decided by then.',
  both: 'Invent a turnaround to help the shape model, then repaint uncovered surfaces from renders. Slowest, and inherits the caveats of the "before" stage.',
};
function strategy() { return $('missingViews').value; }
function usesRefGen() { return strategy() === 'before' || strategy() === 'both'; }
function usesSynth() { return strategy() === 'after' || strategy() === 'both'; }

function syncStrategy() {
  const s = strategy();
  $('strategyNote').textContent = STRATEGY_NOTE[s] || '';
  $('genRefOpts').style.display = usesRefGen() ? '' : 'none';
  // The generator settings drive both stages, so show them whenever either runs.
  $('synthOpts').style.display = (s === 'none') ? 'none' : '';
  $('synthNote').style.display = usesSynth() ? '' : 'none';
  updateCoverageNote();
}
$('missingViews').addEventListener('change', syncStrategy);
// Initial sync runs at the end of this module: syncStrategy reaches
// updateCoverageNote, which reads a const declared further down. Calling it here
// threw a temporal-dead-zone ReferenceError that aborted the rest of the file,
// silently leaving the image-slot handlers unattached.

$('fitAngles').addEventListener('change', () => {
  $('fitOpts').style.display = $('fitAngles').checked ? '' : 'none';
  // Off-axis art has no canonical yaw, so a turnaround has to be generated.
  if ($('fitAngles').checked && strategy() === 'none') {
    $('missingViews').value = 'before';
    syncStrategy();
  }
  updateCoverageNote();
});
$('fitOpts').style.display = 'none';



function syncLowVram() {
  $('lowVramNote').textContent = $('lowVram').checked
    ? 'Caps atlas to 1024, bake maps to 512, generated views to 768, and turns AO off. '
      + 'Less fine detail, but the shape model and image model stop colliding on a 10GB card.'
    : 'Full quality: 2048 atlas, 1024 bake maps, 1024 generated views. '
      + 'Turn on if you hit an out-of-memory error.';
}
$('lowVram').addEventListener('change', syncLowVram);
syncLowVram();

/**
 * Symmetry note. Whether it helps depends entirely on which views you supplied,
 * so the note says what it will actually do with the ones you have rather than
 * describing the option in the abstract.
 */
function syncSymmetric() {
  const on = $('symmetric').checked;
  const have = VIEWS.filter((v) => state.files[v]);
  const note = $('symmetricNote');
  $('rigSymmetrize').checked = on || $('rigSymmetrize').checked;

  if (!on) {
    note.textContent = 'Off: each side is textured only from art that actually shows it.';
    return;
  }
  const mirrors = [];
  if (state.files.left && !state.files.right) mirrors.push('left → right');
  if (state.files.right && !state.files.left) mirrors.push('right → left');
  note.textContent = mirrors.length
    ? 'Will mirror ' + mirrors.join(' and ')
      + ', which is exact for a symmetric subject — better than approximating it. '
      + 'Gaps also borrow from the mirrored side.'
    : (have.length
        ? 'Both sides already have art, so nothing is mirrored; only the gap fill '
          + 'borrows from the mirrored side.'
        : 'Add a left or right view and it will be mirrored onto the other side.');
}
$('symmetric').addEventListener('change', syncSymmetric);

bindSlider('controlNetStrength', 'controlNetStrengthOut', (v) => Number(v).toFixed(2));
bindSlider('controlNetDenoise', 'controlNetDenoiseOut', (v) => Number(v).toFixed(2));
function syncControlNet() {
  $('cnOpts').hidden = !$('controlNet').value;
}
$('controlNet').addEventListener('change', syncControlNet);

bindSlider('paletteColors', 'paletteColorsOut',
           (v) => (Number(v) < 2 ? 'off' : v + ' colours'));
bindSlider('paletteStrength', 'paletteStrengthOut', (v) => Number(v).toFixed(2));
function syncPalette() {
  const on = Number($('paletteColors').value) >= 2;
  $('paletteOpts').hidden = !on;
  $('paletteShadeRow').hidden = !on;
}
$('paletteColors').addEventListener('input', syncPalette);

bindSlider('rigSmoothing', 'rigSmoothingOut');
bindSlider('rigMinConf', 'rigMinConfOut', (v) => Number(v).toFixed(2));
function syncRig() { $('rigOpts').hidden = !$('autoRig').checked; }
$('autoRig').addEventListener('change', syncRig);

bindSlider('kontextGuidance', 'kontextGuidanceOut', (v) => Number(v).toFixed(1));
bindSlider('fitYawRange', 'fitYawRangeOut', (v) => '±' + v + '°');

let backendInitialised = false;
function syncBackend() {
  const kontext = $('synthBackend').value === 'kontext';

  // The same slider means opposite things per backend. Kontext rebuilds from its
  // reference latents and wants denoise 1.0; a low value there returns the render
  // barely changed, which looks like a smeared non-result. SDXL img2img is the
  // reverse. Reset to the right default whenever the backend changes.
  if (backendInitialised) {
    $('synthDenoise').value = kontext ? 1.0 : 0.55;
    $('synthDenoise').dispatchEvent(new Event('input'));
  }
  backendInitialised = true;
  $('synthDenoise').closest('.field').querySelector('label').firstChild.textContent =
    kontext ? 'Rebuild strength ' : 'Freedom ';
  $('sdxlModelRow').style.display = kontext ? 'none' : '';
  $('kontextGuidanceRow').style.display = kontext ? '' : 'none';
  // Kontext is guidance-distilled: it runs at cfg 1 with its own guidance value,
  // and the negative prompt is zeroed, so those controls do nothing there.
  $('synthCfg').closest('.field').style.display = kontext ? 'none' : '';
  $('synthNegative').closest('.field').style.display = kontext ? 'none' : '';
  $('backendNote').textContent = kontext
    ? 'Your source image is passed in as a second reference, so generated views inherit its palette and shapes. Write the prompt as an instruction ("show the same character from the side"), not a description.'
    : 'The model only sees the render, never your source art — style comes from the checkpoint and prompt alone. Faster, and lighter on VRAM.';
}
$('synthBackend').addEventListener('change', syncBackend);
syncBackend();

// Ballpark direct coverage by view count. The 2- and 4-view figures are measured
// on a humanoid at the default settings; 1 and 3 are interpolated. Real coverage
// depends on the silhouette, and mask erosion and the occlusion test both reduce
// it, so the run logs its own exact figure and the coverage map shows where.
const COVERAGE_BY_VIEWS = { 0: 0, 1: 32, 2: 56, 3: 66, 4: 74 };
function updateCoverageNote() {
  const have = VIEWS.filter((v) => state.files[v]);
  const missing = VIEWS.filter((v) => !state.files[v]);
  const n = have.length;
  const pct = COVERAGE_BY_VIEWS[Math.min(n, 4)] || 0;

  if (n === 0) { $('coverageNote').textContent = 'Add reference views to estimate coverage.'; }
  else if (strategy() !== 'none' && missing.length) {
    $('coverageNote').textContent =
      n + (n === 1 ? ' view' : ' views') + ' covers about ' + pct + '% directly; ' +
      missing.join(' and ') + ' will be generated to cover the rest.';
  } else {
    $('coverageNote').textContent =
      n + (n === 1 ? ' view' : ' views') + ': expect roughly ' + pct +
      '% of the surface taken straight from your art, the rest approximated from it. ' +
      'The coverage map shows exactly where.';
  }

  // With angle fitting on, the supplied art sits at its own recovered angle, so
  // every canonical view still has to be generated.
  const willGenerate = $('fitAngles').checked ? VIEWS.slice() : missing;
  $('synthNote').textContent = willGenerate.length === 0
    ? 'You have all four views — nothing to generate.'
    : 'Will generate: ' + willGenerate.join(', ') +
      '. Each is rendered off the part-textured model first, then repainted in your style.';

  const labels = ['front', 'left', 'back', 'right'];
  $('fitNote').textContent = $('fitAngles').checked
    ? 'Each image\'s camera angle is recovered from the model silhouette, then a ' +
      'full ' + labels.join('/') + ' turnaround is generated from it. Slot names ' +
      'are just labels here — put your art in any of them.'
    : 'Off: slots are treated as a level front/left/back/right turnaround. Turn on ' +
      'if your art is isometric or three-quarter, and the angle is recovered instead.';
}

// Triangle budget: slider stays in the 5k-50k band unless explicitly unlocked.
$('tris').addEventListener('input', () => { $('trisExact').value = $('tris').value; });
$('unlockTris').addEventListener('change', (e) => {
  $('trisExact').disabled = !e.target.checked;
  $('tris').disabled = e.target.checked;
});
$('trisExact').addEventListener('input', () => {
  const v = Number($('trisExact').value);
  if (v >= 5000 && v <= 50000) { $('tris').value = v; $('trisOut').textContent = thousands(v); }
  else { $('trisOut').textContent = thousands(v); }
});
function targetTriangles() {
  return $('unlockTris').checked ? Number($('trisExact').value) : Number($('tris').value);
}

$('remesh').addEventListener('change', () => syncTextureUI());
$('texture').addEventListener('change', () => syncTextureUI());

/**
 * Two colour sources, chosen by where the geometry came from:
 *   TRELLIS.2 - its own colour voxel cascade, which describes that exact mesh.
 *   Hunyuan   - your reference art projected onto the mesh from each view.
 * Normal/AO bakes compare a mesh to its own dense version, so they always apply.
 */
function syncTextureUI() {
  const trellis = $('geometry').value === 'trellis2';
  const tex = $('texture');

  const projReady = $('texture').dataset.projReady !== '0';
  tex.disabled = !trellis && !projReady;
  if (tex.disabled) tex.checked = false;

  $('textureNote').textContent = trellis
    ? 'Shape and colour come from the same cascade, so the atlas lines up exactly.'
    : (projReady
        ? 'Your reference images are projected onto the mesh — every surface takes colour from whichever view sees it most directly. All four views texture the model, even when only the front drives the shape.'
        : 'Restart ComfyUI to load the projection nodes, then this becomes available.');

  $('textureOpts').style.display = tex.checked ? '' : 'none';
  $('projOpts').style.display = (!trellis && tex.checked) ? '' : 'none';
  $('useRefRow').style.display = (trellis && tex.checked) ? '' : 'none';

  $('remesh').disabled = !trellis;
  $('remesh').closest('.check').title = trellis ? ''
    : 'RemeshMesh resamples into a fixed [-0.5, 0.5] cube and clips Hunyuan meshes.';
  $('remeshRes').closest('.field').style.display =
    (trellis && $('remesh').checked) ? '' : 'none';
}
$('randomSeed').addEventListener('click', () => {
  $('seed').value = Math.floor(Math.random() * 0xffffffff);
});

// Model-specific defaults, so switching geometry does the right thing.
const GEOMETRY_NOTES = {
  hy3d_mv: 'All four views drive the shape, and all four are projected back on as texture.',
  hy3d_21: 'Front view drives the shape; every view you supply still textures the result.',
  trellis2: 'Generated shape and colour from one cascade. Front view only, art is not projected.',
};
$('geometry').addEventListener('change', () => {
  const g = $('geometry').value;
  $('geometryNote').textContent = GEOMETRY_NOTES[g];
  if (g === 'hy3d_21') { $('steps').value = 30; $('cfg').value = 5; $('latentRes').value = 4096; }
  else if (g === 'hy3d_mv') { $('steps').value = 20; $('cfg').value = 7.5; $('latentRes').value = 3072; }
  for (const [a, b] of [['steps', 'stepsOut'], ['cfg', 'cfgOut'], ['latentRes', 'latentResOut']]) {
    $(b).textContent = $(a).value;
  }
  // Only TRELLIS.2 ignores the extra views outright — the Hunyuan modes always
  // project every supplied view back onto the mesh as texture.
  document.querySelectorAll('.slot[data-view]').forEach((s) => {
    const front = s.dataset.view === 'front';
    const used = g !== 'trellis2' || front;
    s.style.opacity = used ? '1' : '.4';
    s.title = used
      ? (front || g === 'hy3d_mv' ? '' : 'Textures the model, but does not drive the shape')
      : 'Not used by this model';
  });
  syncTextureUI();
  updateCoverageNote();
});
$('geometry').dispatchEvent(new Event('change'));

// -------------------------------------------------------------- naming ----
/**
 * The name every file this run writes is built from.
 *
 * Sanitised to what ComfyUI's filename handling accepts, so a name with spaces
 * or punctuation cannot produce a path the loaders later fail to resolve. The
 * server clamps this too; doing it here as well is what makes the preview
 * honest about the name you will actually get.
 */
function modelName() {
  const raw = ($('modelName').value || '').trim();
  const safe = raw.replace(/[^A-Za-z0-9_\- ]/g, '').replace(/\s+/g, '-');
  return safe || 'pipeline';
}

function syncName() {
  $('namePreview').textContent = '3d/' + modelName() + '_00001_.glb';
}
$('modelName').addEventListener('input', syncName);

// ----------------------------------------------------------- image slots ---
// Scoped to [data-view]: the Image Lab tab reuses the .slot class for its own
// source images, and a bare '.slot' selector picked those up too — registering
// pipeline handlers against an undefined view, and throwing here because the lab
// slots have no remove button, which aborted the rest of this module.
document.querySelectorAll('.slot[data-view]').forEach((slot) => {
  const view = slot.dataset.view;
  const input = slot.querySelector('input');
  const thumb = slot.querySelector('.thumb');

  const clear = slot.querySelector('.slot-clear');

  const accept = async (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    thumb.style.backgroundImage = 'url(' + URL.createObjectURL(file) + ')';
    slot.classList.add('filled');
    const safe = view + '_' + Date.now() + '_' + file.name.replace(/[^A-Za-z0-9_.\-]/g, '_');
    const res = await fetch('/api/upload?name=' + encodeURIComponent(safe), {
      method: 'POST', body: file,
    });
    const json = await res.json();
    if (!res.ok) { log('upload failed: ' + JSON.stringify(json), 'e'); return; }
    // ComfyUI may rename or place the file in a subfolder.
    state.files[view] = (json.subfolder ? json.subfolder + '/' : '') + json.name;
    log('uploaded ' + view + ' → ' + state.files[view]);
    updateCoverageNote();
    syncSymmetric();
  };

  input.addEventListener('change', () => accept(input.files[0]));
  slot.addEventListener('dragover', (e) => { e.preventDefault(); slot.classList.add('dragover'); });
  slot.addEventListener('dragleave', () => slot.classList.remove('dragover'));
  slot.addEventListener('drop', (e) => {
    e.preventDefault(); slot.classList.remove('dragover');
    accept(e.dataTransfer.files[0]);
  });

  // The slot is a <label> wrapping the file input, so a click anywhere inside
  // it opens the picker. Both calls are needed on the remove button: stopping
  // propagation alone still lets the label's default activation through.
  clear.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    delete state.files[view];
    input.value = '';                     // so re-picking the same file still fires
    thumb.style.backgroundImage = '';
    slot.classList.remove('filled');
    log('removed ' + view);
    updateCoverageNote();
    syncStrategy();
    syncSymmetric();
  });
});

// ------------------------------------------------------------ preflight ----
const MODEL_LABELS = {
  hy3d_mv: 'Hunyuan3D 2.0-MV', hy3d_21: 'Hunyuan3D 2.1', t2_unet: 'TRELLIS.2 UNet',
  t2_shape_vae: 'shape VAE', t2_tex_vae: 'texture VAE', t2_clipvision: 'DINOv3', bg: 'BiRefNet',
};
async function preflight() {
  try {
    const r = await fetch('/api/preflight');
    const j = await r.json();
    if (j.error) { $('preflight').innerHTML = '<span class="missing">' + j.error + '</span>'; return; }
    const missing = Object.entries(j.models).filter(([, v]) => !v.ok).map(([k]) => MODEL_LABELS[k] || k);
    const bits = [];
    if (missing.length) bits.push('<span class="missing">missing: ' + missing.join(', ') + '</span>');
    else bits.push('<span class="ok">all models ready</span>');
    if (j.projection && !j.projection.ok) {
      bits.push('<span class="missing">projection nodes not loaded &mdash; restart ComfyUI</span>');
    }
    if (j.rigging && !j.rigging.ok) {
      bits.push('<span class="missing">rigging nodes not loaded &mdash; restart ComfyUI</span>');
    }
    $('preflight').innerHTML = bits.join(' &middot; ');

    // Rigging needs its custom nodes; disable the option rather than letting the
    // run fail at the last stage, after the whole model has been built.
    const rigReady = !j.rigging || j.rigging.ok;
    $('autoRig').disabled = !rigReady;
    if (!rigReady) { $('autoRig').checked = false; syncRig(); }

    // ControlNets installed since the panel loaded should appear without a reload.
    const cns = j.controlNets || [];
    const cnSel = $('controlNet');
    if (cnSel.dataset.list !== cns.join('|')) {
      cnSel.dataset.list = cns.join('|');
      const keep = cnSel.value;
      cnSel.innerHTML = '<option value="">Off &mdash; recolour the render only</option>';
      for (const c of cns) {
        const o = document.createElement('option');
        o.value = c;
        // The filename is a mouthful; the folder it came from is the useful part.
        o.textContent = c.replace(/\.safetensors$/i, '').replace(/[_-]/g, ' ');
        cnSel.appendChild(o);
      }
      if (cns.includes(keep)) cnSel.value = keep;
      syncControlNet();
    }
    $('cnRow').hidden = cns.length === 0;

    // Projection texturing needs those nodes; say so instead of failing at runtime.
    const projReady = !j.projection || j.projection.ok;
    $('texture').dataset.projReady = projReady ? '1' : '0';

    // View synthesis needs a 2D checkpoint; offer whatever is installed.
    const sel = $('imageModel');
    const models = j.imageModels || [];
    if (sel.dataset.list !== models.join('|')) {
      sel.dataset.list = models.join('|');
      const keep = sel.value;
      sel.innerHTML = '';
      for (const m of models) {
        const o = document.createElement('option');
        o.value = m; o.textContent = m;
        sel.appendChild(o);
      }
      if (models.includes(keep)) sel.value = keep;
    }
    const hasImageModel = models.length > 0;
    // Every strategy except "none" needs a 2D model to run.
    for (const o of $('missingViews').options) {
      if (o.value !== 'none') o.disabled = !hasImageModel;
    }
    if (!hasImageModel && strategy() !== 'none') {
      $('missingViews').value = 'none';
      syncStrategy();
    }
    $('missingViews').title = hasImageModel ? ''
      : 'Install a 2D checkpoint in ComfyUI/models/checkpoints to enable this.';

    syncTextureUI();
    updateCoverageNote();
  } catch (e) {
    $('preflight').innerHTML = '<span class="missing">ComfyUI unreachable</span>';
  }
}
preflight();
setInterval(preflight, 20000);

/** Show the previous run's steps on load, so a finished result is never lost. */
async function loadLastRun() {
  try {
    const r = await fetch('/api/last-run');
    const j = await r.json();
    // A run with intermediates turned off has no step images but still has a
    // model, and that model is what the Mixamo and FBX exports act on. Bailing
    // on the image count used to leave those controls disabled after a
    // perfectly good run.
    if (j.images && j.images.length) {
      renderSteps(j.images);
      log('showing steps from the previous run — generate to replace them');
    }
    if (j.glb) {
      const url = viewUrl(j.glb);
      $('download').href = url;
      $('download').download = j.glb.filename.split('/').pop();
      $('download').hidden = false;
      state.lastGlb = j.rigged || null;
      state.lastPlainGlb = j.plain || null;
      state.shownGlb = j.glb; state.fbxUrl = null; state.zipUrl = null;
      syncAnimTarget();
      syncMixamo();
      syncFbxButton();
      showModel(url);
    }
  } catch (e) { /* nothing to restore */ }
}
loadLastRun();

// ------------------------------------------------------------ websocket ----
function connect() {
  const ws = new WebSocket('ws://' + location.host + '/ws?clientId=' + CLIENT_ID);
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;   // binary previews
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    const d = m.data || {};
    if (d.prompt_id && state.promptId && d.prompt_id !== state.promptId) return;

    switch (m.type) {
      case 'execution_start':
        state.cached.clear(); log('execution started', 'k'); break;
      case 'execution_cached':
        (d.nodes || []).forEach((n) => state.cached.add(n)); break;
      case 'executing':
        // In a split run each stage ends with a null node; the run is only over
        // when the job says so, so let followJob decide.
        if (d.node === null) { if (!state.jobId) finish(); }
        else { stage(state.titles[d.node] || ('node ' + d.node)); }
        break;
      case 'progress':
        if (d.max) bar((d.value / d.max) * 100);
        break;
      case 'execution_error':
        stage('error in ' + (d.node_type || 'workflow'), 'err');
        log((d.node_type || '') + ': ' + (d.exception_message || 'unknown error'), 'e');
        if (d.traceback) log([].concat(d.traceback).join('\n'), 'e');
        setRunning(false);
        break;
      case 'execution_interrupted':
        stage('cancelled'); setRunning(false); break;
    }
  };
  ws.onclose = () => setTimeout(connect, 2000);
}
connect();

// ----------------------------------------------------------- staged run ----
/**
 * Follow a multi-stage job. The server runs the prompts back to back; this
 * tracks which one is live so the websocket keeps showing real node progress,
 * and reports the stage boundaries rather than leaving a silent gap while
 * ComfyUI swaps models.
 */
async function followJob(jobId) {
  let lastStage = null;
  while (state.jobId === jobId) {
    await new Promise((r) => setTimeout(r, 1500));
    let j;
    try {
      j = await (await fetch('/api/job/' + jobId)).json();
    } catch (e) {
      continue;                       // transient; the job keeps running
    }
    if (state.jobId !== jobId) return;   // cancelled or superseded

    // Point the websocket filter at whichever prompt is running now.
    if (j.promptId && j.promptId !== state.promptId) {
      state.promptId = j.promptId;
      state.saveNode = j.saveNode;
      state.titles = j.titles || {};
      state.cached.clear();
    }
    if (j.stage && j.stage !== lastStage) {
      lastStage = j.stage;
      if (!j.done) {
        stage('stage ' + (j.stageIndex + 1) + '/' + j.stageCount + ' · ' + j.stage);
        log('stage ' + (j.stageIndex + 1) + '/' + j.stageCount + ': ' + j.stage
            + ' (' + j.nodeCount + ' nodes)', 'k');
      }
    }

    if (j.error) {
      stage('failed', 'err');
      log(j.error, 'e');
      state.jobId = null;
      setRunning(false);
      return;
    }
    if (j.done) {
      state.jobId = null;
      renderSteps(j.images || []);
      const rigged = (j.glbs || []).find((f) => /rigged/i.test(f.filename));
      // Ignore the intermediate stage-1 and dense meshes; the final model is
      // the one without those markers.
      const finals = (j.glbs || []).filter((f) => !/-stage1_|-dense_/.test(f.filename));
      const file = rigged || finals[finals.length - 1] || null;
      if (!file) {
        stage('finished, but no final GLB was written', 'err');
        setRunning(false);
        return;
      }
      state.lastGlb = rigged || null;
      state.lastPlainGlb = finals.find((f) => !/rigged/i.test(f.filename)) || null;
      state.shownGlb = file; state.fbxUrl = null; state.zipUrl = null;
      syncAnimTarget();
      syncMixamo();
      syncFbxButton();
      const url = viewUrl(file);
      $('download').href = url;
      $('download').download = file.filename.split('/').pop();
      $('download').hidden = false;
      showModel(url);
      stage('done — ' + file.filename, 'done');
      bar(100);
      setRunning(false);
      return;
    }
  }
}

// --------------------------------------------------------------- run ------
function setRunning(on) {
  state.running = on;
  $('run').disabled = on;
  $('cancel').disabled = !on;
  if (!on) bar(on ? 0 : 100);
}

/** ComfyUI output descriptor -> a URL the panel can fetch through the proxy. */
function viewUrl(f) {
  return '/api/view?filename=' + encodeURIComponent(f.filename) +
    '&subfolder=' + encodeURIComponent(f.subfolder || '') +
    '&type=' + encodeURIComponent(f.type || 'output');
}

// Human-readable names for the saved stage images, keyed by filename prefix.
const STEP_LABELS = [
  ['step0-refview-', 'Generated reference view'],
  ['step1-anglefit-', 'Fitted camera angle (cyan = model)'],
  ['step2-atlas-real', 'Atlas from your art'],
  ['step2-coverage-real', 'Coverage (white = your art)'],
  ['step3-render-', 'Render of missing view'],
  ['step3b-gaps-', 'Gap mask (repainted area)'],
  ['step4-synth-', 'Generated view'],
  ['step5-geomview-', 'Framed for shape pass'],
  ['step6-atlas-final', 'Final atlas'],
];

function labelFor(name) {
  for (const [prefix, label] of STEP_LABELS) {
    if (name.startsWith(prefix)) {
      const rest = name.slice(prefix.length).replace(/_\d+_?$/, '');
      return rest ? label + ' — ' + rest : label;
    }
  }
  return name;
}

// ------------------------------------------------------------- lightbox ----
let lbItems = [];
let lbIndex = 0;

function lbShow(i) {
  if (!lbItems.length) return;
  lbIndex = (i + lbItems.length) % lbItems.length;
  const it = lbItems[lbIndex];
  $('lbImage').src = it.url;
  $('lbLabel').textContent = it.label + '  (' + (lbIndex + 1) + '/' + lbItems.length + ')';
  $('lbName').textContent = it.filename;
  $('lbOpen').href = it.url;
  $('lightbox').hidden = false;
}
function lbClose() { $('lightbox').hidden = true; $('lbImage').removeAttribute('src'); }

$('lbClose').addEventListener('click', lbClose);
$('lbPrev').addEventListener('click', () => lbShow(lbIndex - 1));
$('lbNext').addEventListener('click', () => lbShow(lbIndex + 1));
// Masks and atlases are worth inspecting texel-exact, so allow nearest sampling.
$('lbPixel').addEventListener('click', () => $('lightbox').classList.toggle('pixelated'));
$('lightbox').addEventListener('click', (e) => { if (e.target === $('lightbox')) lbClose(); });
window.addEventListener('keydown', (e) => {
  if ($('lightbox').hidden) return;
  if (e.key === 'Escape') lbClose();
  else if (e.key === 'ArrowLeft') lbShow(lbIndex - 1);
  else if (e.key === 'ArrowRight') lbShow(lbIndex + 1);
});

/** Renders the saved pipeline stages in run order. */
function renderSteps(files) {
  const grid = $('stepGrid');
  grid.innerHTML = '';
  lbItems = [];
  if (!files.length) { $('stepGallery').hidden = true; return; }

  files.sort((a, b) => a.filename.localeCompare(b.filename, undefined, { numeric: true }));
  files.forEach((f, i) => {
    const url = viewUrl(f);
    const base = f.filename.replace(/\.[a-z]+$/i, '');
    const label = labelFor(base);
    lbItems.push({ url: url, label: label, filename: f.filename });

    const fig = document.createElement('figure');
    fig.className = 'step';
    fig.innerHTML = '<img loading="lazy" alt=""><figcaption></figcaption>';
    fig.querySelector('img').src = url;
    fig.querySelector('figcaption').textContent = label;
    fig.title = f.filename + ' — click to enlarge';
    fig.addEventListener('click', () => lbShow(i));
    grid.appendChild(fig);
  });
  $('stepGallery').hidden = false;
}

async function finish() {
  stage('fetching result');
  try {
    const r = await fetch('/api/history/' + state.promptId);
    const hist = await r.json();
    const entry = hist[state.promptId];
    if (!entry) { stage('finished, but no history entry', 'err'); setRunning(false); return; }

    // Sort every output into the model itself and the per-stage images.
    let file = null;
    const steps = [];
    const glbs = [];
    for (const out of Object.values(entry.outputs || {})) {
      for (const arr of Object.values(out)) {
        if (!Array.isArray(arr)) continue;
        for (const f of arr) {
          if (!f || typeof f.filename !== 'string') continue;
          if (/\.(glb|gltf)$/i.test(f.filename)) glbs.push(f);
          else if (/\.(png|jpe?g|webp)$/i.test(f.filename)) steps.push(f);
        }
      }
    }
    renderSteps(steps);
    // With auto-rig on there are two models. Show the skinned one — it is the
    // full result, and the only one an animation can be applied to.
    const rigged = glbs.find((f) => /rigged/i.test(f.filename));
    file = rigged || glbs[glbs.length - 1] || null;
    if (!file) { stage('no GLB in the outputs', 'err'); log(JSON.stringify(entry.outputs), 'e'); setRunning(false); return; }
    state.lastGlb = rigged || null;
    state.lastPlainGlb = glbs.find((f) => !/rigged|-stage1_|-dense_/i.test(f.filename)) || null;
    state.shownGlb = file; state.fbxUrl = null; state.zipUrl = null;
    syncAnimTarget();
    syncMixamo();
    syncFbxButton();
    if (rigged && glbs.length > 1) log('rigged model shown; the unrigged one is also in the outputs');

    const url = viewUrl(file);
    $('download').href = url;
    $('download').download = file.filename.split('/').pop();
    $('download').hidden = false;
    showModel(url);
    stage('done — ' + file.filename, 'done');
    bar(100);
  } catch (e) {
    stage('could not fetch result: ' + e.message, 'err');
  }
  setRunning(false);
}

$('run').addEventListener('click', async () => {
  if (!state.files.front) { stage('a front image is required', 'err'); return; }
  setRunning(true);
  bar(0);
  $('log').textContent = '';
  stage('queueing');

  const body = {
    clientId: CLIENT_ID,
    images: state.files,
    filenamePrefix: '3d/' + modelName(),
    stepsPrefix: '3d/' + modelName() + '-steps',
    geometry: $('geometry').value,
    seed: Number($('seed').value),
    steps: Number($('steps').value),
    cfg: Number($('cfg').value),
    latentResolution: Number($('latentRes').value),
    octreeResolution: Number($('octree').value),
    removeBackground: $('rembg').checked,
    targetTriangles: targetTriangles(),
    decimatePlacement: $('placement').value,
    remesh: $('remesh').checked,
    remeshResolution: Number($('remeshRes').value),
    creaseAngle: Number($('crease').value),
    texture: $('texture').checked,
    textureResolution: Number($('texRes').value),
    bakeAO: $('bakeAO').checked,
    bakeNormal: $('bakeNormal').checked,
    autoRig: $('autoRig').checked,
    rigFacing: $('rigFacing').value,
    rigSymmetrize: $('rigSymmetrize').checked,
    rigSmoothing: Number($('rigSmoothing').value),
    rigMinConfidence: Number($('rigMinConf').value),
    useReferenceMesh: $('useRef').checked,
    facingPower: Number($('facing').value),
    projectionOcclusion: $('projOcc').checked,
    symmetric: $('symmetric').checked,
    swapSides: $('swapSides').checked,
    generateReferenceViews: usesRefGen(),
    refViewStyle: $('refViewStyle').value,
    fitAngles: $('fitAngles').checked,
    fitYawRange: Number($('fitYawRange').value),
    fitPitchMin: Number($('fitPitchMin').value),
    fitPitchMax: Number($('fitPitchMax').value),
    minFacing: Number($('minFacing').value),
    maskErode: Number($('maskErode').value),
    fillMode: $('fillMode').value,
    controlNet: $('controlNet').value,
    controlNetType: $('controlNetType').value,
    controlNetStrength: Number($('controlNetStrength').value),
    controlNetDenoise: Number($('controlNetDenoise').value),
    paletteColors: Number($('paletteColors').value),
    paletteStrength: Number($('paletteStrength').value),
    paletteKeepShading: $('paletteKeepShading').checked,
    synthesizeViews: usesSynth(),
    imageModel: $('imageModel').value,
    synthBackend: $('synthBackend').value,
    kontextGuidance: Number($('kontextGuidance').value),
    synthPrompt: $('synthPrompt').value,
    synthNegative: $('synthNegative').value,
    synthDenoise: Number($('synthDenoise').value),
    synthSteps: Number($('synthSteps').value),
    synthCfg: Number($('synthCfg').value),
    synthGapsOnly: $('synthGapsOnly').checked,
    gapGrow: Number($('gapGrow').value),
    syntheticWeight: Number($('synthWeight').value),
    refineGeometry: $('refineGeometry').checked,
    saveIntermediates: true,
    lowVram: $('lowVram').checked,
    viewScale: Number($('viewScale').value),
    viewPitch: Number($('viewPitch').value),
    saveCoverage: $('saveCoverage').checked,
  };

  try {
    const r = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) {
      stage('rejected', 'err');
      log(JSON.stringify(j.detail || j, null, 1), 'e');
      setRunning(false);
      return;
    }
    log('seed ' + j.params.seed + ' · target '
        + j.params.targetTriangles.toLocaleString() + ' tris', 'k');

    if (j.jobId) {
      // Split run: the shape model and the image model are loaded by different
      // prompts so neither has to share the card with the other.
      state.jobId = j.jobId;
      state.promptId = null;
      log('split into ' + j.stages + ' stages: ' + j.stageNames.join(' → '), 'k');
      stage('stage 1/' + j.stages + ' · ' + j.stageNames[0]);
      followJob(j.jobId);
    } else {
      state.jobId = null;
      state.promptId = j.prompt_id;
      state.saveNode = j.saveNode;
      state.titles = j.titles || {};
      log('queued ' + j.nodeCount + ' nodes', 'k');
      stage('running');
    }
  } catch (e) {
    stage('could not reach the server', 'err');
    log(e.message, 'e');
    setRunning(false);
  }
});

$('cancel').addEventListener('click', async () => {
  // Drop the job first, or followJob would queue the next stage anyway.
  state.jobId = null;
  await fetch('/api/interrupt', { method: 'POST' });
  stage('cancelling');
});

// Everything is declared by now, so the first sync is safe here.
syncStrategy();
syncRig();
syncPalette();
loadAnimationList();
syncMixamo();
syncFbxButton();
syncName();
syncSymmetric();
