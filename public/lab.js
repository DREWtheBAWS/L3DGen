/**
 * Image lab: the generator on its own, with no 3D in the graph.
 *
 * Inside the full pipeline a poor result could come from the shape model, the
 * angle fit, the projection, or the image model. This isolates the last one so
 * it can be tuned in seconds rather than half an hour.
 *
 * Standalone on purpose - it polls for its own result instead of sharing the
 * pipeline's websocket, so neither tab can break the other.
 */

const $ = (id) => document.getElementById(id);
const lab = { files: {}, running: false, promptId: null, timer: null };

// ------------------------------------------------------------------ tabs ----
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    const want = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('is-active', b === btn));
    $('tab-pipeline').hidden = want !== 'pipeline';
    $('tab-lab').hidden = want !== 'lab';
  });
});

// --------------------------------------------------------------- helpers ----
function labLog(msg, cls) {
  const el = $('labLog');
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}
function labStage(text, cls) {
  $('labStage').textContent = text;
  $('labStage').className = 'stage' + (cls ? ' ' + cls : '');
}
function labBar(pct) { $('labBarFill').style.width = Math.max(0, Math.min(100, pct)) + '%'; }
function labBusy(on) {
  lab.running = on;
  $('labRun').disabled = on;
  $('labCancel').disabled = !on;
}
function viewUrl(f) {
  return '/api/view?filename=' + encodeURIComponent(f.filename) +
    '&subfolder=' + encodeURIComponent(f.subfolder || '') +
    '&type=' + encodeURIComponent(f.type || 'output');
}
function bind(id, outId, fmt) {
  const el = $(id), out = $(outId);
  const sync = () => { out.textContent = fmt ? fmt(el.value) : el.value; };
  el.addEventListener('input', sync);
  sync();
}

bind('labDenoise', 'labDenoiseOut', (v) => Number(v).toFixed(2));
bind('labGuidance', 'labGuidanceOut', (v) => Number(v).toFixed(1));
bind('labCfg', 'labCfgOut', (v) => Number(v).toFixed(1));
bind('labSteps', 'labStepsOut');

// -------------------------------------------------------------- backend ----
function syncLabBackend(resetDenoise) {
  const kontext = $('labBackend').value === 'kontext';
  $('labModelRow').style.display = kontext ? 'none' : '';
  $('labGuidanceRow').style.display = kontext ? '' : 'none';
  $('labCfgRow').style.display = kontext ? 'none' : '';
  $('labNegRow').style.display = kontext ? 'none' : '';

  // At denoise 1.0 img2img keeps nothing of the init image - it is plain
  // text-to-image. With a prompt that describes style but names no subject, the
  // model then invents whatever it likes. Kontext is the opposite: it carries
  // the image through its reference latents, so 1.0 is correct there.
  $('labDenoise').max = kontext ? 1 : 0.9;
  if (resetDenoise) {
    $('labDenoise').value = kontext ? 1.0 : 0.55;
  }
  if (Number($('labDenoise').value) > Number($('labDenoise').max)) {
    $('labDenoise').value = $('labDenoise').max;
  }
  $('labDenoise').dispatchEvent(new Event('input'));

  $('labDenoiseNote').textContent = kontext
    ? 'Kontext rebuilds from its references, so keep this at 1.0. Lower values hand back the input barely changed. Use instruction strength to trade prompt against reference.'
    : 'SDXL img2img: lower keeps more of the source, higher reinvents. 0.4-0.6 is the usable band. Capped at 0.9 because 1.0 ignores your image completely and generates from the prompt alone.';
}
$('labBackend').addEventListener('change', () => syncLabBackend(true));
syncLabBackend(true);

/**
 * Turbo / Lightning / Hyper checkpoints are step-distilled: they expect roughly
 * 4-8 steps at CFG 1-2, and look burnt and oversaturated at ordinary settings.
 */
function syncTurbo() {
  const name = ($('labModel').value || '').toLowerCase();
  const turbo = /turbo|lightning|hyper|lcm/.test(name);
  const note = $('labModelNote');
  if (!note) return;
  note.textContent = turbo
    ? 'This is a step-distilled checkpoint — set steps to about 6 and CFG to about 2. At 25 steps / CFG 6 it will look burnt.'
    : '';
  note.style.display = turbo ? '' : 'none';
}
$('labModel').addEventListener('change', syncTurbo);

$('labApplyTurbo') && $('labApplyTurbo').addEventListener('click', () => {
  $('labSteps').value = 6; $('labSteps').dispatchEvent(new Event('input'));
  $('labCfg').value = 2; $('labCfg').dispatchEvent(new Event('input'));
});

$('labRandomSeed').addEventListener('click', () => {
  $('labSeed').value = Math.floor(Math.random() * 0xffffffff);
});

// Offer whatever checkpoints ComfyUI can see, same as the pipeline tab.
async function labModels() {
  try {
    const j = await (await fetch('/api/preflight')).json();
    const models = j.imageModels || [];
    const sel = $('labModel');
    if (sel.dataset.list === models.join('|')) return;
    sel.dataset.list = models.join('|');
    const keep = sel.value;
    sel.innerHTML = '';
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m; o.textContent = m;
      sel.appendChild(o);
    }
    if (models.includes(keep)) sel.value = keep;
    syncTurbo();
  } catch (e) { /* offline; the pipeline tab reports it */ }
}
labModels();
setInterval(labModels, 20000);

// ---------------------------------------------------------------- slots ----
document.querySelectorAll('.slot[data-lab]').forEach((slot) => {
  const key = slot.dataset.lab;
  const input = slot.querySelector('input');
  const thumb = slot.querySelector('.thumb');

  const accept = async (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const url = URL.createObjectURL(file);
    thumb.style.backgroundImage = 'url(' + url + ')';
    slot.classList.add('filled');
    if (key === 'front') $('labSrcImg').src = url;

    const safe = 'lab_' + key + '_' + Date.now() + '_' +
      file.name.replace(/[^A-Za-z0-9_.\-]/g, '_');
    const res = await fetch('/api/upload?name=' + encodeURIComponent(safe),
      { method: 'POST', body: file });
    const json = await res.json();
    if (!res.ok) { labLog('upload failed: ' + JSON.stringify(json), 'e'); return; }
    lab.files[key] = (json.subfolder ? json.subfolder + '/' : '') + json.name;
    labLog('uploaded ' + key + ' -> ' + lab.files[key]);
  };

  input.addEventListener('change', () => accept(input.files[0]));
  slot.addEventListener('dragover', (e) => { e.preventDefault(); slot.classList.add('dragover'); });
  slot.addEventListener('dragleave', () => slot.classList.remove('dragover'));
  slot.addEventListener('drop', (e) => {
    e.preventDefault();
    slot.classList.remove('dragover');
    accept(e.dataTransfer.files[0]);
  });
});

// ------------------------------------------------------------------ run ----
$('labRun').addEventListener('click', async () => {
  if (!lab.files.front) { labStage('a source image is required', 'err'); return; }
  labBusy(true);
  labBar(0);
  $('labLog').textContent = '';
  labStage('queueing');

  const body = {
    images: { front: lab.files.front, back: lab.files.back },
    synthBackend: $('labBackend').value,
    imageModel: $('labModel').value,
    synthPrompt: $('labPrompt').value,
    synthNegative: $('labNegative').value,
    synthDenoise: Number($('labDenoise').value),
    synthSteps: Number($('labSteps').value),
    synthCfg: Number($('labCfg').value),
    kontextGuidance: Number($('labGuidance').value),
    seed: Number($('labSeed').value),
  };

  try {
    const r = await fetch('/api/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!r.ok) {
      labStage('rejected', 'err');
      labLog(JSON.stringify(j.detail || j, null, 1), 'e');
      labBusy(false);
      return;
    }
    lab.promptId = j.prompt_id;
    labLog('queued ' + j.nodeCount + ' nodes · ' + j.params.synthBackend +
           ' · denoise ' + j.params.synthDenoise +
           ' · steps ' + j.params.synthSteps +
           ' · seed ' + j.params.seed, 'k');
    labStage('running');
    labBar(10);
    labPoll();
  } catch (e) {
    labStage('could not reach the server', 'err');
    labLog(e.message, 'e');
    labBusy(false);
  }
});

$('labCancel').addEventListener('click', async () => {
  await fetch('/api/interrupt', { method: 'POST' });
  labStage('cancelling');
});

/** Poll history until this prompt finishes; the pipeline tab owns the socket. */
function labPoll() {
  clearInterval(lab.timer);
  let ticks = 0;
  lab.timer = setInterval(async () => {
    ticks += 1;
    labBar(Math.min(90, 10 + ticks * 2));
    let entry;
    try {
      const hist = await (await fetch('/api/history/' + lab.promptId)).json();
      entry = hist[lab.promptId];
    } catch (e) { return; }
    if (!entry) return;

    const st = entry.status || {};
    if (st.status_str === 'error') {
      clearInterval(lab.timer);
      const err = (st.messages || []).find((m) => m[0] === 'execution_error');
      labStage('failed', 'err');
      if (err) labLog(err[1].node_type + ': ' + err[1].exception_message, 'e');
      labBusy(false);
      return;
    }
    if (!st.completed) return;

    clearInterval(lab.timer);
    let file = null;
    for (const out of Object.values(entry.outputs || {})) {
      for (const arr of Object.values(out)) {
        if (!Array.isArray(arr)) continue;
        for (const f of arr) if (f && /\.(png|jpe?g|webp)$/i.test(f.filename || '')) file = f;
      }
    }
    if (!file) { labStage('no image in the outputs', 'err'); }
    else {
      $('labOutImg').src = viewUrl(file);
      labStage('done — ' + file.filename, 'done');
      labBar(100);
    }
    labBusy(false);
  }, 2000);
}

// ------------------------------------------------- load a pipeline render ----
// The generator's real job is restyling a render that is already at the target
// viewpoint. Those live in ComfyUI's output folder, but LoadImage only reads the
// input folder - so fetch the bytes back through the proxy and re-upload them.
async function labPopulateRuns() {
  try {
    const j = await (await fetch('/api/last-run')).json();
    const imgs = (j.images || []).filter((f) =>
      /step3-render-|step4-synth-|step2-atlas/.test(f.filename));
    const sel = $('labFromRun');
    const key = imgs.map((f) => f.filename).join('|');
    if (sel.dataset.list === key) return;
    sel.dataset.list = key;
    sel.innerHTML = '<option value="">- pick a stage image -</option>';
    for (const f of imgs) {
      const o = document.createElement('option');
      o.value = JSON.stringify(f);
      o.textContent = f.filename.replace(/_\d+_\.png$/, '');
      sel.appendChild(o);
    }
  } catch (e) { /* nothing to offer yet */ }
}
labPopulateRuns();
setInterval(labPopulateRuns, 20000);

$('labFromRun').addEventListener('change', async (e) => {
  if (!e.target.value) return;
  const f = JSON.parse(e.target.value);
  labStage('loading ' + f.filename);
  try {
    const blob = await (await fetch(viewUrl(f))).blob();
    const name = 'lab_render_' + Date.now() + '_' +
      f.filename.replace(/[^A-Za-z0-9_.\-]/g, '_');
    const res = await fetch('/api/upload?name=' + encodeURIComponent(name),
      { method: 'POST', body: blob });
    const json = await res.json();
    if (!res.ok) { labStage('could not load it', 'err'); return; }
    lab.files.front = (json.subfolder ? json.subfolder + '/' : '') + json.name;

    const url = URL.createObjectURL(blob);
    $('labSrcImg').src = url;
    const slot = document.querySelector('.slot[data-lab="front"]');
    slot.querySelector('.thumb').style.backgroundImage = 'url(' + url + ')';
    slot.classList.add('filled');
    labLog('source <- ' + f.filename, 'k');
    labStage('ready');
  } catch (err) {
    labStage('could not load it: ' + err.message, 'err');
  }
});
