'use strict';
/**
 * Local control panel for the image -> 3D ComfyUI pipeline.
 *
 *   node server.js [--port 8189] [--comfy 127.0.0.1:8188]
 *
 * Zero dependencies. Everything the browser needs is proxied through here so we
 * never depend on ComfyUI having CORS enabled, including its progress websocket.
 */

const http = require('http');
const net = require('net');
const fs = require('fs');
const path = require('path');
const { buildPrompt, buildStages, missingAngles, buildImagePrompt, VIEWS, MODELS } = require('./lib/workflow');
const { normaliseParams } = require('./lib/params');
const animate = require('./lib/animate');

// ---------------------------------------------------------------- args ------
const argv = process.argv.slice(2);
const argOf = (flag, dflt) => {
  const i = argv.indexOf(flag);
  return i !== -1 && argv[i + 1] ? argv[i + 1] : dflt;
};
const PORT = parseInt(argOf('--port', '8189'), 10);
const COMFY = argOf('--comfy', '127.0.0.1:8188');
const COMFY_HOST = COMFY.split(':')[0];
// ComfyUI Desktop does not reliably reclaim its port across restarts, so this is
// a starting guess that gets re-discovered whenever the connection fails.
let COMFY_PORT = parseInt(COMFY.split(':')[1] || '8188', 10);
const PUBLIC = path.join(__dirname, 'public');

/** Ports to sweep when ComfyUI is not where we last saw it. */
function candidatePorts() {
  const out = [COMFY_PORT];
  for (let p = 8188; p <= 8199; p++) if (p !== PORT && !out.includes(p)) out.push(p);
  return out;
}

function probe(port) {
  return new Promise((resolve) => {
    const req = http.request(
      { host: COMFY_HOST, port, path: '/system_stats', method: 'GET', timeout: 1500 },
      (res) => { res.resume(); resolve(res.statusCode === 200); });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
    req.end();
  });
}

let discovering = null;
async function discoverComfyPort() {
  if (discovering) return discovering;                       // coalesce concurrent sweeps
  discovering = (async () => {
    for (const p of candidatePorts()) {
      if (await probe(p)) {
        if (p !== COMFY_PORT) console.log('ComfyUI moved to port ' + p);
        COMFY_PORT = p;
        return p;
      }
    }
    return null;
  })();
  try { return await discovering; } finally { discovering = null; }
}

// ------------------------------------------------------------- helpers ------
// ComfyUI binds its port before it can serve, so a request during startup would
// connect and then hang forever without this. A hung request here freezes the
// whole panel, so every call gets a deadline.
const COMFY_TIMEOUT_MS = 30000;

function comfyRequestOnce(method, reqPath, body, headers) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      host: COMFY_HOST, port: COMFY_PORT, path: reqPath, method,
      headers: Object.assign({}, headers || {}),
      timeout: COMFY_TIMEOUT_MS,
    }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({
        status: res.statusCode,
        headers: res.headers,
        body: Buffer.concat(chunks),
      }));
    });
    req.on('timeout', () => {
      req.destroy(new Error('ComfyUI did not respond within ' +
        (COMFY_TIMEOUT_MS / 1000) + 's (still starting up?)'));
    });
    req.on('error', reject);
    if (body) req.write(body);
    req.end();
  });
}

/** Same, but re-finds ComfyUI and retries once if the connection is refused. */
async function comfyRequest(method, reqPath, body, headers) {
  try {
    return await comfyRequestOnce(method, reqPath, body, headers);
  } catch (e) {
    const found = await discoverComfyPort();
    if (!found) throw e;
    return comfyRequestOnce(method, reqPath, body, headers);
  }
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > (limit || 64 * 1024 * 1024)) { reject(new Error('Payload too large')); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function sendJSON(res, status, obj) {
  const buf = Buffer.from(JSON.stringify(obj));
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': buf.length });
  res.end(buf);
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.glb': 'model/gltf-binary', '.ico': 'image/x-icon',
};

// ----------------------------------------------------------- api routes -----
async function handleUpload(req, res, url) {
  const name = (url.searchParams.get('name') || 'image.png').replace(/[^A-Za-z0-9_.\-]/g, '_');
  const data = await readBody(req, 64 * 1024 * 1024);
  if (!data.length) return sendJSON(res, 400, { error: 'Empty upload' });

  const boundary = '----comfy3d' + Date.now().toString(16);
  const head = Buffer.from(
    '--' + boundary + '\r\n' +
    'Content-Disposition: form-data; name="image"; filename="' + name + '"\r\n' +
    'Content-Type: application/octet-stream\r\n\r\n');
  const mid = Buffer.from(
    '\r\n--' + boundary + '\r\n' +
    'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n' +
    '--' + boundary + '\r\n' +
    'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n' +
    '--' + boundary + '--\r\n');
  const payload = Buffer.concat([head, data, mid]);

  const r = await comfyRequest('POST', '/upload/image', payload, {
    'Content-Type': 'multipart/form-data; boundary=' + boundary,
    'Content-Length': payload.length,
  });
  res.writeHead(r.status, { 'Content-Type': 'application/json' });
  res.end(r.body);
}

/** Submit one prompt graph; resolves to ComfyUI's response. */
async function submitPrompt(prompt, clientId) {
  const payload = Buffer.from(JSON.stringify({ prompt: prompt, client_id: clientId }));
  const r = await comfyRequest('POST', '/prompt', payload, {
    'Content-Type': 'application/json', 'Content-Length': payload.length,
  });
  let parsed;
  try { parsed = JSON.parse(r.body.toString('utf8')); } catch (e) { parsed = { raw: r.body.toString('utf8') }; }
  return { status: r.status, json: parsed };
}

/** Poll history until a prompt finishes. Resolves to its history entry. */
async function awaitPrompt(promptId, timeoutMs) {
  const deadline = Date.now() + (timeoutMs || 45 * 60 * 1000);
  while (Date.now() < deadline) {
    const r = await comfyRequest('GET', '/history/' + encodeURIComponent(promptId), null, {});
    if (r.status === 200) {
      let hist;
      try { hist = JSON.parse(r.body.toString('utf8')); } catch (e) { hist = {}; }
      const entry = hist[promptId];
      const st = entry && entry.status;
      // An interrupt is terminal too, and reports neither completed nor error;
      // without this the chain would sit here until the timeout.
      const interrupted = st && (st.messages || [])
        .some((m) => m[0] === 'execution_interrupted');
      if (st && (st.completed || st.status_str === 'error' || interrupted)) {
        return entry;
      }
    }
    await new Promise((r2) => setTimeout(r2, 2000));
  }
  throw new Error('stage timed out');
}

/** Flatten a history entry's outputs into { glbs, images }. */
function collectOutputs(entry) {
  const glbs = [], images = [];
  for (const out of Object.values((entry && entry.outputs) || {})) {
    for (const arr of Object.values(out)) {
      if (!Array.isArray(arr)) continue;
      for (const f of arr) {
        if (!f || typeof f.filename !== 'string') continue;
        if (/\.(glb|gltf)$/i.test(f.filename)) glbs.push(f);
        else if (/\.(png|jpe?g|webp)$/i.test(f.filename)) images.push(f);
      }
    }
  }
  return { glbs: glbs, images: images };
}

/**
 * ComfyUI's path for an output file.
 *
 * On Windows the reported `subfolder` uses backslashes, so joining with "/"
 * yields a mixed "3d\steps/coverage.png"; normalising to forward slashes works
 * on every platform.
 */
const rel = (f) =>
  ((f.subfolder ? f.subfolder + '/' : '') + f.filename).replace(/\\/g, '/');

/**
 * The same path, annotated for the image loaders.
 *
 * LoadImage resolves a bare name against the *input* directory, so a stage
 * handing the next one a file it just wrote to output silently fails validation.
 * The "[output]" suffix is ComfyUI's own convention for this and is what its UI
 * uses when an output image is fed back into a graph.
 */
const relOut = (f) => rel(f) + ' [output]';

/**
 * Jobs currently running, so the panel can follow a multi-stage run. Kept in
 * memory only: a job outlives a single HTTP request but not the server.
 */
const jobs = new Map();

/**
 * Run the stages back to back, feeding each stage the filenames the previous
 * one wrote. ComfyUI frees its models between prompts, which is the entire
 * point: every stage gets the whole card instead of paging against the others.
 */
async function runStages(job, stages, params, clientId) {
  const carried = {};
  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i];
    let built;
    try {
      built = stage.build(carried);
    } catch (e) {
      job.error = 'Could not build stage "' + stage.name + '": ' + e.message;
      job.done = true;
      return;
    }

    const sub = await submitPrompt(built.prompt, clientId);
    if (sub.status !== 200) {
      job.error = 'ComfyUI rejected stage "' + stage.name + '": '
                  + JSON.stringify(sub.json).slice(0, 400);
      job.done = true;
      return;
    }

    job.stage = stage.name;
    job.stageIndex = i;
    job.promptId = sub.json.prompt_id;
    job.saveNode = built.saveNode;
    job.infoNode = built.infoNode;
    job.nodeCount = Object.keys(built.prompt).length;
    job.titles = {};
    for (const [id, node] of Object.entries(built.prompt)) {
      job.titles[id] = (node._meta && node._meta.title) || node.class_type;
    }
    console.log('stage ' + (i + 1) + '/' + stages.length + ' "' + stage.name
                + '" queued as ' + job.promptId + ' (' + job.nodeCount + ' nodes)');

    let entry;
    try {
      entry = await awaitPrompt(job.promptId);
    } catch (e) {
      job.error = 'Stage "' + stage.name + '" ' + e.message;
      job.done = true;
      return;
    }
    if (((entry.status || {}).messages || []).some((m) => m[0] === 'execution_interrupted')) {
      job.error = 'Cancelled during stage "' + stage.name + '".';
      job.done = true;
      return;
    }
    if ((entry.status || {}).status_str === 'error') {
      const msg = (entry.status.messages || [])
        .filter((m) => m[0] === 'execution_error')
        .map((m) => m[1].node_type + ': ' + m[1].exception_message)[0];
      job.error = 'Stage "' + stage.name + '" failed. ' + (msg || 'see the ComfyUI log');
      job.done = true;
      return;
    }

    const outs = collectOutputs(entry);
    job.images = (job.images || []).concat(outs.images);
    if (outs.glbs.length) job.glbs = outs.glbs;

    // Hand the next stage the paths it needs, by the names stage 1 wrote.
    for (const g of outs.glbs) {
      if (/-stage1_/.test(g.filename)) carried.stage1Glb = rel(g);
      else if (/-dense_/.test(g.filename)) carried.denseGlb = rel(g);
    }
    for (const im of outs.images) {
      const base = im.filename.split(/[\\/]/).pop();
      if (/^coverage_/.test(base)) carried.coverage = relOut(im);
      // View names are alphabetic. \w+ would be greedy across the counter
      // ComfyUI appends and capture "right_00001" out of "repaint-right_00001_.png",
      // which then matches no view and drops the repaint silently.
      const m = /^repaint-([A-Za-z]+)_/.exec(base);
      if (m) {
        carried.repaints = carried.repaints || {};
        carried.repaints[m[1]] = relOut(im);
      }
    }

    // Check the handoff here rather than letting the next stage fail on a
    // missing input. A stage that ran fine but wrote nothing the next one can
    // use is far clearer reported now than three nodes deep in the next graph --
    // and the failure mode that matters is silent: a repaint the assembler
    // cannot find is simply skipped, and the run "succeeds" with the old texture.
    const listed = () => (outs.glbs.concat(outs.images).map((f) => f.filename).join(', ') || 'none');
    const fail = (msg) => { job.error = 'Stage "' + stage.name + '" ' + msg
                                        + ' Outputs were: ' + listed(); job.done = true; };

    if (stage.name === 'geometry') {
      if (!carried.stage1Glb) return fail('wrote no stage-1 mesh.');
      if (!carried.coverage) return fail('wrote no coverage map.');
    }
    if (stage.name === 'repaint') {
      // Named angles, not just "something was produced": matching the wrong key
      // is exactly how a repaint goes missing without anything looking wrong.
      const want = missingAngles(params).map((a) => a.name);
      const got = Object.keys(carried.repaints || {});
      const absent = want.filter((n) => !got.includes(n));
      if (absent.length) {
        return fail('produced no repaint for: ' + absent.join(', ')
                    + (got.length ? ' (found only: ' + got.join(', ') + ').' : '.'));
      }
    }
  }
  job.done = true;
  job.stage = 'done';
}

async function handleGenerate(req, res) {
  const raw = JSON.parse((await readBody(req)).toString('utf8'));
  const params = normaliseParams(raw);
  if (!params.images.front) return sendJSON(res, 400, { error: 'A front image is required.' });

  let stages;
  try {
    stages = buildStages(params);
    // Build the first stage now, so a graph error is reported synchronously
    // rather than surfacing minutes later as a mysterious job failure.
    stages[0].build({});
  } catch (e) {
    return sendJSON(res, 400, { error: 'Could not build workflow: ' + e.message });
  }

  const clientId = String(raw.clientId || 'comfy3d');

  // One stage: keep the original synchronous contract, so the panel's existing
  // progress path is untouched for runs that do not need splitting.
  if (stages.length === 1) {
    const built = stages[0].build({});
    const sub = await submitPrompt(built.prompt, clientId);
    if (sub.status !== 200) {
      return sendJSON(res, sub.status, { error: 'ComfyUI rejected the prompt', detail: sub.json });
    }
    const titles = {};
    for (const [id, node] of Object.entries(built.prompt)) {
      titles[id] = (node._meta && node._meta.title) || node.class_type;
    }
    return sendJSON(res, 200, {
      prompt_id: sub.json.prompt_id,
      number: sub.json.number,
      saveNode: built.saveNode,
      infoNode: built.infoNode,
      titles: titles,
      nodeCount: Object.keys(built.prompt).length,
      stages: 1,
      params: params,
    });
  }

  const jobId = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  const job = { id: jobId, stage: stages[0].name, stageIndex: 0, stageCount: stages.length,
                done: false, error: null, images: [], glbs: [] };
  jobs.set(jobId, job);
  // Prune anything old enough that the panel has long since stopped polling.
  for (const [k, v] of jobs) if (v.done && Date.now() - Number(v.finishedAt || 0) > 3600000) jobs.delete(k);

  runStages(job, stages, params, clientId)
    .catch((e) => { job.error = e.message; job.done = true; })
    .finally(() => { job.finishedAt = Date.now(); });

  sendJSON(res, 200, {
    jobId: jobId,
    stages: stages.length,
    stageNames: stages.map((s) => s.name),
    params: params,
  });
}

/** Image-to-image only, for tuning the generator without the 3D pipeline. */
async function handleGenerateImage(req, res) {
  const raw = JSON.parse((await readBody(req)).toString('utf8'));
  const params = normaliseParams(raw);
  if (!params.images.front) return sendJSON(res, 400, { error: 'A source image is required.' });

  let built;
  try {
    built = buildImagePrompt(params);
  } catch (e) {
    return sendJSON(res, 400, { error: 'Could not build workflow: ' + e.message });
  }

  const payload = Buffer.from(JSON.stringify({
    prompt: built.prompt, client_id: String(raw.clientId || 'comfy3d'),
  }));
  const r = await comfyRequest('POST', '/prompt', payload, {
    'Content-Type': 'application/json', 'Content-Length': payload.length,
  });

  let parsed;
  try { parsed = JSON.parse(r.body.toString('utf8')); } catch (e) { parsed = { raw: r.body.toString('utf8') }; }
  if (r.status !== 200) {
    return sendJSON(res, r.status, { error: 'ComfyUI rejected the prompt', detail: parsed });
  }

  const titles = {};
  for (const [id, node] of Object.entries(built.prompt)) {
    titles[id] = (node._meta && node._meta.title) || node.class_type;
  }
  sendJSON(res, 200, {
    prompt_id: parsed.prompt_id,
    titles: titles,
    nodeCount: Object.keys(built.prompt).length,
    params: params,
  });
}

/**
 * The newest completed run's outputs, so the panel can show the last result and
 * its pipeline steps without making you generate again.
 */
async function handleLastRun(res) {
  let r;
  try {
    r = await comfyRequest('GET', '/history?max_items=30', null, {});
  } catch (e) {
    return sendJSON(res, 200, { offline: true, images: [], glb: null });
  }
  if (r.status !== 200) return sendJSON(res, 200, { images: [], glb: null });

  let hist;
  try { hist = JSON.parse(r.body.toString('utf8')); } catch (e) { hist = {}; }

  // History is ordered oldest-first, so walk backwards for the newest success.
  // Only runs that produced a model count: the image lab writes images too, and
  // without this its output would be restored as if it were pipeline steps.
  const ids = Object.keys(hist);
  for (let i = ids.length - 1; i >= 0; i--) {
    const entry = hist[ids[i]];
    if (!entry || !(entry.status || {}).completed) continue;
    const images = [];
    const glbs = [];
    for (const out of Object.values(entry.outputs || {})) {
      for (const arr of Object.values(out)) {
        if (!Array.isArray(arr)) continue;
        for (const f of arr) {
          if (!f || typeof f.filename !== 'string') continue;
          if (/\.(glb|gltf)$/i.test(f.filename)) glbs.push(f);
          else if (/\.(png|jpe?g|webp)$/i.test(f.filename)) images.push(f);
        }
      }
    }
    if (glbs.length) {
      // A rigged run writes two models; the skinned one is the better default to
      // show, and it is the only one an animation can be applied to.
      const rigged = glbs.find((f) => /rigged/i.test(f.filename)) || null;
      // The unrigged mesh is reported separately: it is what the Mixamo export
      // sends (their rigger wants a bare mesh) and what carries the atlas that
      // gets reattached afterwards. The staged intermediates are not candidates.
      const plain = glbs.find(
        (f) => !/rigged|-stage1_|-dense_/i.test(f.filename)) || null;
      return sendJSON(res, 200, {
        promptId: ids[i], images: images,
        glb: rigged || plain || glbs[glbs.length - 1],
        rigged: rigged, plain: plain,
      });
    }
  }
  sendJSON(res, 200, { images: [], glb: null });
}

/** Reports which required model files ComfyUI can actually see right now. */
async function handlePreflight(res) {
  let r;
  try {
    r = await comfyRequest('GET', '/object_info', null, {});
  } catch (e) {
    // A timeout means the port is bound but not serving yet, which is what a
    // starting ComfyUI looks like; a refused connection means it is not there.
    const starting = /did not respond/.test(e.message || '');
    return sendJSON(res, 200, {
      error: starting
        ? 'ComfyUI is starting up on port ' + COMFY_PORT + ' — retrying automatically'
        : 'ComfyUI not running (checked ports ' + candidatePorts().join(', ') + ')',
      offline: true,
      starting: starting,
    });
  }
  if (r.status !== 200) return sendJSON(res, 200, { error: 'ComfyUI unreachable', offline: true });
  const info = JSON.parse(r.body.toString('utf8'));

  // A combo's choices appear in one of three shapes depending on the node:
  //   ["a","b"]                    - legacy inline list
  //   [{options:[...]}, ...]       - typed object in slot 0
  //   ["COMBO", {options:[...]}]   - v3 schema, options in slot 1
  const optionsFor = (node, input) => {
    try {
      const req = (info[node].input || {}).required || {};
      const opt = (info[node].input || {}).optional || {};
      const spec = req[input] || opt[input];
      if (!spec) return [];
      const t = spec[0];
      if (Array.isArray(t)) return t;
      if (t && Array.isArray(t.options)) return t.options;
      if (spec[1] && Array.isArray(spec[1].options)) return spec[1].options;
    } catch (e) { /* node missing */ }
    return [];
  };

  const checks = [
    { key: 'hy3d_mv', file: MODELS.CKPT_MV, have: optionsFor('ImageOnlyCheckpointLoader', 'ckpt_name') },
    { key: 'hy3d_21', file: MODELS.CKPT_21, have: optionsFor('ImageOnlyCheckpointLoader', 'ckpt_name') },
    { key: 't2_unet', file: MODELS.T2_UNET, have: optionsFor('UNETLoader', 'unet_name') },
    { key: 't2_shape_vae', file: MODELS.T2_SHAPE_VAE, have: optionsFor('VAELoader', 'vae_name') },
    { key: 't2_tex_vae', file: MODELS.T2_TEX_VAE, have: optionsFor('VAELoader', 'vae_name') },
    { key: 't2_clipvision', file: MODELS.T2_CLIPVISION, have: optionsFor('CLIPVisionLoader', 'clip_name') },
    { key: 'bg', file: MODELS.BG_MODEL, have: optionsFor('LoadBackgroundRemovalModel', 'bg_removal_name') },
  ];

  const present = {};
  for (const c of checks) present[c.key] = { file: c.file, ok: c.have.includes(c.file) };

  // The projection nodes only appear after ComfyUI is restarted, so report them
  // separately rather than lumping them in with missing model files.
  const projectionNodes = ['ProjectionViewAdd', 'MultiviewProjectionBake', 'ProjectionViewPreview'];
  const missingNodes = projectionNodes.filter((n) => !info[n]);

  const rigNodes = ['AutoRigHumanoid', 'PreviewRig', 'SaveRiggedGLB'];
  const missingRigNodes = rigNodes.filter((n) => !info[n]);

  // Checkpoints usable for the view-synthesis stage: everything except the two
  // 3D shape models, which are loaded through their own nodes.
  const allCkpts = optionsFor('CheckpointLoaderSimple', 'ckpt_name');
  const shape3d = [MODELS.CKPT_MV, MODELS.CKPT_21];
  const imageModels = allCkpts.filter((c) => !shape3d.includes(c));

  sendJSON(res, 200, {
    models: present,
    projection: { ok: missingNodes.length === 0, missing: missingNodes },
    rigging: { ok: missingRigNodes.length === 0, missing: missingRigNodes },
    // Only a union ControlNet is offered: the pipeline wants depth and normals
    // from one adapter, and loading two on a 10 GB card competes with the shape
    // model for memory.
    controlNets: optionsFor('ControlNetLoader', 'control_net_name'),
    imageModels: imageModels,
    comfyPort: COMFY_PORT,
  });
}

// ------------------------------------------------------------ animation -----
/**
 * Retarget a Mixamo clip onto a rigged GLB and hand back a URL for the viewer.
 *
 * The model is pulled from ComfyUI through the same /view route the panel
 * already uses, rather than reaching into ComfyUI's output directory: Comfy
 * Desktop relocates that directory, and the API is the one path that is always
 * correct about where a file actually is.
 */
async function handleAnimate(req, res) {
  const body = JSON.parse((await readBody(req, 1024 * 1024)).toString('utf8'));
  if (!body.filename || !body.fbx) {
    return sendJSON(res, 400, { error: 'filename and fbx are required' });
  }
  const q = new URLSearchParams({
    filename: body.filename,
    subfolder: body.subfolder || '',
    type: body.type || 'output',
  });
  const got = await comfyRequest('GET', '/view?' + q.toString(), null, {});
  if (got.status !== 200 || !got.body.length) {
    return sendJSON(res, 502, { error: 'Could not fetch the model from ComfyUI' });
  }
  if (got.body.slice(0, 4).toString('binary') !== 'glTF') {
    return sendJSON(res, 400, { error: 'That output is not a GLB' });
  }

  try {
    const out = await animate.retarget(got.body, body.fbx, { fps: Number(body.fps) || 30 });
    console.log('retargeted ' + body.fbx + ' -> ' + out.id
                + ' (' + out.matched + '/' + out.total + ' bones, ' + out.frames + ' frames)');
    sendJSON(res, 200, {
      url: '/api/animated/' + out.id + '.glb',
      matched: out.matched, total: out.total, frames: out.frames,
    });
  } catch (e) {
    sendJSON(res, 500, { error: e.message });
  }
}

/** Fetch one of ComfyUI's output files as a Buffer. */
async function fetchOutput(q) {
  const params = new URLSearchParams({
    filename: q.filename, subfolder: q.subfolder || '', type: q.type || 'output',
  });
  const r = await comfyRequest('GET', '/view?' + params.toString(), null, {});
  if (r.status !== 200 || !r.body.length) return null;
  return r.body;
}

/**
 * GLB -> FBX for Mixamo's auto-rigger.
 *
 * Mixamo reads FBX, not glTF, so the pipeline's output needs converting before
 * it can be rigged there. This is the outbound half of the hybrid flow; the
 * return half is /api/apply-textures.
 */
async function handleExportFbx(req, res) {
  const body = JSON.parse((await readBody(req, 1024 * 1024)).toString('utf8'));
  if (!body.filename) return sendJSON(res, 400, { error: 'filename is required' });
  const glb = await fetchOutput(body);
  if (!glb) return sendJSON(res, 502, { error: 'Could not fetch the model from ComfyUI' });
  if (glb.slice(0, 4).toString('binary') !== 'glTF') {
    return sendJSON(res, 400, { error: 'That output is not a GLB' });
  }
  try {
    const out = await animate.exportFbx(glb, { embed: body.embed !== false });
    const warn = out.log.split(/\r?\n/).filter((l) => /WARNING/i.test(l));
    console.log('exported FBX ' + out.id + (warn.length ? ' (' + warn.length + ' warnings)' : ''));
    sendJSON(res, 200, {
      url: '/api/artifact/' + out.id + '-mixamo.fbx',
      name: body.filename.replace(/\.glb$/i, '') + '.fbx',
      warnings: warn,
    });
  } catch (e) {
    sendJSON(res, 500, { error: e.message });
  }
}

/**
 * Rigged FBX from Mixamo + the original textured GLB -> textured rigged GLB.
 *
 * The FBX arrives as the raw request body; the original is named in the query,
 * since it is already one of ComfyUI's outputs.
 */
async function handleApplyTextures(req, res, url) {
  const fbx = await readBody(req, 256 * 1024 * 1024);
  if (!fbx.length) return sendJSON(res, 400, { error: 'No FBX uploaded' });
  const q = {
    filename: url.searchParams.get('filename'),
    subfolder: url.searchParams.get('subfolder') || '',
    type: url.searchParams.get('type') || 'output',
  };
  if (!q.filename) return sendJSON(res, 400, { error: 'filename of the original GLB is required' });
  const glb = await fetchOutput(q);
  if (!glb) return sendJSON(res, 502, { error: 'Could not fetch the original model' });

  try {
    const out = await animate.applyTextures(fbx, glb, { alsoFbx: true });
    // The script reports whether Mixamo preserved the UV layout. If it ever
    // stops doing so, a plain material assignment is no longer exact and this
    // is the line that says so.
    const report = out.log.split(/\r?\n/)
      .filter((l) => /uv bounds|uv bound drift|vertex count|texture:|rigged:|WARNING|assigned|dropped|applied the/i.test(l));
    console.log('applied textures to rigged mesh ' + out.id);
    sendJSON(res, 200, {
      url: '/api/artifact/' + out.id + '-textured.glb',
      fbxUrl: '/api/artifact/' + out.id + '-textured.fbx',
      zipUrl: '/api/artifact/' + out.id + '-textured.zip',
      report: report,
    });
  } catch (e) {
    sendJSON(res, 500, { error: e.message });
  }
}

// -------------------------------------------------------------- server ------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const p = url.pathname;

  try {
    if (p === '/api/upload' && req.method === 'POST') return await handleUpload(req, res, url);
    if (p === '/api/generate' && req.method === 'POST') return await handleGenerate(req, res);
    if (p === '/api/generate-image' && req.method === 'POST') return await handleGenerateImage(req, res);
    if (p === '/api/preflight') return await handlePreflight(res);
    if (p === '/api/last-run') return await handleLastRun(res);

    // Progress for a multi-stage run. The panel polls this and follows the
    // current stage's prompt through the normal websocket.
    if (p.startsWith('/api/job/')) {
      const job = jobs.get(p.split('/').pop());
      if (!job) return sendJSON(res, 404, { error: 'No such job' });
      return sendJSON(res, 200, job);
    }

    if (p === '/api/animations') {
      return sendJSON(res, 200, {
        blender: !!animate.findBlender(),
        animations: animate.listAnimations(),
        dir: animate.ANIM_DIR,
      });
    }
    if (p === '/api/animate' && req.method === 'POST') return await handleAnimate(req, res);
    if (p === '/api/export-fbx' && req.method === 'POST') return await handleExportFbx(req, res);
    if (p === '/api/apply-textures' && req.method === 'POST') {
      return await handleApplyTextures(req, res, url);
    }

    // Files Blender produced, served out of the cache.
    if (p.startsWith('/api/animated/') || p.startsWith('/api/artifact/')) {
      const name = p.split('/').pop();
      const m = /^([a-z0-9]+)(-[a-z]+)?\.(glb|fbx|zip)$/i.exec(name);
      if (!m) { res.writeHead(400); return res.end('Bad artifact name'); }
      const file = animate.cachedPath(m[1], (m[2] || '') + '.' + m[3]);
      if (!file) { res.writeHead(404); return res.end('Not found'); }
      const data = await fs.promises.readFile(file);
      res.writeHead(200, {
        'Content-Type': { glb: 'model/gltf-binary', fbx: 'application/octet-stream',
                          zip: 'application/zip' }[m[3].toLowerCase()]
                        || 'application/octet-stream',
        'Content-Length': data.length,
        'Content-Disposition': 'attachment; filename="' + name + '"',
      });
      return res.end(data);
    }

    if (p === '/api/interrupt' && req.method === 'POST') {
      const r = await comfyRequest('POST', '/interrupt', Buffer.from(''), { 'Content-Length': 0 });
      return sendJSON(res, 200, { ok: r.status === 200 });
    }

    // Transparent proxies for history and file fetches.
    if (p.startsWith('/api/history/')) {
      const r = await comfyRequest('GET', '/history/' + encodeURIComponent(p.split('/').pop()), null, {});
      res.writeHead(r.status, { 'Content-Type': 'application/json' });
      return res.end(r.body);
    }
    if (p === '/api/view') {
      const r = await comfyRequest('GET', '/view?' + url.searchParams.toString(), null, {});
      res.writeHead(r.status, {
        'Content-Type': r.headers['content-type'] || 'application/octet-stream',
        'Content-Length': r.body.length,
      });
      return res.end(r.body);
    }

    // Static files.
    let file = p === '/' ? '/index.html' : p;
    const full = path.join(PUBLIC, path.normalize(file).replace(/^(\.\.[/\\])+/, ''));
    if (!full.startsWith(PUBLIC)) { res.writeHead(403); return res.end('Forbidden'); }
    const data = await fs.promises.readFile(full).catch(() => null);
    if (!data) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(full)] || 'application/octet-stream' });
    res.end(data);
  } catch (e) {
    sendJSON(res, 500, { error: e.message });
  }
});

/**
 * Raw TCP tunnel for ComfyUI's /ws progress socket. Piping the upgrade through
 * untouched avoids needing a websocket library, and keeps the browser talking to
 * a single origin.
 */
server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url, 'http://localhost');
  const target = '/ws?' + url.searchParams.toString();
  const upstream = net.connect(COMFY_PORT, COMFY_HOST, () => {
    const upstreamOrigin = 'http://' + COMFY_HOST + ':' + COMFY_PORT;
    const lines = ['GET ' + target + ' HTTP/1.1', 'Host: ' + COMFY_HOST + ':' + COMFY_PORT];
    for (const [k, v] of Object.entries(req.headers)) {
      const lk = k.toLowerCase();
      // ComfyUI 403s any upgrade whose Origin is not its own, so present as same-origin.
      if (lk === 'host' || lk === 'origin' || lk === 'referer') continue;
      // Skip compression negotiation: this is a raw byte pipe, not a frame parser.
      if (lk === 'sec-websocket-extensions') continue;
      lines.push(k + ': ' + v);
    }
    lines.push('Origin: ' + upstreamOrigin);
    upstream.write(lines.join('\r\n') + '\r\n\r\n');
    if (head && head.length) upstream.write(head);
    upstream.pipe(socket);
    socket.pipe(upstream);
  });
  const bail = () => { try { socket.destroy(); } catch (e) {} try { upstream.destroy(); } catch (e) {} };
  upstream.on('error', () => {
    // The browser reconnects on close, so re-find ComfyUI now and the retry lands.
    discoverComfyPort().catch(() => {});
    bail();
  });
  socket.on('error', bail);
});

const PORT_EXPLICIT = argv.includes('--port');

async function announce(port) {
  console.log('image -> 3D control panel  http://127.0.0.1:' + port);
  const found = await discoverComfyPort();
  console.log(found
    ? 'proxying ComfyUI at        http://' + COMFY_HOST + ':' + COMFY_PORT
    : 'ComfyUI not found on ports ' + candidatePorts().join(', ') + ' — will keep looking');
}

/**
 * A stale copy of the panel holding the port used to crash this with an
 * unhandled 'error' event and a stack trace. Report it in a sentence instead,
 * and when the port was not asked for explicitly, just take the next free one.
 */
let listenPort = PORT;
let portAttempts = 0;

server.on('error', (e) => {
  if (e.code !== 'EADDRINUSE') throw e;

  if (PORT_EXPLICIT || portAttempts >= 10) {
    console.error('Port ' + listenPort + ' is already in use — most likely another ' +
                  'copy of this panel is still running.');
    console.error('Close it, or start on a different port:  node server.js --port ' +
                  (listenPort + 1));
    process.exit(1);
  }
  portAttempts += 1;
  listenPort = PORT + portAttempts;
  console.log('Port ' + (listenPort - 1) + ' busy, trying ' + listenPort + '…');
  server.listen(listenPort, '127.0.0.1');
});

server.listen(PORT, '127.0.0.1');
server.on('listening', () => { announce(server.address().port); });
