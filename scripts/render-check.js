'use strict';
/**
 * Verification helper: rebuilds a run's graph, appends turntable RenderMesh +
 * SaveImage nodes on the final mesh, and submits it. Every upstream node hits
 * ComfyUI's cache, so this costs only the renders.
 *
 *   node scripts/render-check.js <params.json> [--yaws 0,90,180,270] [--mode texture]
 */
const fs = require('fs');
const http = require('http');
const { buildPrompt } = require('../lib/workflow');

const file = process.argv[2];
const arg = (f, d) => { const i = process.argv.indexOf(f); return i > -1 ? process.argv[i + 1] : d; };
const yaws = arg('--yaws', '0,90,180,270').split(',').map(Number);
const mode = arg('--mode', 'texture');
const port = Number(arg('--port', '8189'));
// ComfyUI Desktop does not always come back on the same port after a restart.
const comfyPort = Number(arg('--comfy-port', process.env.COMFY_PORT || '8188'));

const req = (method, path, body) => new Promise((resolve, reject) => {
  const r = http.request({ host: '127.0.0.1', port, path, method,
    headers: body ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } : {} },
    (res) => { let s = ''; res.on('data', (d) => s += d); res.on('end', () => { try { resolve(JSON.parse(s)); } catch (e) { resolve({ raw: s }); } }); });
  r.on('error', reject); if (body) r.write(body); r.end();
});

// Normalise through the exact same code path the server uses. Keeping a private
// copy of the defaults here meant the graph differed from the panel's by a few
// unset fields, which silently missed ComfyUI's execution cache and re-ran the
// entire pipeline — including the image model — instead of just the renders.
const { normaliseParams } = require('../lib/params');

(async () => {
  const p = normaliseParams(JSON.parse(fs.readFileSync(file, 'utf8')));
  const built = buildPrompt(p);
  const g = built.prompt;

  // The mesh feeding SaveGLB is the finished, textured mesh.
  const meshLink = g[built.saveNode].inputs.mesh;

  let next = Math.max(...Object.keys(g).map(Number)) + 1;
  for (const yaw of yaws) {
    const cam = String(next++);
    g[cam] = { class_type: 'CreateCameraInfo', inputs: {
      mode: 'orbit', 'mode.yaw': yaw, 'mode.pitch': 12, 'mode.distance': 3.2,
      target_x: 0, target_y: 0, target_z: 0, roll: 0, fov: 35, zoom: 1,
      camera_type: 'perspective',
    } };
    const rend = String(next++);
    g[rend] = { class_type: 'RenderMesh', inputs: {
      mesh: meshLink, mode: mode, width: 640, height: 640,
      background: '#202430', camera_info: [cam, 0],
    } };
    g[String(next++)] = { class_type: 'SaveImage', inputs: {
      images: [rend, 0], filename_prefix: 'rendercheck/' + p.filenamePrefix.split('/').pop() + '_' + mode + '_yaw' + yaw,
    } };
  }

  const r = await new Promise((resolve, reject) => {
    const body = JSON.stringify({ prompt: g, client_id: 'render-check' });
    const rq = http.request({ host: '127.0.0.1', port: comfyPort, path: '/prompt', method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } },
      (res) => { let s = ''; res.on('data', (d) => s += d); res.on('end', () => resolve({ status: res.statusCode, body: s })); });
    rq.on('error', reject); rq.write(body); rq.end();
  });

  if (r.status !== 200) { console.error('rejected:', r.body.slice(0, 2000)); process.exit(1); }
  const id = JSON.parse(r.body).prompt_id;
  console.log('queued', id, '·', Object.keys(g).length, 'nodes');

  for (let i = 0; i < 240; i++) {
    await new Promise((res) => setTimeout(res, 5000));
    const h = await req('GET', '/api/history/' + id);
    const e = h[id];
    if (!e) { process.stdout.write('.'); continue; }
    if ((e.status || {}).status_str === 'error') {
      const err = (e.status.messages || []).find((m) => m[0] === 'execution_error');
      console.log('\nERROR', err ? err[1].node_type + ': ' + err[1].exception_message : '');
      process.exit(2);
    }
    if ((e.status || {}).completed) {
      const files = [];
      for (const out of Object.values(e.outputs || {}))
        for (const arr of Object.values(out))
          if (Array.isArray(arr)) for (const f of arr) if (f && f.filename) files.push(f);
      console.log('\nDONE');
      files.filter((f) => /\.png$/i.test(f.filename)).forEach((f) => console.log('  ' + f.subfolder + '/' + f.filename));
      process.exit(0);
    }
    process.stdout.write('.');
  }
  console.log('\nTIMEOUT'); process.exit(3);
})();
