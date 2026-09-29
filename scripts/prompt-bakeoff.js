'use strict';
/**
 * Run several Kontext prompts over one source image and collect the results.
 *
 * Kontext does support camera/viewpoint edits, but the instruction wording
 * dominates the outcome: a prompt that opens by asking to preserve the image
 * pulls it toward reproducing the input rather than re-posing the subject.
 * This makes that comparison measurable instead of anecdotal.
 *
 *   node scripts/prompt-bakeoff.js <input-image-name> [--port 8189]
 */
const http = require('http');

const image = process.argv[2];
const arg = (f, d) => { const i = process.argv.indexOf(f); return i > -1 ? process.argv[i + 1] : d; };
const port = Number(arg('--port', '8189'));

const PROMPTS = [
  ['A_original',
   'Keep the exact art style, palette and proportions of the reference. Show the same subject from the left side.'],
  ['B_camera_move',
   'Rotate the camera 90 degrees to the left so the creature is seen in profile from its left side. Same creature, same colours, same low poly art style, plain background.'],
  ['C_turnaround',
   'Turn the creature 90 degrees to show its left side profile. Keep the identical character design, colour palette and flat low-poly shading. Full body, centred, plain grey background.'],
  ['D_back',
   'Rotate the camera to show the creature from directly behind. Same creature, same colours, same low poly art style, plain background.'],
];

const post = (path, body) => new Promise((resolve, reject) => {
  const data = JSON.stringify(body);
  const r = http.request({ host: '127.0.0.1', port, path, method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } },
    (res) => { let s = ''; res.on('data', (d) => s += d);
      res.on('end', () => { try { resolve({ status: res.statusCode, json: JSON.parse(s) }); }
                            catch (e) { resolve({ status: res.statusCode, json: { raw: s } }); } }); });
  r.on('error', reject); r.write(data); r.end();
});

const get = (path) => new Promise((resolve, reject) => {
  http.get({ host: '127.0.0.1', port, path }, (res) => {
    let s = ''; res.on('data', (d) => s += d);
    res.on('end', () => { try { resolve(JSON.parse(s)); } catch (e) { resolve({}); } });
  }).on('error', reject);
});

(async () => {
  for (const [label, prompt] of PROMPTS) {
    process.stdout.write(label.padEnd(14));
    const sub = await post('/api/generate-image', {
      images: { front: image },
      synthBackend: 'kontext',
      synthPrompt: prompt,
      synthDenoise: 1.0,
      synthSteps: 25,
      kontextGuidance: 2.5,
      seed: 12345,                       // fixed, so only the wording varies
    });
    if (sub.status !== 200) { console.log('REJECTED ' + JSON.stringify(sub.json).slice(0, 200)); continue; }
    const id = sub.json.prompt_id;

    let done = false;
    for (let i = 0; i < 180 && !done; i++) {
      await new Promise((r) => setTimeout(r, 5000));
      const hist = await get('/api/history/' + id);
      const e = hist[id];
      if (!e) { process.stdout.write('.'); continue; }
      const st = e.status || {};
      if (st.status_str === 'error') { console.log(' ERROR'); done = true; break; }
      if (!st.completed) { process.stdout.write('.'); continue; }
      let file = null;
      for (const o of Object.values(e.outputs || {}))
        for (const a of Object.values(o))
          if (Array.isArray(a)) for (const f of a) if (f && /\.png$/i.test(f.filename || '')) file = f;
      console.log(' -> ' + (file ? file.filename : 'no image'));
      done = true;
    }
    if (!done) console.log(' TIMEOUT');
  }
})();
