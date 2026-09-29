'use strict';
/**
 * Headless driver for the pipeline - submits a params JSON and polls to
 * completion, so runs can be verified without a browser.
 *
 *   node scripts/run-test.js <params.json> [--port 8189] [--timeout 1800]
 */
const fs = require('fs');
const http = require('http');

const file = process.argv[2];
const port = Number((process.argv.indexOf('--port') > -1 && process.argv[process.argv.indexOf('--port') + 1]) || 8189);
const timeout = Number((process.argv.indexOf('--timeout') > -1 && process.argv[process.argv.indexOf('--timeout') + 1]) || 1800);

const req = (method, path, body) => new Promise((resolve, reject) => {
  const r = http.request({ host: '127.0.0.1', port, path, method,
    headers: body ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } : {} },
    (res) => {
      let s = '';
      res.on('data', (d) => s += d);
      res.on('end', () => { try { resolve({ status: res.statusCode, json: JSON.parse(s) }); } catch (e) { resolve({ status: res.statusCode, json: { raw: s } }); } });
    });
  r.on('error', reject);
  if (body) r.write(body);
  r.end();
});

(async () => {
  const params = JSON.parse(fs.readFileSync(file, 'utf8'));
  const sub = await req('POST', '/api/generate', JSON.stringify(params));
  if (sub.status !== 200) {
    console.error('REJECTED', JSON.stringify(sub.json, null, 1).slice(0, 4000));
    process.exit(1);
  }
  const id = sub.json.prompt_id;
  console.log('queued', id, '·', sub.json.nodeCount, 'nodes · seed', sub.json.params.seed,
    '· target', sub.json.params.targetTriangles, 'tris');

  const t0 = Date.now();
  let lastNode = '';
  while ((Date.now() - t0) / 1000 < timeout) {
    await new Promise((r) => setTimeout(r, 5000));
    const h = await req('GET', '/api/history/' + id);
    const entry = h.json[id];
    if (!entry) { process.stdout.write('.'); continue; }
    const st = entry.status || {};
    if (st.status_str === 'error') {
      const err = (st.messages || []).find((m) => m[0] === 'execution_error');
      console.log('\nERROR after', Math.round((Date.now() - t0) / 1000) + 's');
      if (err) {
        console.log('  node   :', err[1].node_id, err[1].node_type);
        console.log('  message:', err[1].exception_message);
        console.log('  inputs :', JSON.stringify(err[1].current_inputs).slice(0, 900));
      }
      process.exit(2);
    }
    if (st.completed) {
      console.log('\nDONE in', Math.round((Date.now() - t0) / 1000) + 's');
      const files = [];
      for (const out of Object.values(entry.outputs || {}))
        for (const arr of Object.values(out))
          if (Array.isArray(arr)) for (const f of arr) if (f && f.filename) files.push(f);
      console.log('outputs:', JSON.stringify(files));
      process.exit(0);
    }
    const running = (st.messages || []).filter((m) => m[0] === 'executing').pop();
    const label = running ? String(running[1].node) : '?';
    if (label !== lastNode) { process.stdout.write('\n  node ' + label + ' '); lastNode = label; }
    else process.stdout.write('.');
  }
  console.log('\nTIMEOUT');
  process.exit(3);
})();
