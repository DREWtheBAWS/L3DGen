'use strict';
/**
 * Check a built prompt against the node definitions before submitting it.
 *
 * ComfyUI only reports a bad input once the node is reached, which on this
 * pipeline is minutes into a run: the palette parameters were missing from the
 * live bake node and the failure surfaced 78 seconds in, after the geometry
 * stage had already been paid for. Validating up front turns that into an
 * instant, offline error.
 *
 *   node scripts/validate-graph.js [--port 8188]
 *
 * Reads /object_info from a running ComfyUI, so it also catches the case where
 * custom nodes on disk are newer than the ones the server has loaded.
 */
const http = require('http');
const { buildPrompt, buildStages } = require('../lib/workflow');
const { normaliseParams } = require('../lib/params');

const arg = (f, d) => { const i = process.argv.indexOf(f); return i > -1 ? process.argv[i + 1] : d; };
const port = Number(arg('--port', '8188'));

const get = (path) => new Promise((resolve, reject) => {
  const req = http.get({ host: '127.0.0.1', port, path, timeout: 60000 }, (res) => {
    let s = '';
    res.on('data', (d) => s += d);
    res.on('end', () => { try { resolve(JSON.parse(s)); } catch (e) { reject(e); } });
  });
  req.on('error', reject);
  req.on('timeout', () => req.destroy(new Error('timeout')));
});

const CASES = {
  'pig 3-view + rig + palette': {
    images: { front: 'front-pig.png', left: 'left-pig.png', back: 'back-pig.png' },
    geometry: 'hy3d_mv', texture: true, autoRig: true,
    paletteColors: 12, fillMode: 'surface', saveIntermediates: true, lowVram: true,
  },
  'front only, synthesis, no rig': {
    images: { front: 'front-pig.png' },
    geometry: 'hy3d_mv', texture: true, synthesizeViews: true, autoRig: false,
  },
  'trellis + rig': {
    images: { front: 'front-pig.png' },
    geometry: 'trellis2', texture: true, autoRig: true,
  },
  // Covers the depth/normal outputs of ProjectionViewRender, which is exactly
  // the kind of thing that only breaks after a custom node changes on disk and
  // ComfyUI has not been restarted.
  'depth-locked repaint (ControlNet)': {
    images: { front: 'front-pig.png', left: 'left-pig.png', back: 'back-pig.png' },
    geometry: 'hy3d_mv', texture: true, synthesizeViews: true, synthBackend: 'sdxl',
    controlNet: 'diffusion_pytorch_model_promax.safetensors', controlNetType: 'depth',
  },
  'normal-locked repaint (ControlNet)': {
    images: { front: 'front-pig.png' },
    geometry: 'hy3d_mv', texture: true, synthesizeViews: true, synthBackend: 'sdxl',
    controlNet: 'diffusion_pytorch_model_promax.safetensors', controlNetType: 'normal',
  },
};

(async () => {
  let info;
  try {
    info = await get('/object_info');
  } catch (e) {
    console.error('Could not reach ComfyUI on port ' + port + ': ' + e.message);
    process.exit(2);
  }

  let failures = 0;

  // Every graph the pipeline can emit, single-prompt and staged alike. Staged
  // runs are fed the filenames the previous stage would have written, so the
  // later graphs are exercised exactly as the server will build them.
  const graphs = [];
  for (const [label, raw] of Object.entries(CASES)) {
    graphs.push([label, buildPrompt(normaliseParams(raw)).prompt]);
  }
  {
    const p = normaliseParams({
      images: { front: 'front-pig.png', left: 'left-pig.png', back: 'back-pig.png' },
      geometry: 'hy3d_mv', texture: true, synthesizeViews: true, synthBackend: 'sdxl',
      controlNet: 'diffusion_pytorch_model_promax.safetensors',
      autoRig: true, saveIntermediates: true, useReferenceMesh: true,
    });
    const carried = {};
    for (const stage of buildStages(p)) {
      graphs.push(['staged: ' + stage.name, stage.build(carried).prompt]);
      if (stage.name === 'geometry') {
        carried.stage1Glb = '3d/pipeline-stage1_00001_.glb';
        carried.denseGlb = '3d/pipeline-dense_00001_.glb';
        carried.coverage = '3d/steps/coverage_00001_.png';
      }
      if (stage.name === 'repaint') {
        carried.repaints = { right: '3d/steps/repaint-right_00001_.png' };
      }
    }
  }

  for (const [label, prompt] of graphs) {
    const problems = [];

    for (const [id, node] of Object.entries(prompt)) {
      const def = info[node.class_type];
      if (!def) {
        problems.push(`node ${id}: class ${node.class_type} is not loaded in ComfyUI`);
        continue;
      }
      const req = (def.input && def.input.required) || {};
      const opt = (def.input && def.input.optional) || {};
      const known = new Set([...Object.keys(req), ...Object.keys(opt)]);

      for (const k of Object.keys(node.inputs)) {
        if (!known.has(k)) {
          problems.push(`node ${id} (${node.class_type}): unknown input "${k}"`);
        }
      }
      for (const k of Object.keys(req)) {
        if (!(k in node.inputs)) {
          problems.push(`node ${id} (${node.class_type}): missing required input "${k}"`);
        }
      }
      // Links must point at a node that exists and an output slot it really has.
      for (const [k, v] of Object.entries(node.inputs)) {
        if (!Array.isArray(v) || v.length !== 2 || typeof v[0] !== 'string') continue;
        const src = prompt[v[0]];
        if (!src) { problems.push(`node ${id}: input "${k}" points at missing node ${v[0]}`); continue; }
        const srcDef = info[src.class_type];
        const nOut = srcDef && srcDef.output ? srcDef.output.length : 0;
        if (srcDef && v[1] >= nOut) {
          problems.push(`node ${id}: input "${k}" reads slot ${v[1]} of ${src.class_type}, `
                        + `which has only ${nOut} output(s)`);
        }
      }
    }

    const n = Object.keys(prompt).length;
    if (problems.length) {
      failures += problems.length;
      console.log(`FAIL  ${label}  (${n} nodes)`);
      for (const p of problems) console.log('        ' + p);
    } else {
      console.log(`ok    ${label}  (${n} nodes)`);
    }
  }

  console.log(failures ? `\n${failures} problem(s) found` : '\nGraph matches the loaded node definitions');
  process.exit(failures ? 1 : 0);
})();
