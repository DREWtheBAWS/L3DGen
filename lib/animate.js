'use strict';
/**
 * Retarget a Mixamo FBX clip onto a rigged GLB, by driving Blender headlessly.
 *
 * Blender rather than a hand-rolled FBX parser: FBX is a closed binary format
 * whose animation data lives behind a graph of curve nodes, and a partial reader
 * would be a large amount of code to maintain for something Blender already does
 * correctly. It is also the tool most likely to already be installed on a
 * machine doing 3D work.
 *
 * The actual retargeting maths lives in scripts/retarget-mixamo.py.
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFile } = require('child_process');

const ROOT = path.join(__dirname, '..');
const SCRIPTS = path.join(ROOT, 'scripts');
const SCRIPT = path.join(SCRIPTS, 'retarget-mixamo.py');
const ANIM_DIR = path.join(ROOT, 'animations');
const CACHE_DIR = path.join(os.tmpdir(), 'image-to-3d-anim');

/** Where Blender usually is, newest first. BLENDER env var wins. */
function blenderCandidates() {
  const out = [];
  if (process.env.BLENDER) out.push(process.env.BLENDER);
  const roots = [
    'C:\\Program Files\\Blender Foundation',
    'C:\\Program Files (x86)\\Blender Foundation',
    'D:\\BlenderFoundation\\Blender',
    'D:\\Program Files\\Blender Foundation',
  ];
  for (const r of roots) {
    let entries = [];
    try { entries = fs.readdirSync(r); } catch (e) { continue; }
    // Newest version first, so a machine with several installs uses the latest.
    entries.sort().reverse();
    for (const e of entries) out.push(path.join(r, e, 'blender.exe'));
  }
  out.push('/usr/bin/blender', '/usr/local/bin/blender',
           '/Applications/Blender.app/Contents/MacOS/Blender');
  return out;
}

let cachedBlender;
function findBlender() {
  if (cachedBlender !== undefined) return cachedBlender;
  cachedBlender = blenderCandidates().find((p) => {
    try { return fs.statSync(p).isFile(); } catch (e) { return false; }
  }) || null;
  return cachedBlender;
}

/** Mixamo clips the panel can offer. Drop .fbx files into animations/. */
function listAnimations() {
  try {
    return fs.readdirSync(ANIM_DIR)
      .filter((f) => f.toLowerCase().endsWith('.fbx'))
      .sort()
      .map((f) => ({
        file: f,
        // "X Bot@Standing Idle (1).fbx" -> "Standing Idle". Mixamo puts the
        // character before the @ and a browser download counter after the name.
        label: f.replace(/\.fbx$/i, '').replace(/^.*@/, '').replace(/\s*\(\d+\)$/, ''),
      }));
  } catch (e) {
    return [];
  }
}

/**
 * Run the retarget. `glbBuffer` is the rigged model; `fbxName` is a file in
 * animations/. Resolves to the path of an animated GLB in the cache directory.
 */
function retarget(glbBuffer, fbxName, opts) {
  return new Promise((resolve, reject) => {
    const blender = findBlender();
    if (!blender) {
      return reject(new Error(
        'Blender was not found. Install it, or set the BLENDER environment '
        + 'variable to blender.exe. Retargeting needs it to read the FBX.'));
    }
    // Reject anything that is not a plain name inside animations/.
    if (!fbxName || /[\\/]/.test(fbxName) || !fbxName.toLowerCase().endsWith('.fbx')) {
      return reject(new Error('Invalid animation name'));
    }
    const fbx = path.join(ANIM_DIR, fbxName);
    if (!fs.existsSync(fbx)) return reject(new Error('No such animation: ' + fbxName));

    fs.mkdirSync(CACHE_DIR, { recursive: true });
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    const inPath = path.join(CACHE_DIR, id + '-in.glb');
    const outPath = path.join(CACHE_DIR, id + '.glb');
    fs.writeFileSync(inPath, glbBuffer);

    const args = ['--background', '--python', SCRIPT, '--',
                  inPath, fbx, outPath, '--fps', String((opts && opts.fps) || 30)];

    execFile(blender, args, { timeout: 300000, maxBuffer: 32 * 1024 * 1024 },
      (err, stdout, stderr) => {
        try { fs.unlinkSync(inPath); } catch (e) { /* best effort */ }
        const log = String(stdout || '') + String(stderr || '');
        if (!fs.existsSync(outPath)) {
          // Blender's own diagnostics are far more useful than the exit code,
          // so surface the lines the script printed rather than "exit 1".
          const lines = log.split(/\r?\n/)
            .filter((l) => /ERROR|usage:|not found|Traceback|matched|armature/i.test(l))
            .slice(-6).join('; ');
          return reject(new Error('Retarget failed. ' + (lines || (err && err.message) || 'no output')));
        }
        const matched = /matched (\d+) of (\d+)/.exec(log);
        const frames = /baked (\d+) frames/.exec(log);
        resolve({
          path: outPath,
          id: id,
          matched: matched ? Number(matched[1]) : null,
          total: matched ? Number(matched[2]) : null,
          frames: frames ? Number(frames[1]) : null,
        });
      });
  });
}

/** Run one of the scripts/ Blender jobs. Resolves to { log, out }. */
function runBlender(scriptName, buildArgs, outName) {
  return new Promise((resolve, reject) => {
    const blender = findBlender();
    if (!blender) {
      return reject(new Error(
        'Blender was not found. Install it, or set the BLENDER environment '
        + 'variable to blender.exe.'));
    }
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
    const outPath = path.join(CACHE_DIR, id + '-' + outName);
    let inputs = [];
    let args;
    try {
      const built = buildArgs(id, outPath);
      args = built.args;
      inputs = built.inputs || [];
    } catch (e) {
      return reject(e);
    }

    execFile(blender,
      ['--background', '--python', path.join(SCRIPTS, scriptName), '--'].concat(args),
      { timeout: 300000, maxBuffer: 32 * 1024 * 1024 },
      (err, stdout, stderr) => {
        for (const f of inputs) { try { fs.unlinkSync(f); } catch (e) { /* best effort */ } }
        const log = String(stdout || '') + String(stderr || '');
        if (!fs.existsSync(outPath)) {
          // Blender's own diagnostics beat an exit code every time.
          const lines = log.split(/\r?\n/)
            .filter((l) => /ERROR|WARNING|usage:|Traceback|not found/i.test(l))
            .slice(-6).join('; ');
          return reject(new Error(lines || (err && err.message) || 'Blender produced no output'));
        }
        resolve({ id: id, path: outPath, log: log });
      });
  });
}

/** GLB -> FBX, for uploading to Mixamo's auto-rigger. */
function exportFbx(glbBuffer, opts) {
  return runBlender('glb-to-fbx.py', (id, outPath) => {
    const inPath = path.join(CACHE_DIR, id + '-in.glb');
    fs.writeFileSync(inPath, glbBuffer);
    const args = [inPath, outPath];
    if (!opts || opts.embed !== false) args.push('--embed');
    return { args: args, inputs: [inPath] };
  }, 'mixamo.fbx');
}

/**
 * Rigged FBX from Mixamo + the original GLB -> textured, rigged model.
 *
 * Writes three things: a GLB for the viewer, an FBX with the texture embedded,
 * and a zip of the FBX beside its .fbm texture folder. The zip exists because
 * embedded textures depend on the importer choosing to extract them, while
 * loose textures in a sibling .fbm folder are resolved by everything.
 */
function applyTextures(fbxBuffer, glbBuffer, opts) {
  return runBlender('apply-textures.py', (id, outPath) => {
    const fbxPath = path.join(CACHE_DIR, id + '-rigged.fbx');
    const glbPath = path.join(CACHE_DIR, id + '-original.glb');
    fs.writeFileSync(fbxPath, fbxBuffer);
    fs.writeFileSync(glbPath, glbBuffer);
    const args = [fbxPath, glbPath, outPath, '--report'];
    if (!opts || opts.alsoFbx !== false) args.push('--fbx', outPath.replace(/\.glb$/, '.fbx'));
    if (!opts || opts.zip !== false) args.push('--zip', outPath.replace(/\.glb$/, '.zip'));
    return { args: args, inputs: [fbxPath, glbPath] };
  }, 'textured.glb');
}

function cachedPath(id, suffix) {
  if (!/^[a-z0-9]+$/i.test(id)) return null;
  const p = path.join(CACHE_DIR, id + (suffix || '.glb'));
  return fs.existsSync(p) ? p : null;
}

module.exports = {
  findBlender: findBlender,
  listAnimations: listAnimations,
  retarget: retarget,
  exportFbx: exportFbx,
  applyTextures: applyTextures,
  cachedPath: cachedPath,
  ANIM_DIR: ANIM_DIR,
  CACHE_DIR: CACHE_DIR,
};
