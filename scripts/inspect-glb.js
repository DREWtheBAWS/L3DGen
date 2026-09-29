'use strict';
/**
 * Reports triangle/vertex counts, UV presence and embedded textures for a GLB,
 * so a run can be verified against the requested polygon budget.
 *
 *   node scripts/inspect-glb.js <file.glb>
 */
const fs = require('fs');

const buf = fs.readFileSync(process.argv[2]);
if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error('not a GLB');

let off = 12, json = null, binLen = 0;
while (off < buf.length) {
  const len = buf.readUInt32LE(off);
  const type = buf.readUInt32LE(off + 4);
  if (type === 0x4e4f534a) json = JSON.parse(buf.slice(off + 8, off + 8 + len).toString('utf8'));
  if (type === 0x004e4942) binLen = len;
  off += 8 + len + ((4 - (len % 4)) % 4);
}

let tris = 0, verts = 0;
const attrs = new Set();
for (const mesh of json.meshes || []) {
  for (const prim of mesh.primitives || []) {
    const pos = json.accessors[prim.attributes.POSITION];
    verts += pos.count;
    tris += prim.indices !== undefined ? json.accessors[prim.indices].count / 3 : pos.count / 3;
    Object.keys(prim.attributes).forEach((a) => attrs.add(a));
  }
}

const maps = [];
for (const m of json.materials || []) {
  const pbr = m.pbrMetallicRoughness || {};
  if (pbr.baseColorTexture) maps.push('baseColor');
  if (pbr.metallicRoughnessTexture) maps.push('metallicRoughness');
  if (m.normalTexture) maps.push('normal');
  if (m.occlusionTexture) maps.push('occlusion');
}

console.log('file       :', process.argv[2].split(/[/\\]/).pop());
console.log('size       :', (buf.length / 1048576).toFixed(2), 'MB  (bin', (binLen / 1048576).toFixed(2), 'MB)');
console.log('triangles  :', Math.round(tris).toLocaleString());
console.log('vertices   :', verts.toLocaleString());
console.log('attributes :', [...attrs].join(', ') || '(none)');
console.log('materials  :', (json.materials || []).length, '· maps:', maps.join(', ') || '(none)');
console.log('images     :', (json.images || []).length);
