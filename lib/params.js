"use strict";
/**
 * Single source of truth for pipeline parameters.
 *
 * Both the HTTP server and the headless test scripts normalise through this, so
 * a preset run from the command line builds byte-identical graphs to the panel.
 * When these drifted apart, test runs silently missed ComfyUI's execution cache
 * and re-ran the whole pipeline.
 */

const { VIEWS } = require("./workflow");

// -------------------------------------------------------- param handling ----
const NUM = (v, dflt, lo, hi) => {
  const n = Number(v);
  if (!isFinite(n)) return dflt;
  return Math.min(hi, Math.max(lo, n));
};

function normaliseParams(raw) {
  const images = {};
  for (const v of VIEWS) if (raw.images && raw.images[v]) images[v] = String(raw.images[v]);

  const geometry = ['hy3d_mv', 'hy3d_21', 'trellis2'].includes(raw.geometry) ? raw.geometry : 'hy3d_mv';

  // Hunyuan 2.1 and 2.0-MV want different defaults; the UI sends explicit values
  // but we still clamp everything server-side.
  const p = {
    images: images,
    geometry: geometry,
    seed: raw.seed === -1 || raw.seed === undefined || raw.seed === null
      ? Math.floor(Math.random() * 0xffffffff)
      : NUM(raw.seed, 0, 0, 0xffffffffff),
    steps: NUM(raw.steps, geometry === 'hy3d_21' ? 30 : 20, 1, 100),
    cfg: NUM(raw.cfg, geometry === 'hy3d_21' ? 5.0 : 7.5, 0, 20),
    samplerName: String(raw.samplerName || 'euler'),
    scheduler: String(raw.scheduler || 'normal'),
    shift: NUM(raw.shift, 1.0, 0.01, 10),
    latentResolution: NUM(raw.latentResolution, geometry === 'hy3d_21' ? 4096 : 3072, 256, 8192),
    octreeResolution: NUM(raw.octreeResolution, 256, 16, 512),
    numChunks: NUM(raw.numChunks, 8000, 1000, 500000),
    surfaceThreshold: NUM(raw.surfaceThreshold, 0.6, -1, 1),

    // Voxel resolution of the TRELLIS upsample. The node itself refuses
    // anything below 1024, so the old 512 floor produced a graph ComfyUI would
    // reject. This is also the dominant VRAM term in the shape decode: the cost
    // goes with the cube of the resolution, and 1536 is roughly 3.4x the 1024
    // decode.
    trellisUpsample: Math.round(NUM(raw.trellisUpsample, 1024, 1024, 2048) / 128) * 128,

    removeBackground: raw.removeBackground !== false,
    padFactor: NUM(raw.padFactor, 1.1, 1.0, 2.0),
    hunyuanBackground: /^#[0-9a-fA-F]{6}$/.test(raw.hunyuanBackground || '') ? raw.hunyuanBackground : '#ffffff',

    // the low-poly budget
    targetTriangles: Math.round(NUM(raw.targetTriangles, 20000, 100, 2000000)),
    decimatePlacement: raw.decimatePlacement === 'qem' ? 'qem' : 'midpoint',
    // Only honoured for TRELLIS.2 geometry; workflow.js refuses it for Hunyuan.
    remesh: !!raw.remesh,
    remeshResolution: Math.round(NUM(raw.remeshResolution, 512, 64, 1024)),
    remeshSmooth: Math.round(NUM(raw.remeshSmooth, 0, 0, 20)),
    creaseAngle: NUM(raw.creaseAngle, 60, 0, 180),

    texture: raw.texture !== false,
    textureResolution: [512, 1024, 2048, 4096].includes(Number(raw.textureResolution))
      ? Number(raw.textureResolution) : 2048,
    bakeResolution: [512, 1024, 2048, 4096].includes(Number(raw.bakeResolution))
      ? Number(raw.bakeResolution) : 1024,
    bakeAO: !!raw.bakeAO,
    bakeNormal: !!raw.bakeNormal,
    aoSamples: Math.round(NUM(raw.aoSamples, 64, 4, 1024)),
    useReferenceMesh: raw.useReferenceMesh !== false,

    // Projection texturing (Hunyuan geometry): the reference art is projected
    // onto the mesh from each view's camera.
    facingPower: NUM(raw.facingPower, 3.0, 0.1, 16),
    projectionOcclusion: raw.projectionOcclusion !== false,
    minFacing: NUM(raw.minFacing, 0.2, 0.0, 0.9),
    maskErode: Math.round(NUM(raw.maskErode, 6, 0, 64)),
    fillMode: ['surface', 'surface+mirror', 'nearest', 'none'].includes(raw.fillMode)
      ? raw.fillMode : 'surface',
    // Correct approximated texels toward a palette taken from the art itself.
    // Blending makes gradients whose midpoint hues are in no reference image; 0
    // disables it. Shading is preserved by default, so this fixes colour without
    // flattening the model into cel shading.
    paletteColors: Math.round(NUM(raw.paletteColors, 0, 0, 64)),
    paletteStrength: NUM(raw.paletteStrength, 0.7, 0, 1),
    paletteKeepShading: raw.paletteKeepShading !== false,
    // Flip which yaw the "left"/"right" images map to, for art labelled by
    // camera position rather than by the character's own left and right.
    swapSides: !!raw.swapSides,

    // The subject is bilaterally symmetric. A side view can then be mirrored to
    // stand in for the opposite side exactly, and the gap fill is allowed to
    // borrow from the mirrored position on the model.
    symmetric: !!raw.symmetric,

    // Off-axis reference art (isometric / three-quarter): recover each image's
    // camera angle from the mesh silhouette instead of assuming a turnaround.
    fitAngles: !!raw.fitAngles,
    fitYawSteps: Math.round(NUM(raw.fitYawSteps, 24, 4, 72)),
    // Half-width of the yaw search. Narrow by default: silhouettes rarely
    // identify a pose on their own.
    fitYawRange: NUM(raw.fitYawRange, 30, 5, 180),
    fitPitchMin: NUM(raw.fitPitchMin, -60, -89, 0),
    fitPitchMax: NUM(raw.fitPitchMax, 60, 0, 89),
    fitPitchSteps: Math.round(NUM(raw.fitPitchSteps, 13, 1, 41)),
    viewPitch: NUM(raw.viewPitch, 0, -89, 89),
    viewScale: NUM(raw.viewScale, 1.0, 0.5, 2.0),
    saveCoverage: !!raw.saveCoverage,

    // View synthesis: render the missing angle off the part-textured mesh, then
    // let an image model clean it up in the same style.
    // Invent the turnaround from one image before any 3D runs, so the generated
    // views condition the shape model too - not just the texture.
    generateReferenceViews: !!raw.generateReferenceViews,
    refViewStyle: String(raw.refViewStyle || 'Same subject, same colours, same low poly art style, full body, centred, plain background.'),

    synthesizeViews: !!raw.synthesizeViews,

    // 'sdxl'    - img2img over the render; fast, but never sees the source art.
    // 'kontext' - Flux Kontext, with the source art as a second reference latent,
    //             which is what makes generated views match the original style.
    synthBackend: raw.synthBackend === 'kontext' ? 'kontext' : 'sdxl',
    kontextUnet: String(raw.kontextUnet || 'flux1-dev-kontext_fp8_scaled.safetensors'),
    kontextT5: String(raw.kontextT5 || 't5xxl_fp8_e4m3fn_scaled.safetensors'),
    kontextClipL: String(raw.kontextClipL || 'clip_l.safetensors'),
    kontextVae: String(raw.kontextVae || 'ae.safetensors'),
    kontextGuidance: NUM(raw.kontextGuidance, 2.5, 1.0, 10.0),
    kontextRefMethod: ['offset', 'index', 'uxo/uno', 'index_timestep_zero']
      .includes(raw.kontextRefMethod) ? raw.kontextRefMethod : 'offset',

    // Structural conditioning for the SDXL synthesis pass. Empty = off.
    // With a ControlNet locking the generation to the mesh's own depth or
    // normals, denoise can go high enough to invent surface detail on geometry
    // no reference ever showed, instead of only re-tinting the blend.
    controlNet: String(raw.controlNet || ''),
    controlNetType: ['depth', 'normal'].includes(raw.controlNetType) ? raw.controlNetType : 'depth',
    controlNetStrength: NUM(raw.controlNetStrength, 0.85, 0, 2),
    // Releasing the constraint before the end lets the last steps add texture
    // detail that a hard structural lock would otherwise flatten.
    controlNetEnd: NUM(raw.controlNetEnd, 0.85, 0.1, 1.0),
    controlNetDenoise: NUM(raw.controlNetDenoise, 0.85, 0.1, 1.0),

    imageModel: String(raw.imageModel || 'sd_xl_base_1.0.safetensors'),
    synthPrompt: String(raw.synthPrompt || ''),
    synthNegative: String(raw.synthNegative || ''),
    synthSteps: Math.round(NUM(raw.synthSteps, 25, 1, 80)),
    synthCfg: NUM(raw.synthCfg, 6.0, 1, 20),
    // Clamped per backend below: img2img at denoise 1.0 discards the init image
    // entirely, which is never what this pipeline wants from SDXL.
    synthDenoise: NUM(raw.synthDenoise, 0.55, 0.05, 1.0),
    synthResolution: [512, 768, 1024].includes(Number(raw.synthResolution))
      ? Number(raw.synthResolution) : 1024,
    synthBackground: /^#[0-9a-fA-F]{6}$/.test(raw.synthBackground || '') ? raw.synthBackground : '#7f7f7f',
    synthGapsOnly: raw.synthGapsOnly !== false,
    gapGrow: Math.round(NUM(raw.gapGrow, 6, 0, 96)),
    syntheticWeight: NUM(raw.syntheticWeight, 0.6, 0.05, 2.0),
    refineGeometry: !!raw.refineGeometry,

    // Automatic humanoid rigging: fit the Mixamo joint hierarchy to the finished
    // mesh and write a second, skinned GLB alongside the static one.
    autoRig: !!raw.autoRig,
    rigFacing: raw.rigFacing === '-Z' ? '-Z' : '+Z',
    rigSymmetrize: raw.rigSymmetrize !== false,
    rigSmoothing: Math.round(NUM(raw.rigSmoothing, 48, 0, 200)),
    rigMaxInfluences: Math.round(NUM(raw.rigMaxInfluences, 4, 1, 4)),
    rigMinConfidence: NUM(raw.rigMinConfidence, 0, 0, 1),

    saveIntermediates: raw.saveIntermediates !== false,
    stepsPrefix: String(raw.stepsPrefix || '3d/steps').replace(/[^A-Za-z0-9_\-/]/g, ''),

    filenamePrefix: String(raw.filenamePrefix || '3d/pipeline').replace(/[^A-Za-z0-9_\-/]/g, ''),
    lowVram: !!raw.lowVram,
  };

  // SDXL img2img at denoise 1.0 keeps nothing of the source image, so the whole
  // point of this pipeline (start from the render, restyle it) is lost. Kontext
  // carries the image through reference latents instead, so it needs the full
  // range. Enforced here as well as in the UI, since presets bypass the UI.
  if (p.synthBackend !== 'kontext') {
    p.synthDenoise = Math.min(p.synthDenoise, 0.9);
  }

  // Saying the subject is symmetric and then filling gaps without consulting the
  // mirrored side would throw away the very information just declared, so the
  // plain surface blend is upgraded. An explicit 'nearest' or 'none' is left
  // alone: those are deliberate choices about the fill, not defaults.
  if (p.symmetric && p.fillMode === 'surface') p.fillMode = 'surface+mirror';

  return p.lowVram ? applyLowVram(p) : p;
}

/**
 * Cap the settings that dominate VRAM. A single run loads a 3D shape model and
 * a 2D image model, and on a 10 GB card the peaks collide; these limits keep the
 * whole pipeline inside roughly 6 GB of working set at some cost in detail.
 */
function applyLowVram(p) {
  p.textureResolution = Math.min(p.textureResolution, 1024);
  p.bakeResolution = Math.min(p.bakeResolution, 512);
  p.synthResolution = Math.min(p.synthResolution, 768);
  p.octreeResolution = Math.min(p.octreeResolution, 256);
  p.latentResolution = Math.min(p.latentResolution, 3072);
  p.aoSamples = Math.min(p.aoSamples, 32);
  // The TRELLIS shape decode runs while the 5 GB DiT is still resident, and
  // ComfyUI will not evict it, so the decode has only the remainder to work in.
  // At 1536 it asks for 3.1 GiB against ~2 GiB free and dies; 1024 is the
  // largest that reliably fits on a 10 GB card.
  p.trellisUpsample = Math.min(p.trellisUpsample, 1024);
  // Ambient occlusion casts a full hemisphere of rays per texel; it is the most
  // expensive optional stage and the least missed.
  p.bakeAO = false;
  return p;
}

module.exports = {
  normaliseParams: normaliseParams,
  applyLowVram: applyLowVram,
  NUM: NUM,
};
