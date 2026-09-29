'use strict';
/**
 * Builds ComfyUI API-format prompt graphs for the image -> 3D pipeline.
 *
 * Geometry sources:
 *   hy3d_mv  - Hunyuan3D 2.0 multiview DiT, conditioned on front/left/back/right.
 *   hy3d_21  - Hunyuan3D 2.1, single (front) image.
 *   trellis2 - TRELLIS.2 cascade; also the only local source of colour.
 *
 * Texture always comes from the TRELLIS.2 texture cascade, baked into a UV atlas
 * with BakeTextureFromVoxel. When geometry is a Hunyuan mesh the colour voxels
 * are baked onto that mesh instead of the TRELLIS one - see TEXTURE_ALIGNMENT in
 * the README, that is the experimental hybrid path.
 */

const CKPT_MV = 'hunyuan3d-dit-v2-mv_fp16.safetensors';
const CKPT_21 = 'hunyuan_3d_v2.1.safetensors';
const T2_UNET = 'trellis_2_int8_convrot.safetensors';
const T2_SHAPE_VAE = 'trellis_2_shape_vae_bf16.safetensors';
const T2_TEX_VAE = 'trellis_2_texture_vae_bf16.safetensors';
const T2_CLIPVISION = 'dino_v3_L_naf_fp32.safetensors';
const BG_MODEL = 'birefnet.safetensors';

const VIEWS = ['front', 'left', 'back', 'right'];

/**
 * Camera yaw per named view. World is Y-up and at yaw 0 the camera sits on +Z,
 * so the subject's face points toward +Z (matches CreateCameraInfo's orbit).
 *
 * Which side is "left" follows from that: for a character facing +Z with up +Y,
 * their own right hand points along `forward x up` = cross(+Z, +Y) = -X. A camera
 * at yaw 90 sits at +X, so it is looking at the character's LEFT side.
 *
 * Labels on turnaround art are not universally the character's own left/right --
 * some sheets name views by where the camera sits -- so `swapSides` flips this.
 */
const VIEW_YAW = { front: 0, left: 90, back: 180, right: 270 };
const VIEW_YAW_SWAPPED = { front: 0, right: 90, back: 180, left: 270 };

function yawFor(view, swapSides) {
  return (swapSides ? VIEW_YAW_SWAPPED : VIEW_YAW)[view];
}

class Graph {
  constructor() { this.n = 0; this.g = {}; }
  add(class_type, inputs, title) {
    const id = String(++this.n);
    this.g[id] = { class_type, inputs, _meta: { title: title || class_type } };
    return id;
  }
  out(id, slot) { return [id, slot || 0]; }
  json() { return this.g; }
}

/** Deterministic per-stage seeds derived from one user seed. */
function seedOf(base, i) { return (Number(base) + i * 7919) % 0xffffffffff; }

/**
 * Load an image and, optionally, cut it out of its background and re-frame it
 * square around the subject.
 *
 * Returns { raw, mask, framed }. The shape models want `framed` (subject
 * centred at a fixed size); projection texturing wants `raw` + `mask`, because
 * it derives its own framing from the mask bounding box and would be thrown off
 * by a pre-cropped image.
 */
function prepImage(G, filename, opts) {
  const load = G.add('LoadImage', { image: filename }, 'Load ' + opts.label);
  const raw = G.out(load, 0);
  if (!opts.removeBackground) return { raw: raw, mask: null, framed: raw };

  const rb = G.add('RemoveBackground', {
    bg_removal_model: G.out(opts.bgModelId, 0),
    image: raw,
  }, 'Cutout ' + opts.label);
  const mask = G.out(rb, 0);

  const crop = G.add('ImageCropToMask', {
    images: raw,
    masks: mask,
    width: opts.size,
    height: opts.size,
    pad_factor: opts.padFactor,
    grow_mask: 0,
    background: opts.background,
  }, 'Frame ' + opts.label);
  return { raw: raw, mask: mask, framed: G.out(crop, 0) };
}

/**
 * Project the reference art straight onto the finished low-poly mesh and bake it
 * into the UV atlas. This is what puts the user's own artwork on a Hunyuan
 * multiview mesh: every texel takes colour from whichever view sees that part of
 * the surface most directly.
 *
 * Needs the comfyui_multiview_projection custom nodes.
 */
function buildProjectionTexture(G, p, prepped, lowPoly, denseMesh, extraViews) {
  let views = extraViews || null;
  const fits = [];

  for (const v of VIEWS) {
    if (!prepped[v]) continue;

    // Generated turnaround images are never valid projection references. The
    // bake needs an exact yaw and an orthographic camera; a generative model
    // gives neither - it returns a perspective image at whatever angle it felt
    // like, and "left" and "right" prompts commonly produce the same rotation.
    // They are still useful as shape conditioning, which tolerates approximate
    // views, so they stay in `prepped` for buildHunyuan.
    if (prepped[v].generated) continue;

    let yaw = yawFor(v, p.swapSides);
    let pitch = p.viewPitch;
    let title = 'View: ' + v + ' (' + yaw + ' deg)';

    // Off-axis art (isometric, three-quarter) has no fixed yaw, so recover it
    // from the mesh silhouette instead of assuming a turnaround layout.
    if (p.fitAngles && prepped[v].mask) {
      const fit = G.add('ProjectionViewFit', {
        // Silhouette matching only needs the outline, and the low-poly has the
        // same one for a fraction of the rasterisation cost.
        mesh: lowPoly,
        mask: prepped[v].mask,
        image: prepped[v].raw,
        // Hunyuan puts its conditioning image at the front, so the yaw is known
        // for the art that drove the shape. Searching wide invites a wrong side:
        // rounded subjects match nearly as well from any angle.
        yaw_center: 0.0,
        yaw_range: p.fitYawRange,
        yaw_steps: p.fitYawSteps,
        pitch_min: p.fitPitchMin,
        pitch_max: p.fitPitchMax,
        pitch_steps: p.fitPitchSteps,
        refine_passes: 2,
        search_resolution: 160,
      }, 'Fit angle: ' + v);
      yaw = G.out(fit, 0);
      pitch = G.out(fit, 1);
      title = 'View: ' + v + ' (fitted)';
      fits.push({ node: fit, name: v });
    }

    const inputs = {
      image: prepped[v].raw,
      yaw: yaw,
      pitch: pitch,
      scale: p.viewScale,
      offset_x: 0.0,
      offset_y: 0.0,
      weight: 1.0,
    };
    if (prepped[v].mask) inputs.mask = prepped[v].mask;
    if (views) inputs.views = views;
    views = G.out(G.add('ProjectionViewAdd', inputs, title), 0);
  }

  // Bilateral symmetry: a side view mirrored horizontally *is* the opposite
  // side view. At yaw 90 the camera sees the subject's left with the face
  // toward frame-left; at yaw 270 it sees the right with the face toward
  // frame-right, which is precisely the horizontal mirror. So for a symmetric
  // subject this recovers the missing side exactly, rather than approximating
  // it by blending neighbouring surface.
  //
  // Only ever applied to a side that has no art of its own, and only to real
  // reference images: a generated view has no reliable camera to mirror.
  if (p.symmetric) {
    for (const [have, want] of [['left', 'right'], ['right', 'left']]) {
      if (!prepped[have] || prepped[want] || prepped[have].generated) continue;

      const img = G.out(G.add('ImageFlip', {
        image: prepped[have].raw, flip_method: 'y-axis: horizontally',
      }, 'Mirror ' + have + ' -> ' + want), 0);

      const inputs = {
        image: img,
        yaw: yawFor(want, p.swapSides),
        pitch: p.viewPitch,
        scale: p.viewScale,
        offset_x: 0.0,
        offset_y: 0.0,
        // Just under a real view's weight: symmetry is an assumption about the
        // subject, so genuine art wins anywhere the two overlap.
        weight: 0.9,
      };
      if (prepped[have].mask) {
        // Masks have no flip node of their own, so it goes round through IMAGE.
        const m = G.add('MaskToImage', { mask: prepped[have].mask }, 'Mask ' + have);
        const mf = G.add('ImageFlip', {
          image: G.out(m, 0), flip_method: 'y-axis: horizontally',
        }, 'Mirror mask ' + have);
        inputs.mask = G.out(G.add('ImageToMask',
          { image: G.out(mf, 0), channel: 'red' }, 'Mirrored mask ' + want), 0);
      }
      if (views) inputs.views = views;
      views = G.out(G.add('ProjectionViewAdd', inputs,
                          'View: ' + want + ' (mirrored from ' + have + ')'), 0);
    }
  }

  const bake = G.add('MultiviewProjectionBake', {
    mesh: lowPoly,
    views: views,
    texture_size: p.textureResolution,
    facing_power: p.facingPower,
    occlusion: p.projectionOcclusion,
    // Reject grazing samples and approximate those texels instead - this is what
    // keeps a front+back-only set from smearing down the sides.
    mask_erode: p.maskErode,
    min_facing: p.minFacing,
    fill_unseen: p.fillMode,
    palette_colors: p.paletteColors,
    palette_strength: p.paletteStrength,
    palette_keep_shading: p.paletteKeepShading,
    dilate: 4,
    // Occlusion is tested against the dense mesh so the low-poly silhouette
    // does not shadow itself along grazing edges.
    reference_mesh: denseMesh,
  }, 'Bake texture from views');

  return {
    baseColor: G.out(bake, 0),
    coverage: G.out(bake, 1),
    views: views,
    fits: fits,
  };
}

/**
 * View synthesis: invent the reference art you did not draw.
 *
 * Rather than generating a viewpoint from nothing, this renders the partly
 * textured mesh from the missing angle first. That render already carries your
 * own art wherever a real view reached, so an img2img pass at moderate denoise
 * inherits your palette and style instead of inventing a new one — and because
 * ProjectionViewRender uses the same orthographic camera as the bake, the
 * refined image reprojects onto the model with no alignment error.
 *
 * Returns { views, saves } where `views` chains onto an existing view set.
 */
function buildViewSynthesis(G, p, texturedMesh, coverage, existingViews, angles, styleRef) {
  const kontext = p.synthBackend === 'kontext';

  // Two backends. SDXL is a plain img2img over the render: fast, but the model
  // never sees the source art, so style comes only from the render and prompt.
  // Flux Kontext takes the source art as a second reference latent, which is
  // what actually makes a generated view match the original's art style.
  let ckpt = null, fluxUnet = null, fluxVae = null, fluxClip = null;
  let pos, neg;

  if (kontext) {
    fluxUnet = G.add('UNETLoader',
      { unet_name: p.kontextUnet, weight_dtype: 'default' }, 'Flux Kontext');
    fluxClip = G.add('DualCLIPLoader', {
      clip_name1: p.kontextClipL, clip_name2: p.kontextT5,
      type: 'flux', device: 'default',
    }, 'Flux text encoders');
    fluxVae = G.add('VAELoader', { vae_name: p.kontextVae }, 'Flux VAE');
    pos = null;            // built per angle, so the instruction can name the view
    neg = null;            // Kontext runs cfg 1 with a zeroed negative
  } else {
    ckpt = G.add('CheckpointLoaderSimple',
      { ckpt_name: p.imageModel }, 'Image model');
    pos = G.add('CLIPTextEncode',
      { clip: G.out(ckpt, 1), text: p.synthPrompt }, 'Style prompt');
    neg = G.add('CLIPTextEncode',
      { clip: G.out(ckpt, 1), text: p.synthNegative }, 'Negative prompt');
  }

  // Structural conditioning. Without it there is no useful denoise setting: low
  // enough to stay on the geometry only recolours what is already there, and
  // high enough to invent a satchel's leather and buckles walks off the
  // silhouette so the result no longer reprojects. A depth/normal ControlNet
  // pins the generation to the mesh and leaves the model free to paint.
  // Loaded once and shared by every view.
  let controlNet = null;
  if (!kontext && p.controlNet) {
    const loaded = G.add('ControlNetLoader',
      { control_net_name: p.controlNet }, 'ControlNet');
    controlNet = G.out(G.add('SetUnionControlNetType', {
      control_net: G.out(loaded, 0),
      type: p.controlNetType,
    }, 'ControlNet type: ' + p.controlNetType), 0);
  }

  // The style reference is encoded once and reused across every view.
  let styleLatent = null;
  if (kontext && styleRef) {
    const scaled = G.add('FluxKontextImageScale',
      { image: styleRef }, 'Scale style reference');
    styleLatent = G.out(G.add('VAEEncode',
      { pixels: G.out(scaled, 0), vae: G.out(fluxVae, 0) }, 'Encode style reference'), 0);
  }

  let views = existingViews;
  const saves = [];
  const made = [];

  for (const a of angles) {
    const label = a.name + ' ' + a.yaw + 'deg';

    const renderInputs = {
      mesh: texturedMesh,
      yaw: a.yaw,
      pitch: p.viewPitch,
      resolution: p.synthResolution,
      background: p.synthBackground,
      shade: 0.0,
    };
    // Optional in the node, and an explicit null is not a valid API link.
    if (coverage) renderInputs.coverage = coverage;
    const render = G.add('ProjectionViewRender', renderInputs, 'Render ' + label);

    // Gaps get grown and softened so the model has room to blend, rather than
    // painting up to a hard edge against the real art.
    const gap = G.add('GrowMask', {
      mask: G.out(render, 2), expand: p.gapGrow, tapered_corners: true,
    }, 'Grow gaps ' + label);

    let decoded;
    if (kontext) {
      // Kontext responds to instructions, not descriptions, and naming the target
      // view is what stops it simply reproducing the reference. {view} lets the
      // user place it; otherwise it is appended.
      const viewWord = { front: 'front', back: 'back', left: 'left side', right: 'right side' }[a.name] || a.name;
      const text = p.synthPrompt.includes('{view}')
        ? p.synthPrompt.split('{view}').join(viewWord)
        : p.synthPrompt + ' Show the same subject from the ' + viewWord + '.';
      pos = G.add('CLIPTextEncode',
        { clip: G.out(fluxClip, 0), text: text }, 'Instruction ' + label);

      // Kontext edits from its reference latents, so the render goes in both as
      // the starting latent and as reference 1; the source art is reference 2.
      const rScaled = G.add('FluxKontextImageScale',
        { image: G.out(render, 0) }, 'Scale render ' + label);
      const rLatent = G.add('VAEEncode',
        { pixels: G.out(rScaled, 0), vae: G.out(fluxVae, 0) }, 'Encode ' + label);

      let cond = G.add('ReferenceLatent',
        { conditioning: G.out(pos, 0), latent: G.out(rLatent, 0) },
        'Ref: render ' + label);
      if (styleLatent) {
        cond = G.add('ReferenceLatent',
          { conditioning: G.out(cond, 0), latent: styleLatent },
          'Ref: source art ' + label);
        cond = G.add('FluxKontextMultiReferenceLatentMethod', {
          conditioning: G.out(cond, 0),
          reference_latents_method: p.kontextRefMethod,
        }, 'Combine references ' + label);
      }
      const guided = G.add('FluxGuidance',
        { conditioning: G.out(cond, 0), guidance: p.kontextGuidance }, 'Guidance ' + label);
      const zeroed = G.add('ConditioningZeroOut',
        { conditioning: G.out(pos, 0) }, 'Empty negative ' + label);

      const startLatent = p.synthGapsOnly
        ? G.add('SetLatentNoiseMask',
            { samples: G.out(rLatent, 0), mask: G.out(gap, 0) }, 'Limit to gaps ' + label)
        : rLatent;

      const samp = G.add('KSampler', {
        model: G.out(fluxUnet, 0),
        seed: seedOf(p.seed, 11 + angles.indexOf(a)),
        steps: p.synthSteps,
        cfg: 1.0,                                  // Kontext is guidance-distilled
        sampler_name: 'euler',
        scheduler: 'simple',
        positive: G.out(guided, 0),
        negative: G.out(zeroed, 0),
        latent_image: G.out(startLatent, 0),
        denoise: p.synthDenoise,
      }, 'Kontext ' + label);

      const dec = G.add('VAEDecode',
        { samples: G.out(samp, 0), vae: G.out(fluxVae, 0) }, 'Decode ' + label);
      // FluxKontextImageScale snaps to its own supported resolutions, so bring
      // the result back to the render's size - the bake samples the image and
      // the render's mask at the same pixel coordinates.
      decoded = G.add('ImageScale', {
        image: G.out(dec, 0), upscale_method: 'lanczos',
        width: p.synthResolution, height: p.synthResolution, crop: 'disabled',
      }, 'Fit to render ' + label);
    } else {
      const enc = G.add('VAEEncode',
        { pixels: G.out(render, 0), vae: G.out(ckpt, 2) }, 'Encode ' + label);

      // Restricting the noise to the gap mask keeps the real art bit-exact.
      const latent = p.synthGapsOnly
        ? G.add('SetLatentNoiseMask',
            { samples: G.out(enc, 0), mask: G.out(gap, 0) }, 'Limit to gaps ' + label)
        : enc;

      // Depth (or normal) comes off the same render node, so it is the same
      // camera, framing and resolution the bake will reproject through - the
      // conditioning cannot be misaligned with the geometry by construction.
      let posLink = G.out(pos, 0);
      let negLink = G.out(neg, 0);
      if (controlNet) {
        const structure = G.out(render, p.controlNetType === 'normal' ? 4 : 3);
        saves.push({ image: structure, name: 'step4-' + p.controlNetType + '-' + a.name });
        const applied = G.add('ControlNetApplyAdvanced', {
          positive: posLink,
          negative: negLink,
          control_net: controlNet,
          image: structure,
          strength: p.controlNetStrength,
          start_percent: 0.0,
          end_percent: p.controlNetEnd,
        }, 'Lock to geometry ' + label);
        posLink = G.out(applied, 0);
        negLink = G.out(applied, 1);
      }

      const samp = G.add('KSampler', {
        model: G.out(ckpt, 0),
        seed: seedOf(p.seed, 11 + angles.indexOf(a)),
        steps: p.synthSteps,
        cfg: p.synthCfg,
        sampler_name: 'dpmpp_2m',
        scheduler: 'karras',
        positive: posLink,
        negative: negLink,
        latent_image: G.out(latent, 0),
        // With the geometry locked, denoise can be raised far enough for the
        // model to add detail rather than just re-tint what the blend produced.
        denoise: controlNet ? p.controlNetDenoise : p.synthDenoise,
      }, 'Refine ' + label);

      decoded = G.add('VAEDecode',
        { samples: G.out(samp, 0), vae: G.out(ckpt, 2) }, 'Decode ' + label);
    }

    // Hard-preserve everything a real reference already covered: only the gap
    // region is allowed to come from the generated image.
    const merged = p.synthGapsOnly
      ? G.add('ImageCompositeMasked', {
          destination: G.out(render, 0), source: G.out(decoded, 0),
          x: 0, y: 0, resize_source: false, mask: G.out(gap, 0),
        }, 'Merge ' + label)
      : decoded;

    const addInputs = {
      image: G.out(merged, 0),
      mask: G.out(render, 1),
      yaw: a.yaw,
      pitch: p.viewPitch,
      scale: 1.0,
      offset_x: 0.0,
      offset_y: 0.0,
      // Synthetic art loses to real art wherever the two overlap.
      weight: p.syntheticWeight,
    };
    if (views) addInputs.views = views;
    views = G.out(G.add('ProjectionViewAdd', addInputs,
                        'View: ' + a.name + ' (synthetic)'), 0);

    saves.push({ image: G.out(render, 0), name: 'step3-render-' + a.name });
    saves.push({ image: G.out(merged, 0), name: 'step4-synth-' + a.name });
    saves.push({ mask: G.out(gap, 0), name: 'step3b-gaps-' + a.name });

    made.push({ name: a.name, yaw: a.yaw,
                image: G.out(merged, 0), mask: G.out(render, 1) });
  }

  return { views: views, saves: saves, made: made };
}

// Camera instructions for inventing a turnaround from a single image. Phrased as
// camera moves: a prompt that opens by asking to preserve the image pulls Kontext
// toward reproducing it instead of re-posing the subject, which is exactly how
// the first attempt at this failed.
// Spelled out as a rotation direction rather than a side name: "left" and
// "right" alone are ambiguous to the model and reliably produce the same
// three-quarter view twice. Even so, the result is an approximate perspective
// image, not a calibrated orthographic profile - see buildProjectionTexture.
const VIEW_INSTRUCTION = {
  left: 'Turn the subject 90 degrees clockwise as seen from above, so the camera '
      + 'looks at its left flank. Strict side profile, nose pointing to the right '
      + 'edge of the frame, no part of the chest visible.',
  right: 'Turn the subject 90 degrees anticlockwise as seen from above, so the '
       + 'camera looks at its right flank. Strict side profile, nose pointing to '
       + 'the left edge of the frame, no part of the chest visible.',
  back: 'Turn the subject 180 degrees so the camera looks at its back. The head '
      + 'faces directly away, no facial features visible.',
  front: 'Show the subject from directly in front, facing the camera.',
};

/**
 * Invent the missing reference views from one source image, before any 3D runs.
 *
 * This is upstream of everything else: the generated views become real
 * references, so they improve the geometry (Hunyuan multiview conditions on all
 * four) as well as the texture coverage. That is a different job from the
 * post-geometry synthesis stage, which only restyles renders.
 *
 * Returns a map of view name -> { raw, mask, framed }, plus images to save.
 */
function buildReferenceViewGeneration(G, p, frontRaw, wanted, bgModelId) {
  const unet = G.add('UNETLoader',
    { unet_name: p.kontextUnet, weight_dtype: 'default' }, 'Flux Kontext (views)');
  const clip = G.add('DualCLIPLoader', {
    clip_name1: p.kontextClipL, clip_name2: p.kontextT5,
    type: 'flux', device: 'default',
  }, 'Flux text encoders (views)');
  const vae = G.add('VAELoader', { vae_name: p.kontextVae }, 'Flux VAE (views)');

  const scaled = G.add('FluxKontextImageScale', { image: frontRaw }, 'Scale source');
  const srcLatent = G.add('VAEEncode',
    { pixels: G.out(scaled, 0), vae: G.out(vae, 0) }, 'Encode source');

  const made = {};
  const saves = [];

  wanted.forEach((view, i) => {
    const text = (VIEW_INSTRUCTION[view] || '') + ' ' + p.refViewStyle;
    const txt = G.add('CLIPTextEncode',
      { clip: G.out(clip, 0), text: text }, 'Instruction: ' + view);

    const cond = G.add('ReferenceLatent',
      { conditioning: G.out(txt, 0), latent: G.out(srcLatent, 0) }, 'Ref: source (' + view + ')');
    const guided = G.add('FluxGuidance',
      { conditioning: G.out(cond, 0), guidance: p.kontextGuidance }, 'Guidance ' + view);
    const zero = G.add('ConditioningZeroOut',
      { conditioning: G.out(txt, 0) }, 'Empty negative ' + view);

    const samp = G.add('KSampler', {
      model: G.out(unet, 0),
      seed: seedOf(p.seed, 31 + i),
      steps: p.synthSteps,
      cfg: 1.0,
      sampler_name: 'euler',
      scheduler: 'simple',
      positive: G.out(guided, 0),
      negative: G.out(zero, 0),
      latent_image: G.out(srcLatent, 0),
      denoise: 1.0,                       // Kontext rebuilds from its reference
    }, 'Generate view: ' + view);

    const img = G.out(G.add('VAEDecode',
      { samples: G.out(samp, 0), vae: G.out(vae, 0) }, 'Decode ' + view), 0);

    // Generated views arrive on whatever background the model chose, so they get
    // the same cutout and framing treatment as art the user supplied.
    let mask = null, framed = img;
    if (bgModelId) {
      mask = G.out(G.add('RemoveBackground',
        { bg_removal_model: G.out(bgModelId, 0), image: img }, 'Cutout ' + view), 0);
      framed = G.out(G.add('ImageCropToMask', {
        images: img, masks: mask,
        width: 1024, height: 1024,
        pad_factor: p.padFactor, grow_mask: 0,
        background: p.hunyuanBackground,
      }, 'Frame ' + view), 0);
    }

    // Flagged so the projection bake skips these: see buildProjectionTexture.
    made[view] = { raw: img, mask: mask, framed: framed, generated: true };
    saves.push({ image: img, name: 'step0-refview-' + view });
  });

  return { views: made, saves: saves };
}

/**
 * Image-to-image on its own, with no 3D anywhere in the graph.
 *
 * The synthesis stage is hard to judge inside the full pipeline: a bad result
 * could be the shape model, the angle fit, the projection, or the image model.
 * This builds only the image half, using the same wiring, so the generator can
 * be tuned in seconds instead of half an hour.
 */
function buildImagePrompt(p) {
  const G = new Graph();
  const kontext = p.synthBackend === 'kontext';

  const src = G.out(G.add('LoadImage', { image: p.images.front }, 'Source image'), 0);
  const ref2 = p.images.back
    ? G.out(G.add('LoadImage', { image: p.images.back }, 'Style reference'), 0)
    : null;

  let decoded;
  if (kontext) {
    const unet = G.add('UNETLoader',
      { unet_name: p.kontextUnet, weight_dtype: 'default' }, 'Flux Kontext');
    const clip = G.add('DualCLIPLoader', {
      clip_name1: p.kontextClipL, clip_name2: p.kontextT5,
      type: 'flux', device: 'default',
    }, 'Flux text encoders');
    const vae = G.add('VAELoader', { vae_name: p.kontextVae }, 'Flux VAE');

    const scaled = G.add('FluxKontextImageScale', { image: src }, 'Scale source');
    const lat = G.add('VAEEncode',
      { pixels: G.out(scaled, 0), vae: G.out(vae, 0) }, 'Encode source');

    const txt = G.add('CLIPTextEncode',
      { clip: G.out(clip, 0), text: p.synthPrompt }, 'Instruction');

    let cond = G.add('ReferenceLatent',
      { conditioning: G.out(txt, 0), latent: G.out(lat, 0) }, 'Ref: source');
    if (ref2) {
      const s2 = G.add('FluxKontextImageScale', { image: ref2 }, 'Scale reference 2');
      const l2 = G.add('VAEEncode',
        { pixels: G.out(s2, 0), vae: G.out(vae, 0) }, 'Encode reference 2');
      cond = G.add('ReferenceLatent',
        { conditioning: G.out(cond, 0), latent: G.out(l2, 0) }, 'Ref: second');
      cond = G.add('FluxKontextMultiReferenceLatentMethod', {
        conditioning: G.out(cond, 0),
        reference_latents_method: p.kontextRefMethod,
      }, 'Combine references');
    }
    const guided = G.add('FluxGuidance',
      { conditioning: G.out(cond, 0), guidance: p.kontextGuidance }, 'Guidance');
    const zero = G.add('ConditioningZeroOut',
      { conditioning: G.out(txt, 0) }, 'Empty negative');

    const samp = G.add('KSampler', {
      model: G.out(unet, 0), seed: p.seed, steps: p.synthSteps, cfg: 1.0,
      sampler_name: 'euler', scheduler: 'simple',
      positive: G.out(guided, 0), negative: G.out(zero, 0),
      latent_image: G.out(lat, 0), denoise: p.synthDenoise,
    }, 'Kontext sampler');
    decoded = G.add('VAEDecode',
      { samples: G.out(samp, 0), vae: G.out(vae, 0) }, 'Decode');
  } else {
    const ckpt = G.add('CheckpointLoaderSimple',
      { ckpt_name: p.imageModel }, 'Checkpoint');
    const pos = G.add('CLIPTextEncode',
      { clip: G.out(ckpt, 1), text: p.synthPrompt }, 'Prompt');
    const neg = G.add('CLIPTextEncode',
      { clip: G.out(ckpt, 1), text: p.synthNegative }, 'Negative');
    const lat = G.add('VAEEncode',
      { pixels: src, vae: G.out(ckpt, 2) }, 'Encode source');
    const samp = G.add('KSampler', {
      model: G.out(ckpt, 0), seed: p.seed, steps: p.synthSteps, cfg: p.synthCfg,
      sampler_name: 'dpmpp_2m', scheduler: 'karras',
      positive: G.out(pos, 0), negative: G.out(neg, 0),
      latent_image: G.out(lat, 0), denoise: p.synthDenoise,
    }, 'SDXL sampler');
    decoded = G.add('VAEDecode',
      { samples: G.out(samp, 0), vae: G.out(ckpt, 2) }, 'Decode');
  }

  const save = G.add('SaveImage',
    { images: G.out(decoded, 0), filename_prefix: 'imagelab/out' }, 'Save result');
  return { prompt: G.json(), saveNode: save };
}

/** Hunyuan3D geometry -> dense MESH link. */
function buildHunyuan(G, p, prepped) {
  const multiview = p.geometry === 'hy3d_mv';
  const ckpt = G.add('ImageOnlyCheckpointLoader',
    { ckpt_name: multiview ? CKPT_MV : CKPT_21 }, 'Hunyuan3D checkpoint');

  const model = G.add('ModelSamplingAuraFlow',
    { model: G.out(ckpt, 0), shift: p.shift }, 'Model sampling');

  // The MV checkpoint wants uncropped CLIP framing; 2.1 centre-crops.
  const crop = multiview ? 'none' : 'center';
  const encode = (imgLink, label) => G.add('CLIPVisionEncode', {
    clip_vision: G.out(ckpt, 1), image: imgLink, crop: crop,
  }, 'CLIP Vision ' + label);

  let positive, negative;
  if (multiview) {
    const inputs = {};
    for (const v of VIEWS) if (prepped[v]) inputs[v] = G.out(encode(prepped[v].framed, v), 0);
    const cond = G.add('Hunyuan3Dv2ConditioningMultiView', inputs, 'Multiview conditioning');
    positive = G.out(cond, 0); negative = G.out(cond, 1);
  } else {
    const cond = G.add('Hunyuan3Dv2Conditioning',
      { clip_vision_output: G.out(encode(prepped.front.framed, 'front'), 0) }, 'Conditioning');
    positive = G.out(cond, 0); negative = G.out(cond, 1);
  }

  const latent = G.add('EmptyLatentHunyuan3Dv2',
    { resolution: p.latentResolution, batch_size: 1 }, 'Empty latent');

  const sampler = G.add('KSampler', {
    model: G.out(model, 0),
    seed: seedOf(p.seed, 0),
    steps: p.steps,
    cfg: p.cfg,
    sampler_name: p.samplerName,
    scheduler: p.scheduler,
    positive: positive,
    negative: negative,
    latent_image: G.out(latent, 0),
    denoise: 1.0,
  }, 'Shape sampler');

  const voxel = G.add('VAEDecodeHunyuan3D', {
    samples: G.out(sampler, 0),
    vae: G.out(ckpt, 2),
    num_chunks: p.numChunks,
    octree_resolution: p.octreeResolution,
  }, 'Decode to voxels');

  const mesh = G.add('VoxelToMesh', {
    voxel: G.out(voxel, 0),
    algorithm: 'surface net',
    threshold: p.surfaceThreshold,
  }, 'Voxels to mesh');

  return G.out(mesh, 0);
}

/**
 * TRELLIS.2 cascade. Mirrors the shipped 3d_pixal3d_trellis2 template exactly:
 * structure -> shape -> upsample -> detail -> texture.
 */
function buildTrellis2(G, p, frontBlack, wantColour) {
  const unet = G.add('UNETLoader',
    { unet_name: T2_UNET, weight_dtype: 'default' }, 'TRELLIS.2 UNet');
  const shapeVae = G.add('VAELoader', { vae_name: T2_SHAPE_VAE }, 'Shape VAE');
  const dino = G.add('CLIPVisionLoader', { clip_name: T2_CLIPVISION }, 'DINOv3');

  const cond = G.add('Trellis2Conditioning',
    { clip_vision_model: G.out(dino, 0), image: frontBlack }, 'TRELLIS.2 conditioning');
  const pos0 = G.out(cond, 0), neg0 = G.out(cond, 1);

  // CFG override + rescale reproduce the reference pipeline's default behaviour.
  const mStructCfg = G.add('CFGOverride',
    { model: G.out(unet, 0), cfg: 1, start_percent: 0.667, end_percent: 1 },
    'CFG override (structure)');
  const mStructRes = G.add('RescaleCFG',
    { model: G.out(mStructCfg, 0), multiplier: 0.7 }, 'Rescale CFG (structure)');
  const mStruct = G.add('ModelSamplingSD3',
    { model: G.out(mStructRes, 0), shift: 5 }, 'Model sampling SD3');

  const lat0 = G.add('EmptyTrellis2LatentStructure', { batch_size: 1 }, 'Empty structure latent');
  const k1 = G.add('KSampler', {
    model: G.out(mStruct, 0), seed: seedOf(p.seed, 1), steps: 12, cfg: 7.5,
    sampler_name: 'euler', scheduler: 'normal',
    positive: pos0, negative: neg0, latent_image: G.out(lat0, 0), denoise: 1.0,
  }, 'Structure sampler');

  const structVox = G.add('VaeDecodeStructureTrellis2',
    { samples: G.out(k1, 0), vae: G.out(shapeVae, 0), resolution: '32' }, 'Decode structure');

  const shapeStage = G.add('Trellis2ShapeStage',
    { positive: pos0, negative: neg0, voxel: G.out(structVox, 0) }, 'Shape stage');

  const mShapeCfg = G.add('CFGOverride',
    { model: G.out(unet, 0), cfg: 1, start_percent: 0.769, end_percent: 1 },
    'CFG override (shape)');
  const mShape = G.add('RescaleCFG',
    { model: G.out(mShapeCfg, 0), multiplier: 0.5 }, 'Rescale CFG (shape)');

  const k2 = G.add('KSampler', {
    model: G.out(mShape, 0), seed: seedOf(p.seed, 2), steps: 20, cfg: 7.5,
    sampler_name: 'euler', scheduler: 'normal',
    positive: G.out(shapeStage, 0), negative: G.out(shapeStage, 1),
    latent_image: G.out(shapeStage, 2), denoise: 1.0,
  }, 'Shape sampler');

  const upsample = G.add('Trellis2UpsampleStage', {
    positive: G.out(shapeStage, 0), negative: G.out(shapeStage, 1),
    shape_latent: G.out(k2, 0), vae: G.out(shapeVae, 0),
    target_resolution: p.trellisUpsample,
  }, 'Upsample stage');

  const k3 = G.add('KSampler', {
    model: G.out(mShape, 0), seed: seedOf(p.seed, 3), steps: 12, cfg: 7.5,
    sampler_name: 'euler', scheduler: 'simple',
    positive: G.out(upsample, 0), negative: G.out(upsample, 1),
    latent_image: G.out(upsample, 2), denoise: 1.0,
  }, 'Detail sampler');

  const shapeDec = G.add('VaeDecodeShapeTrellis',
    { samples: G.out(k3, 0), vae: G.out(shapeVae, 0) }, 'Decode shape');

  const result = {
    mesh: G.out(shapeDec, 0),
    shapeSubdivides: G.out(shapeDec, 1),
    voxelColors: null,
  };
  if (!wantColour) return result;

  const texVae = G.add('VAELoader', { vae_name: T2_TEX_VAE }, 'Texture VAE');
  const texStage = G.add('Trellis2TextureStage', {
    positive: G.out(upsample, 0), negative: G.out(upsample, 1),
    shape_latent: G.out(k3, 0),
  }, 'Texture stage');

  // Texture stage runs at cfg 1 off the raw UNet - no CFG rescale.
  const k4 = G.add('KSampler', {
    model: G.out(unet, 0), seed: seedOf(p.seed, 4), steps: 12, cfg: 1.0,
    sampler_name: 'euler', scheduler: 'normal',
    positive: G.out(texStage, 0), negative: G.out(texStage, 1),
    latent_image: G.out(texStage, 2), denoise: 1.0,
  }, 'Texture sampler');

  result.voxelColors = G.out(G.add('VaeDecodeTextureTrellis', {
    samples: G.out(k4, 0), vae: G.out(texVae, 0),
    shape_subdivides: result.shapeSubdivides,
  }, 'Decode texture'), 0);

  return result;
}

/**
 * Dense mesh -> low-poly, UV-unwrapped mesh at the requested triangle budget.
 *
 * `allowRemesh` gates RemeshMesh, which resamples into a hardcoded [-0.5, 0.5]
 * cube. TRELLIS.2 meshes live in exactly that domain; Hunyuan meshes overrun it
 * and come back with the head and lower legs sliced off, so remesh is refused
 * for them regardless of what the caller asks for.
 */
function buildRetopo(G, p, denseMesh, allowRemesh) {
  let m = denseMesh;

  if (p.remesh && allowRemesh) {
    // DynamicCombo inputs are sent flat: the option key on the input itself, and
    // that option's sub-widgets as "<input>.<sub>" siblings.
    m = G.out(G.add('RemeshMesh', {
      mesh: m,
      resolution: p.remeshResolution,
      sign_mode: 'udf',
      'sign_mode.qef': false,
      'sign_mode.drop_inverted_components': false,
      'sign_mode.drop_enclosed_components': false,
      band: 1.0,
      project_back: 0.0,
      fix_poles: false,
      smooth_iters: p.remeshSmooth,
      drop_small_components: 0.01,
      precluster_max_verts: 20000000,
    }, 'Remesh (narrow-band DC)'), 0);
  }

  // target_face_count is the triangle budget - this is the low-poly knob.
  const decInputs = {
    mesh: m,
    target_face_count: p.targetTriangles,
    placement_mode: p.decimatePlacement,
  };
  if (p.decimatePlacement === 'qem') {
    decInputs['placement_mode.line_quadric_weight'] = 0.0;
    decInputs['placement_mode.feature_edge_quadric_weight'] = 0.0;
    decInputs['placement_mode.feature_edge_min_dihedral_deg'] = 30.0;
    decInputs['placement_mode.clamp_v_to_edge'] = true;
  }
  const dec = G.add('DecimateMesh', decInputs, 'Decimate to ' + p.targetTriangles + ' tris');

  const smooth = G.add('MeshSmoothNormals',
    { mesh: G.out(dec, 0), crease_angle: p.creaseAngle }, 'Smooth normals');

  const uv = G.add('UnwrapMesh', {
    mesh: G.out(smooth, 0),
    segmenter: 'pec',
    resolution: p.textureResolution,
    padding: 2,
    weld_distance: 0.0002,
  }, 'Unwrap UVs');

  return { lowPoly: G.out(uv, 0), dense: m };
}

/** Main entry point. `params` must already be normalised by the caller. */
/**
 * Fit the Mixamo skeleton to the finished mesh and write a second, skinned GLB.
 *
 * Always given the *final* mesh. Skin weights are per-vertex, and both
 * UnwrapMesh (which splits vertices along chart seams) and MeshSmoothNormals
 * (which splits them at creases) change the vertex count, so rigging anything
 * earlier would bind weights to indices that no longer mean the same thing.
 */
function buildRigging(G, p, finalMesh, baseColor) {
  if (!p.autoRig) return null;

  const rig = G.add('AutoRigHumanoid', {
    mesh: finalMesh,
    facing: p.rigFacing,
    symmetrize: p.rigSymmetrize,
    smoothing: p.rigSmoothing,
    max_influences: p.rigMaxInfluences,
    min_confidence: p.rigMinConfidence,
  }, 'Auto rig humanoid');

  const rigSave = {
    mesh: finalMesh,
    rig: G.out(rig, 0),
    filename_prefix: p.filenamePrefix + '-rigged',
    bone_prefix: 'mixamorig:',
  };
  // The rigged GLB is written by our own writer, which does not inherit the
  // material SaveGLB builds, so hand it the atlas directly.
  if (baseColor) rigSave.base_color = baseColor;
  G.add('SaveRiggedGLB', rigSave, 'Save rigged GLB');

  if (p.saveIntermediates) {
    for (const [mode, name] of [['skeleton', 'step7-rig-skeleton'],
                                ['weights', 'step7-rig-weights']]) {
      const prev = G.add('PreviewRig', {
        mesh: finalMesh, rig: G.out(rig, 0),
        resolution: 768, view: 'front+side', mode: mode,
      }, 'Rig preview (' + mode + ')');
      G.add('SaveImage', {
        images: G.out(prev, 0), filename_prefix: p.stepsPrefix + '/' + name,
      }, 'Save ' + name);
    }
  }
  return rig;
}

function buildPrompt(params) {
  const p = params;
  const G = new Graph();

  const bgModelId = p.removeBackground
    ? G.add('LoadBackgroundRemovalModel', { bg_removal_name: BG_MODEL }, 'BiRefNet')
    : null;

  const prepped = {};
  for (const v of VIEWS) {
    if (!p.images[v]) continue;
    prepped[v] = prepImage(G, p.images[v], {
      label: v,
      removeBackground: p.removeBackground,
      bgModelId: bgModelId,
      size: 1024,
      padFactor: p.padFactor,
      background: p.hunyuanBackground,
    });
  }
  if (!prepped.front) throw new Error('A front image is required.');

  // Invent the turnaround from the one image, before any 3D. These become real
  // reference views, so they feed the shape model as well as the texture bake.
  const refGen = [];
  if (p.generateReferenceViews) {
    const missingRefs = VIEWS.filter((v) => !prepped[v]);
    if (missingRefs.length) {
      const gen = buildReferenceViewGeneration(
        G, p, prepped.front.raw, missingRefs, bgModelId);
      for (const [v, entry] of Object.entries(gen.views)) prepped[v] = entry;
      gen.saves.forEach((s) => refGen.push(s));
    }
  }

  const trellisGeometry = p.geometry === 'trellis2';
  const wantTexture = !!p.texture;

  // Two different colour sources, picked by where the geometry came from:
  //   TRELLIS.2  - its own colour voxel cascade, which describes that exact mesh.
  //   Hunyuan    - project the reference art onto the mesh (custom nodes).
  // Baking TRELLIS colour voxels onto a Hunyuan mesh is NOT an option: they are
  // independent reconstructions in different domains (Hunyuan spans ~[-1,1],
  // TRELLIS [-0.5,0.5]) with different proportions, and the result smears.
  const textureMode = !wantTexture ? 'none' : (trellisGeometry ? 'voxel' : 'projection');

  // TRELLIS.2 expects the subject composited on black at pad_factor 1.0.
  let frontBlack = null;
  if (trellisGeometry) {
    frontBlack = (p.removeBackground && p.hunyuanBackground === '#000000' && p.padFactor === 1.0)
      ? prepped.front.framed
      : prepImage(G, p.images.front, {
          label: 'front (texture)',
          removeBackground: p.removeBackground,
          bgModelId: bgModelId,
          size: 1024,
          padFactor: 1.0,
          background: '#000000',
        }).framed;
  }

  // --- geometry -------------------------------------------------------------
  let denseMesh, trellis = null;
  if (trellisGeometry) {
    trellis = buildTrellis2(G, p, frontBlack, textureMode === 'voxel');
    denseMesh = trellis.mesh;
  } else {
    denseMesh = buildHunyuan(G, p, prepped);
  }

  // --- retopo ---------------------------------------------------------------
  let retopo = buildRetopo(G, p, denseMesh, trellisGeometry);
  let lowPoly = retopo.lowPoly;

  // Views synthesis should invent. Normally the named slots left empty; but when
  // the supplied art is off-axis its fitted angle matches none of the canonical
  // four, so every canonical view is worth generating to build a real turnaround.
  const missing = (p.fitAngles ? VIEWS : VIEWS.filter((v) => !p.images[v]))
    .map((v) => ({ name: v, yaw: yawFor(v, p.swapSides) }));
  const doSynthesis = textureMode === 'projection' && p.synthesizeViews && missing.length > 0;
  const intermediates = [];
  refGen.forEach((s) => intermediates.push(s));

  // --- texture + bakes ------------------------------------------------------
  // Normal and AO bakes are always safe: they compare the low-poly against the
  // dense version of the same mesh, so no cross-model correspondence is needed.
  const applyInputs = { mesh: lowPoly };
  let haveMaps = false;

  let coverage = null;
  if (textureMode === 'voxel' && trellis && trellis.voxelColors) {
    const bakeInputs = {
      mesh: lowPoly,
      voxel_colors: trellis.voxelColors,
      texture_size: p.textureResolution,
    };
    // Back-project texels onto the dense surface so a 5k-tri mesh does not bake faceted.
    if (p.useReferenceMesh) bakeInputs.reference_mesh = denseMesh;
    const bake = G.add('BakeTextureFromVoxel', bakeInputs, 'Bake texture from voxels');
    applyInputs.base_color = G.out(bake, 0);
    applyInputs.metallic = G.out(bake, 1);
    applyInputs.roughness = G.out(bake, 2);
    haveMaps = true;
  } else if (textureMode === 'projection') {
    // Pass 1: only the art you actually supplied.
    const proj1 = buildProjectionTexture(G, p, prepped, lowPoly, denseMesh);
    let baseColor = proj1.baseColor;
    coverage = proj1.coverage;
    // The fit overlay shows the recovered pose against the reference art, so a
    // bad angle estimate is visible rather than silently baked in.
    for (const fit of (proj1.fits || [])) {
      intermediates.push({ image: G.out(fit.node, 3), name: 'step1-anglefit-' + fit.name });
    }
    intermediates.push({ image: baseColor, name: 'step2-atlas-real' });
    intermediates.push({ mask: coverage, name: 'step2-coverage-real' });

    if (doSynthesis) {
      // Texture the mesh with what we have, so the renders handed to the image
      // model already carry the user's own art.
      const partial = G.add('ApplyTextureToMesh',
        { mesh: lowPoly, base_color: baseColor }, 'Apply partial texture');

      // The style reference is the user's own art: whichever view they actually
      // supplied, preferring the front.
      const refView = VIEWS.find((v) => prepped[v]) || 'front';
      const synth = buildViewSynthesis(
        G, p, G.out(partial, 0), coverage, proj1.views, missing,
        prepped[refView] ? prepped[refView].raw : null);
      synth.saves.forEach((s) => intermediates.push(s));

      // Optionally let the invented views reshape the mesh too.
      if (p.refineGeometry) {
        const prepped2 = Object.assign({}, prepped);
        for (const m of synth.made) {
          const framed = G.add('ImageCropToMask', {
            images: m.image, masks: m.mask,
            width: 1024, height: 1024,
            pad_factor: p.padFactor, grow_mask: 0,
            background: p.hunyuanBackground,
          }, 'Frame synthetic ' + m.name);
          prepped2[m.name] = { raw: m.image, mask: m.mask, framed: G.out(framed, 0) };
          intermediates.push({ image: G.out(framed, 0), name: 'step5-geomview-' + m.name });
        }
        denseMesh = buildHunyuan(G, p, prepped2);
        retopo = buildRetopo(G, p, denseMesh, false);
        lowPoly = retopo.lowPoly;
        applyInputs.mesh = lowPoly;
      }

      // Pass 2: rebake with real + synthetic views on the final mesh.
      const bake2 = G.add('MultiviewProjectionBake', {
        mesh: lowPoly,
        views: synth.views,
        texture_size: p.textureResolution,
        facing_power: p.facingPower,
        occlusion: p.projectionOcclusion,
        mask_erode: p.maskErode,
        min_facing: p.minFacing,
        fill_unseen: p.fillMode,
        palette_colors: p.paletteColors,
        palette_strength: p.paletteStrength,
        palette_keep_shading: p.paletteKeepShading,
        dilate: 4,
        reference_mesh: denseMesh,
      }, 'Bake texture (all views)');
      baseColor = G.out(bake2, 0);
      coverage = G.out(bake2, 1);
    }

    applyInputs.base_color = baseColor;
    intermediates.push({ image: baseColor, name: 'step6-atlas-final' });
    haveMaps = true;
  }

  if (p.bakeAO) {
    applyInputs.occlusion = G.out(G.add('BakeAmbientOcclusion', {
      low_poly: lowPoly, high_poly: retopo.dense,
      resolution: p.bakeResolution, samples: p.aoSamples,
      max_distance: 0.71, strength: 1.0, bias: 0.01,
    }, 'Bake AO'), 0);
    haveMaps = true;
  }
  if (p.bakeNormal) {
    applyInputs.normal_map = G.out(G.add('BakeNormalMapFromMesh', {
      low_poly: lowPoly, high_poly: retopo.dense,
      resolution: p.bakeResolution, cage_distance: 0.05, ignore_backfaces: true,
    }, 'Bake normal map'), 0);
    haveMaps = true;
  }

  let finalMesh = lowPoly;
  if (haveMaps) {
    // ApplyTextureToMesh needs a base colour even when only normal/AO were baked.
    if (!applyInputs.base_color) {
      applyInputs.base_color = G.out(G.add('EmptyImage', {
        width: p.textureResolution, height: p.textureResolution,
        batch_size: 1, color: 0xb4b4b4,
      }, 'Neutral base colour'), 0);
    }
    const applied = G.add('ApplyTextureToMesh', applyInputs, 'Apply texture');
    finalMesh = G.out(G.add('MeshSmoothNormals',
      { mesh: G.out(applied, 0), crease_angle: p.creaseAngle }, 'Final smooth'), 0);
  }

  // Write out every stage so the run can be inspected rather than taken on
  // trust: what each view contributed, where the gaps were, what was invented.
  if (p.saveIntermediates) {
    for (const it of intermediates) {
      const img = it.image !== undefined
        ? it.image
        : G.out(G.add('MaskToImage', { mask: it.mask }, 'Mask ' + it.name), 0);
      G.add('SaveImage', {
        images: img,
        filename_prefix: p.stepsPrefix + '/' + it.name,
      }, 'Save ' + it.name);
    }
  } else if (coverage && p.saveCoverage) {
    G.add('SaveImage', {
      images: G.out(G.add('MaskToImage', { mask: coverage }, 'Coverage to image'), 0),
      filename_prefix: p.filenamePrefix + '-coverage',
    }, 'Save coverage map');
  }

  // --- report + save --------------------------------------------------------
  const info = G.add('GetMeshInfo', { mesh: finalMesh }, 'Mesh info');
  const save = G.add('SaveGLB',
    { mesh: G.out(info, 0), filename_prefix: p.filenamePrefix }, 'Save GLB');

  // --- rigging --------------------------------------------------------------
  // Built after the plain SaveGLB on purpose. Rigging is the last thing that
  // happens after minutes of geometry and texture work, and ComfyUI aborts the
  // entire prompt when any node raises -- so a rig that could not be fitted used
  // to take the unrigged model down with it and leave the run with nothing. The
  // real guarantee is that the rig nodes never raise (they report failure through
  // the RIG object instead); emitting the save first only means ComfyUI reaches
  // it earlier, which also covers a hard crash such as an OOM.
  //
  // Fitted against `finalMesh` rather than the low-poly, and saved with that
  // same mesh. Skin weights are per-vertex, and both UnwrapMesh (which splits
  // vertices along chart seams) and MeshSmoothNormals (which splits them at
  // creases) change the vertex count, so rigging anything earlier would bind
  // weights to indices that no longer mean the same thing.
  buildRigging(G, p, finalMesh, applyInputs.base_color);

  return { prompt: G.json(), saveNode: save, infoNode: info };
}

// ============================================================ staged runs ===
/**
 * Split the pipeline into prompts that each fit on the GPU on their own.
 *
 * Held in one prompt, the shape model and the image model are both resident:
 * Hunyuan, SDXL and a ControlNet together are far past a 10 GB card, so ComfyUI
 * pages weights between GPU and CPU for the whole run. A measured pig took 23
 * minutes, almost none of it sampling -- eight turbo steps are seconds of
 * arithmetic.
 *
 * Split across prompts, each stage gets the card to itself, because ComfyUI can
 * free everything between prompts. It also makes the run resumable: the repaint
 * can be re-run against the same geometry without rebuilding the mesh.
 *
 *   geometry  shape model -> retopo -> bake your real art -> stage1.glb
 *   repaint   stage1.glb -> render missing angles -> SDXL + ControlNet -> pngs
 *   assemble  stage1.glb + real art + repainted views -> final bake -> rig
 *
 * Geometry crosses the boundary as a GLB, read back by the LoadMeshGLB node.
 * Nothing else needs to survive: the masks stage 3 wants are re-rendered from
 * the same mesh at the same camera, so they are identical by construction
 * rather than by agreement between two files.
 */
function missingAngles(p) {
  return (p.fitAngles ? VIEWS : VIEWS.filter((v) => !p.images[v]))
    .map((v) => ({ name: v, yaw: yawFor(v, p.swapSides) }));
}

function wantsRepaint(p) {
  return !!p.texture && p.geometry !== 'trellis2'
    && !!p.synthesizeViews && missingAngles(p).length > 0;
}

/** Load every supplied reference image, cut out and framed. */
function prepAll(G, p) {
  const bgModelId = p.removeBackground
    ? G.add('LoadBackgroundRemovalModel', { bg_removal_name: BG_MODEL }, 'BiRefNet')
    : null;
  const prepped = {};
  for (const v of VIEWS) {
    if (!p.images[v]) continue;
    prepped[v] = prepImage(G, p.images[v], {
      label: v, removeBackground: p.removeBackground, bgModelId: bgModelId,
      size: 1024, padFactor: p.padFactor, background: p.hunyuanBackground,
    });
  }
  if (!prepped.front) throw new Error('A front image is required.');
  return { bgModelId: bgModelId, prepped: prepped };
}

function saveSteps(G, p, intermediates) {
  if (!p.saveIntermediates) return;
  for (const it of intermediates) {
    const img = it.image !== undefined
      ? it.image
      : G.out(G.add('MaskToImage', { mask: it.mask }, 'Mask ' + it.name), 0);
    G.add('SaveImage', { images: img, filename_prefix: p.stepsPrefix + '/' + it.name },
          'Save ' + it.name);
  }
}

/** Stage 1: geometry and a first bake of the art you actually supplied. */
function buildGeometryStage(p) {
  const G = new Graph();
  const { prepped } = prepAll(G, p);

  const denseMesh = buildHunyuan(G, p, prepped);
  const retopo = buildRetopo(G, p, denseMesh, false);
  const lowPoly = retopo.lowPoly;

  const proj = buildProjectionTexture(G, p, prepped, lowPoly, denseMesh);
  const intermediates = [];
  for (const fit of (proj.fits || [])) {
    intermediates.push({ image: G.out(fit.node, 3), name: 'step1-anglefit-' + fit.name });
  }
  intermediates.push({ image: proj.baseColor, name: 'step2-atlas-real' });
  saveSteps(G, p, intermediates);

  // Coverage has to cross the stage boundary under a stable name: it is what
  // tells stage 2 which texels are approximated, and therefore which pixels the
  // image model is allowed to repaint. Without it the gap mask comes back empty
  // and the repaint pass silently changes nothing.
  G.add('SaveImage', {
    images: G.out(G.add('MaskToImage', { mask: proj.coverage }, 'Coverage to image'), 0),
    filename_prefix: p.stepsPrefix + '/coverage',
  }, 'Save coverage');

  // Carries the partial texture, so stage 2's renders already show your art.
  const applied = G.add('ApplyTextureToMesh',
    { mesh: lowPoly, base_color: proj.baseColor }, 'Apply partial texture');
  const save = G.add('SaveGLB',
    { mesh: G.out(applied, 0), filename_prefix: p.filenamePrefix + '-stage1' },
    'Save stage 1 mesh');

  // The dense mesh rides along only as an occlusion reference for the final
  // bake; without it a low-poly silhouette shadows itself along grazing edges.
  if (p.useReferenceMesh) {
    G.add('SaveGLB', { mesh: denseMesh, filename_prefix: p.filenamePrefix + '-dense' },
          'Save dense reference');
  }
  return { prompt: G.json(), saveNode: save };
}

/** Stage 2: paint the angles no reference art covers. No 3D model is loaded. */
function buildRepaintStage(p, prev) {
  const G = new Graph();
  const loaded = G.add('LoadMeshGLB',
    { filename: prev.stage1Glb, directory: 'output' }, 'Load stage 1 mesh');
  const mesh = G.out(loaded, 0);

  // Coverage from stage 1, back as a mask. This is what marks the region the
  // image model may repaint; everything your real art reached stays untouched.
  const covImg = G.add('LoadImage', { image: prev.coverage }, 'Coverage from stage 1');
  const coverage = G.out(G.add('ImageToMask',
    { image: G.out(covImg, 0), channel: 'red' }, 'Coverage to mask'), 0);

  const styleView = VIEWS.find((v) => p.images[v]) || 'front';
  let styleRef = null;
  if (p.synthBackend === 'kontext') {
    styleRef = G.out(G.add('LoadImage', { image: p.images[styleView] },
                           'Style reference'), 0);
  }

  const synth = buildViewSynthesis(G, p, mesh, coverage, null, missingAngles(p), styleRef);
  const saves = synth.saves.slice();
  // The repainted views are what stage 3 consumes, so they are saved under a
  // stable name rather than as numbered step images.
  for (const m of synth.made) {
    G.add('SaveImage', {
      images: m.image,
      filename_prefix: p.stepsPrefix + '/repaint-' + m.name,
    }, 'Save repaint ' + m.name);
  }
  saveSteps(G, p, saves);
  return { prompt: G.json(), saveNode: null };
}

/** Stage 3: final bake from real plus repainted views, then rig and save. */
function buildAssembleStage(p, prev) {
  const G = new Graph();
  const { prepped } = prepAll(G, p);

  const loaded = G.add('LoadMeshGLB',
    { filename: prev.stage1Glb, directory: 'output' }, 'Load stage 1 mesh');
  const lowPoly = G.out(loaded, 0);
  const denseMesh = prev.denseGlb
    ? G.out(G.add('LoadMeshGLB', { filename: prev.denseGlb, directory: 'output' },
                  'Load dense reference'), 0)
    : lowPoly;

  // Re-render each repainted angle purely for its silhouette mask. Re-deriving
  // it from the same mesh at the same camera is exact; carrying a mask file
  // across stages would only be as good as the two agreeing.
  let views = null;
  for (const a of missingAngles(p)) {
    const file = prev.repaints && prev.repaints[a.name];
    if (!file) continue;
    const img = G.add('LoadImage', { image: file }, 'Repainted ' + a.name);
    const render = G.add('ProjectionViewRender', {
      mesh: lowPoly, yaw: a.yaw, pitch: p.viewPitch,
      resolution: p.synthResolution, background: p.synthBackground, shade: 0.0,
    }, 'Mask for ' + a.name);
    const inputs = {
      image: G.out(img, 0),
      mask: G.out(render, 1),
      yaw: a.yaw, pitch: p.viewPitch, scale: 1.0,
      offset_x: 0.0, offset_y: 0.0,
      weight: p.syntheticWeight,
    };
    if (views) inputs.views = views;
    views = G.out(G.add('ProjectionViewAdd', inputs,
                        'View: ' + a.name + ' (repainted)'), 0);
  }

  const proj = buildProjectionTexture(G, p, prepped, lowPoly, denseMesh, views);
  const intermediates = [{ image: proj.baseColor, name: 'step6-atlas-final' }];

  const applyInputs = { mesh: lowPoly, base_color: proj.baseColor };
  if (p.bakeNormal) {
    applyInputs.normal_map = G.out(G.add('BakeNormalMapFromMesh', {
      low_poly: lowPoly, high_poly: denseMesh,
      resolution: p.bakeResolution, cage_distance: 0.05, ignore_backfaces: true,
    }, 'Bake normal map'), 0);
  }
  if (p.bakeAO) {
    applyInputs.occlusion = G.out(G.add('BakeAmbientOcclusion', {
      low_poly: lowPoly, high_poly: denseMesh,
      resolution: p.bakeResolution, samples: p.aoSamples,
      max_distance: 0.71, strength: 1.0, bias: 0.01,
    }, 'Bake AO'), 0);
  }

  const applied = G.add('ApplyTextureToMesh', applyInputs, 'Apply texture');
  const finalMesh = G.out(G.add('MeshSmoothNormals',
    { mesh: G.out(applied, 0), crease_angle: p.creaseAngle }, 'Final smooth'), 0);

  saveSteps(G, p, intermediates);

  const info = G.add('GetMeshInfo', { mesh: finalMesh }, 'Mesh info');
  const save = G.add('SaveGLB',
    { mesh: G.out(info, 0), filename_prefix: p.filenamePrefix }, 'Save GLB');

  buildRigging(G, p, finalMesh, proj.baseColor);
  return { prompt: G.json(), saveNode: save, infoNode: info };
}

/**
 * Stage list for a set of parameters. One entry when nothing needs repainting,
 * three when it does.
 */
function buildStages(params) {
  const p = params;
  if (!wantsRepaint(p)) {
    return [{ name: 'all', split: false, build: () => buildPrompt(p) }];
  }
  return [
    { name: 'geometry', split: true, build: () => buildGeometryStage(p) },
    { name: 'repaint', split: true, build: (prev) => buildRepaintStage(p, prev) },
    { name: 'assemble', split: true, build: (prev) => buildAssembleStage(p, prev) },
  ];
}

module.exports = {
  buildPrompt: buildPrompt,
  buildStages: buildStages,
  wantsRepaint: wantsRepaint,
  missingAngles: missingAngles,
  buildImagePrompt: buildImagePrompt,
  VIEWS: VIEWS,
  MODELS: {
    CKPT_MV: CKPT_MV, CKPT_21: CKPT_21, T2_UNET: T2_UNET,
    T2_SHAPE_VAE: T2_SHAPE_VAE, T2_TEX_VAE: T2_TEX_VAE,
    T2_CLIPVISION: T2_CLIPVISION, BG_MODEL: BG_MODEL,
  },
};
