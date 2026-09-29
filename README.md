# L3DGen — reference art to a rigged, textured game model

A local pipeline that turns hand-drawn or AI-generated character art into a
low-poly, **textured**, **rigged** 3D model, driven from a browser panel that
talks to [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

The distinguishing idea is that **your own artwork ends up on the model**. The
shape model supplies geometry; the texture comes from projecting your reference
images onto that geometry from the exact camera each one was drawn from, rather
than from a generated colour field that only resembles your art.

```
  reference art  ->  shape model  ->  retopo + UV  ->  project your art  ->  rig  ->  .glb / .fbx
   front/left/                                          onto the mesh       Mixamo
   back/right                                                               or built-in
```

---

## Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Setup](#setup)
- [Models to download](#models-to-download)
- [Running it](#running-it)
- [Guide: your first model](#guide-your-first-model)
- [Guide: texturing sides you never drew](#guide-texturing-sides-you-never-drew)
- [Guide: rigging](#guide-rigging)
- [Guide: animation](#guide-animation)
- [Guide: exporting to Unity](#guide-exporting-to-unity)
- [Performance and VRAM](#performance-and-vram)
- [Custom nodes reference](#custom-nodes-reference)
- [Scripts](#scripts)
- [Troubleshooting](#troubleshooting)
- [Licence](#licence)

---

## What it does

**Geometry** from one of three sources:

| Source | Input | Notes |
| --- | --- | --- |
| Hunyuan3D 2.0-MV | front / left / back / right | Multiview, best silhouette fidelity |
| Hunyuan3D 2.1 | front only | Single image |
| TRELLIS.2 | front only | Also the only local source of colour voxels |

**Texture** by projection. Each reference image is projected onto the finished
mesh from its own camera; every texel takes colour from whichever view sees that
part of the surface most directly. Surfaces no view reached are approximated on
the mesh surface, optionally corrected toward a palette taken from your own art.

**Rigging**, two ways:

- **Built in** — fits the 25-joint `mixamorig:*` humanoid to the mesh and skins
  it. No download, works offline, exports a skinned GLB.
- **Via Mixamo** — export FBX, rig on mixamo.com, load the result back and the
  pipeline reattaches your textures. Better on humanoids.

**Animation** — retarget any Mixamo clip onto the rigged model and play it in
the panel's viewer.

**Export** — GLB, FBX with embedded textures, or a zip of FBX + `.fbm` texture
folder for Unity.

---

## Requirements

| | Version | Why |
| --- | --- | --- |
| **ComfyUI** | recent (with native Hunyuan3D / TRELLIS.2 nodes) | runs the models |
| **Node.js** | 18+ | the control panel server (zero dependencies) |
| **Python** | ComfyUI's own venv | custom nodes; needs `numpy`, `scipy`, `torch`, `Pillow` |
| **Blender** | 3.6+, 4.x tested | FBX conversion and animation retargeting |
| **GPU** | 10 GB VRAM tested | see [Performance and VRAM](#performance-and-vram) |

`numpy`, `torch` and `Pillow` already ship with ComfyUI. **`scipy` may not** —
the rigger and the surface-fill need it:

```bash
# from ComfyUI's python
<comfyui>/.venv/Scripts/python.exe -m pip install scipy
```

Blender is optional: without it the pipeline still generates and rigs models,
but FBX export, the Mixamo round-trip and animation retargeting are disabled.
The panel detects this and greys out the controls rather than failing mid-run.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/DREWtheBAWS/L3DGen.git
cd L3DGen
```

### 2. Install the custom nodes

Two ComfyUI node packages live in `comfyui/custom_nodes/`. Copy or symlink both
into your ComfyUI installation:

```bash
# Windows (PowerShell, as administrator for symlinks)
New-Item -ItemType SymbolicLink -Path "<comfyui>\custom_nodes\comfyui_multiview_projection" -Target "<repo>\comfyui\custom_nodes\comfyui_multiview_projection"
New-Item -ItemType SymbolicLink -Path "<comfyui>\custom_nodes\comfyui_auto_rig"             -Target "<repo>\comfyui\custom_nodes\comfyui_auto_rig"

# macOS / Linux
ln -s "$PWD/comfyui/custom_nodes/comfyui_multiview_projection" <comfyui>/custom_nodes/
ln -s "$PWD/comfyui/custom_nodes/comfyui_auto_rig"             <comfyui>/custom_nodes/
```

Copying works just as well if symlinks are awkward — you then re-copy after
pulling changes.

**Restart ComfyUI.** Custom nodes only load at startup. The panel checks for
them and tells you if a restart is needed.

### 3. Download the models

See [Models to download](#models-to-download). Nothing is bundled here.

### 4. Start the panel

```bash
npm start          # or: node server.js
```

Open <http://127.0.0.1:8189>. ComfyUI must already be running; the panel finds
it by sweeping ports 8188–8199, so it keeps working when ComfyUI Desktop moves.

---

## Models to download

None of these are in the repository. Put them where ComfyUI can see them — its
own `models/` tree, or a separate drive registered through
`extra_model_paths.yaml`.

### Required

| File | Folder | Purpose |
| --- | --- | --- |
| `hunyuan3d-dit-v2-mv_fp16.safetensors` | `checkpoints/` | multiview shape model |
| `birefnet.safetensors` | `background_removal/` | cutting the subject out |

### Optional, by feature

| File | Folder | Enables |
| --- | --- | --- |
| `hunyuan_3d_v2.1.safetensors` | `checkpoints/` | single-image geometry |
| `trellis_2_int8_convrot.safetensors` | `diffusion_models/` | TRELLIS.2 geometry + colour |
| `trellis_2_shape_vae_bf16.safetensors` | `vae/` | TRELLIS.2 |
| `trellis_2_texture_vae_bf16.safetensors` | `vae/` | TRELLIS.2 |
| `dino_v3_L_naf_fp32.safetensors` | `clip_vision/` | TRELLIS.2 |
| any SDXL checkpoint | `checkpoints/` | painting unreferenced sides |
| `controlnet-union-sdxl-1.0` promax | `controlnet/` | locking that painting to the geometry |

`scripts/download-models.sh`, `download-image-models.sh` and
`download-controlnet.sh` fetch these; they are plain `curl` and resume safely.
Read them before running — they write several gigabytes.

The panel's header shows which models ComfyUI can currently see, polled live.

### Storing models on another drive

ComfyUI reads `extra_model_paths.yaml` from its root:

```yaml
models_on_d:
  base_path: 'D:\ComfyUI-Models'
  checkpoints: 'checkpoints/'
  controlnet: 'controlnet/'
  vae: 'vae/'
  clip_vision: 'clip_vision/'
  diffusion_models: 'diffusion_models/'
  background_removal: 'background_removal/'
```

A mechanical hard drive is fine. The cost is paid once when weights are read
into memory, not during inference.

---

## Running it

The panel has two tabs:

- **Pipeline** — the full art-to-model run.
- **Image lab** — image-to-image on its own, for testing prompts and
  checkpoints without spending a 3D run on it.

---

## Guide: your first model

1. **Name your model.** The field at the top names every file the run writes,
   so outputs do not pile up under `pipeline_0000N`.

2. **Add reference views.** Drop images into the *front*, *left*, *back*,
   *right* slots. Only front is required. Click the **×** on a slot to remove an
   image.

   Views should be a level turnaround of the same subject at the same scale. If
   your art is isometric or three-quarter, tick **My art is at odd angles** and
   each image's camera angle is recovered from the mesh silhouette instead.

3. **Pick geometry.** `Hunyuan3D 2.0 multiview` uses all four views and is the
   default.

4. **Set the polygon budget.** 5k–50k triangles. This is a *maximum* — heavy
   decimation often lands under it.

5. **Generate.** Roughly 60–90 s for geometry and texture on a 10 GB card.

The finished model appears in the viewer with triangle count, vertex count and
bone count. The steps gallery below shows every intermediate: what each view
contributed, where the gaps were, what was invented.

### If the sides come out mirrored

Turnaround sheets are not consistent about whether "left" means the character's
left or the camera's. Tick **Swap left / right** if the side art lands on the
wrong side.

The convention here: world is Y-up, at yaw 0 the camera is on +Z, so the subject
faces +Z. Its own right hand points along `cross(+Z, +Y) = -X`, which puts **yaw
90 on the subject's left**.

---

## Guide: texturing sides you never drew

With fewer than four views, part of the model has no reference art. Three
mechanisms, cheapest first — and they compose.

### 1. Symmetry (free, exact)

Tick **Subject is symmetric**. A side view mirrored horizontally *is* the
opposite side view: at yaw 90 the camera sees the subject's left with the face
toward frame-left, and yaw 270 sees the right with the face toward frame-right,
which is precisely the mirror.

So for a symmetric subject this recovers the missing side **exactly**, rather
than approximating it. The mirrored view is weighted just under a real one, so
genuine art always wins where they overlap.

Do not use it on asymmetric characters — a satchel on one hip gets mirrored onto
the other side where something else belongs.

### 2. Palette-corrected fill (free)

Surfaces no view reached are filled by blending across the mesh surface. That
produces gradients, and the midpoint of two of your colours is a hue that
appears in none of your references — it reads as mud.

**Correct fill to source palette** clusters the colours your art actually put on
the model and pulls the filled areas onto them, then re-lights each to the
texel's own brightness so shading survives. Measured on a synthetic test, hue
error dropped from 11.94° to 1.22° with luminance spread preserved. Try 8–16
colours. Areas your art reaches are never altered.

Uncheck **Keep shading** for flat cel-shading.

### 3. Depth-locked repaint (needs SDXL + ControlNet)

Filling gaps gives plausible *colour*. It cannot invent *content* — if the shape
model puts a satchel on a side you never drew, nothing in the existing colours
can produce leather and buckles.

Set **Missing views** to *After 3D* and **Lock to geometry** to your ControlNet:

- the mesh is rendered at the missing angle, orthographically, at an exact yaw;
- depth and normal maps come off that **same rasterisation**, so the
  conditioning cannot be misaligned with the geometry;
- the image model repaints the gap region at high denoise — high enough to
  invent detail rather than re-tint;
- the result reprojects with no alignment error, because the camera was never in
  question.

Without the structural lock there is no good denoise setting: low only
recolours, high drifts off the silhouette and no longer reprojects.

> **Note on generated turnarounds.** The *Before 3D* strategy invents the
> missing views before any geometry exists. Those images feed the shape model
> but are **never** used for texture: the bake needs an exact yaw and an
> orthographic camera, and a generative model gives neither.

---

## Guide: rigging

### Built-in auto-rig

Tick **Auto-rig as a Mixamo humanoid**. A second, skinned GLB is written
alongside the plain one.

It fits the 25-joint `mixamorig:*` hierarchy by working on the mesh's own
connectivity: geodesic farthest-point sampling finds the extremities whatever
the pose, and shortest paths back to the body trace the limbs. Learned riggers
predict *a* skeleton; Mixamo clips bind strictly by joint name, so producing
*that* hierarchy is the whole point.

Check `step7-rig-skeleton` and `step7-rig-weights` in the steps gallery before
relying on it. The report gives a confidence score and names anything odd about
the proportions.

Requires a roughly humanoid, roughly A- or T-posed mesh. Non-humanoids
(quadrupeds, spiders) are not supported.

### Hybrid: rig on Mixamo, keep your textures

Better results on humanoids. Three buttons under the viewer:

1. **Export .fbx** — sends the *unrigged* mesh, which is what Mixamo wants.
2. **Open Mixamo** — upload it, place the markers, let it rig.
3. **Load rigged .fbx** — download **with skin, FBX, no animation**, then load
   it back. Your atlas is reattached automatically.

Mixamo strips the material but preserves the UV layout, so reattaching is a
material assignment rather than a re-bake — exact, not approximate. The panel
verifies this on every round-trip and prints the UV drift; if Mixamo ever
changes, you will see it rather than get a quietly wrong model.

---

## Guide: animation

1. Put Mixamo `.fbx` clips in `animations/` (see the README there).
2. Pick one under **Mixamo clip** and press **Apply to rig**.
3. It plays in the viewer, with pause and a scrubber.

This is a real retarget, not a bone-name copy. A Mixamo clip stores rotations
against *Mixamo's* rest pose; your model's rest pose is whatever it was
generated as. The transfer goes through world space — the source bone's rotation
relative to its own rest, applied to your bone's rest, parents first — which is
why the character does not fold up.

---

## Guide: exporting to Unity

Three download buttons:

| Button | Contents | Use |
| --- | --- | --- |
| **.glb** | mesh, texture, skeleton | web, Godot, three.js |
| **.fbx** | same, texture embedded | general |
| **.zip (Unity)** | FBX + `.fbm` texture folder | **Unity** |

**Use the zip for Unity.** Unzip it into your Assets folder and keep the `.fbm`
folder beside the FBX. Embedded textures depend on the importer choosing to
extract them; loose textures in a sibling `.fbm` folder are resolved by
everything, with nothing to click.

The FBX is exported Y-up and metre-scaled with an identity root transform.
Mixamo's own downloads are centimetre-scaled and Z-up, and that gets baked out
on the way through, so models do not arrive sideways or a hundred times too
small.

---

## Performance and VRAM

Measured on an RTX 3080 (10 GB):

| Run | Time |
| --- | --- |
| Geometry + projection texture + rig | ~60 s |
| Same, plus depth-locked repaint, one prompt | 1386 s |
| Same, **split into three prompts** | ~200 s |

The difference is not compute. Held in one prompt, the shape model, SDXL and a
ControlNet are all resident — about 7.3 GB on a 10 GB card — so ComfyUI pages
weights between GPU and CPU for the entire run.

**The panel splits automatically** when a run needs repainting, into three
prompts that each get the card to themselves:

```
geometry   shape model -> retopo -> bake your real art -> stage1.glb
repaint    stage1.glb -> render missing angles -> SDXL + ControlNet -> pngs
assemble   stage1.glb + real art + repainted views -> final bake -> rig
```

Progress shows the current stage. A side effect is that the run becomes
resumable: re-running the repaint against unchanged geometry costs seconds.

### Low VRAM mode

Caps texture, bake and synthesis resolutions, disables ambient occlusion, and
limits the TRELLIS upsample. Keeps a run inside roughly 6 GB.

**TRELLIS.2 specifically:** its DiT is ~5 GB and stays resident through the
shape VAE decode, which ComfyUI will not evict. At upsample 1536 the decode asks
for 3.1 GiB against ~2 GiB free and dies. 1024 is the largest that fits
reliably on 10 GB, and low-VRAM mode enforces it.

---

## Custom nodes reference

### `comfyui_multiview_projection`

| Node | Purpose |
| --- | --- |
| Add Projection View | one reference image plus its camera angle; chainable |
| Bake Texture From Views | mesh + views → base colour atlas + coverage mask |
| Render Projection View | orthographic render at an exact yaw; also depth and normal |
| Fit View Angle | recover the camera angle of off-axis reference art |
| Preview Projection Views | check that art lines up with the geometry |
| Load Mesh From GLB | read a GLB back into a MESH — what makes staged runs possible |

### `comfyui_auto_rig`

| Node | Purpose |
| --- | --- |
| Auto Rig Humanoid | fit the Mixamo skeleton and skin it |
| Preview Rig | draw the skeleton and weights over the mesh |
| Save Rigged GLB | write a skinned GLB with `mixamorig:*` joints |

Both are free of ComfyUI imports in their maths modules (`geometry.py`,
`rig_fit.py`), so the algorithms can be exercised standalone — see
[Scripts](#scripts).

---

## Scripts

Offline tools. The Python ones run under **ComfyUI's** interpreter; the Blender
ones under Blender's.

| Script | What it does |
| --- | --- |
| `validate-graph.js` | check built graphs against the loaded node definitions — catches a stale ComfyUI before a run does |
| `test-autorig.py` | rig-fitter test suite on synthetic humanoids |
| `rig-glb.py` | fit the rig to any `.glb` in seconds, with a preview image |
| `render-glb.py` | render a `.glb` from several angles, no GPU |
| `retarget-mixamo.py` | Blender: retarget a Mixamo clip onto a rigged GLB |
| `render-anim.py` | Blender: render frames of an animated GLB |
| `glb-to-fbx.py` | Blender: GLB → FBX for Mixamo |
| `apply-textures.py` | Blender: reattach textures to a Mixamo-rigged FBX |
| `verify-projection.py` | check the camera convention and framing maths |

```bash
node scripts/validate-graph.js
"<comfyui>/.venv/Scripts/python.exe" scripts/test-autorig.py
"<blender>/blender.exe" --background --python scripts/glb-to-fbx.py -- in.glb out.fbx --embed
```

---

## Troubleshooting

**"projection nodes not loaded — restart ComfyUI"**
Custom nodes load only at startup. Restart ComfyUI. If it persists, check they
are in `custom_nodes/` and that `scipy` is installed in ComfyUI's venv.

**A run fails partway with a node error**
`node scripts/validate-graph.js` compares every graph the pipeline can build
against what ComfyUI currently has loaded, and names the mismatch offline. It is
the fastest way to spot a custom node changed on disk but not reloaded.

**Out of memory**
Turn on **Low VRAM mode**. If it is TRELLIS.2, also drop the upsample resolution
to 1024.

**The texture is smeared down the sides**
Too few views. Add a side view, turn on **Subject is symmetric** if it applies,
or raise **Reject grazing samples**.

**The model imports into Unity untextured**
Use the **.zip (Unity)** download and keep the `.fbm` folder next to the FBX.

**Blender features greyed out**
Blender was not found. Set the `BLENDER` environment variable to your
`blender.exe`.

---

## Licence

**GPL-3.0.** The ComfyUI custom nodes import ComfyUI internals, and ComfyUI is
GPL-3.0, so this inherits it.

The control panel (`server.js`, `lib/`, `public/`) talks to ComfyUI only over
HTTP and imports nothing from it, so it is separable if you ever want to
relicense that part on its own.

Nothing in this repository is redistributed third-party content: no model
weights, no Mixamo assets, no reference art. Those come from their own sources
under their own terms.
