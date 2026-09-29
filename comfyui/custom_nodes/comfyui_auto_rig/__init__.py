"""
Automatic humanoid rigging for ComfyUI, targeting Mixamo's bone names.

Nodes
-----
  Auto Rig Humanoid   mesh -> RIG (fitted skeleton + skin weights + a report)
  Preview Rig         draws the fitted skeleton over the mesh, so a bad fit is
                      visible before anything is exported
  Save Rigged GLB     writes a skinned .glb with mixamorig:* joint names

Why this exists
---------------
The only rigging nodes shipping with ComfyUI (Meshy, Tripo) are paid cloud APIs.
Local learned auto-riggers predict *a* skeleton rather than *the* Mixamo one,
which is the thing that actually matters: Mixamo animation tracks bind strictly
by joint name, so an arbitrary hierarchy cannot play them without a hand-built
retarget map.

On rest poses
-------------
A Mixamo clip stores local joint rotations against Mixamo's own rest pose. This
rig's rest pose is the mesh exactly as generated, so clips should be applied
through a retargeter that reconciles the two -- Unity's Humanoid avatar, Blender
(Rokoko / Auto-Rig Pro), Godot's retarget profile. Feeding raw local rotations
straight in without retargeting will look wrong, and no auto-rigger can avoid
that; generating the character in a T-pose keeps the two rest poses closest.
"""

import logging
import os

import numpy as np
import torch

import folder_paths
from comfy_extras.nodes_save_3d import get_mesh_batch_item

from .rig_fit import MIXAMO_JOINTS, fit_humanoid, skin_mesh
from .glb_skin import save_rigged_glb

RIG_TYPE = "HUMANOID_RIG"

log = logging.getLogger(__name__)


def _failed_rig(reason):
    """A RIG that carries a failure instead of raising one.

    Downstream nodes check `ok` and skip; the run continues and the unrigged
    model is still written.
    """
    return {"ok": False, "error": str(reason), "confidence": 0.0,
            "notes": [str(reason)], "vertex_count": -1,
            "joints": np.zeros((0, 3), np.float32),
            "parents": np.zeros((0,), np.int32), "names": [],
            "joint_idx": np.zeros((0, 4), np.uint16),
            "joint_w": np.zeros((0, 4), np.float32),
            "height": 0.0, "centre_x": 0.0}


def _as_numpy(mesh, index=0):
    verts, faces, _colors, uvs, _normals = get_mesh_batch_item(mesh, index)
    v = verts.detach().cpu().numpy().astype(np.float64)
    f = faces.detach().cpu().numpy().astype(np.int64)
    u = uvs.detach().cpu().numpy().astype(np.float32) if uvs is not None else None
    return v, f, u


class AutoRigHumanoid:
    """Fit the Mixamo skeleton to a mesh and compute skin weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Y-up, subject facing +Z, roughly A- or "
                                             "T-posed. This is what the pipeline's own "
                                             "geometry stage already produces."}),
                "facing": ("COMBO", {"default": "+Z", "options": ["+Z", "-Z"],
                                     "tooltip": "Which way the character looks. It decides "
                                                "which side gets the Left bones: for a "
                                                "subject facing +Z with up +Y, its own left "
                                                "is +X."}),
                "symmetrize": ("BOOLEAN", {"default": True,
                                           "tooltip": "Mirror-average the left and right "
                                                      "joints. An asymmetric rig makes every "
                                                      "clip look lopsided even on a "
                                                      "symmetric mesh."}),
                "smoothing": ("INT", {"default": 48, "min": 0, "max": 200,
                                      "tooltip": "Diffusion passes for the skin weights. "
                                                 "Higher = softer joint bends. 0 gives rigid "
                                                 "nearest-bone binding."}),
                "max_influences": ("INT", {"default": 4, "min": 1, "max": 4,
                                           "tooltip": "Bones per vertex. Most engines cap "
                                                      "this at 4."}),
                "min_confidence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "Refuse the rig below this fit "
                                                        "confidence rather than emitting a "
                                                        "plausible-looking but wrong "
                                                        "skeleton. 0 never refuses."}),
            },
        }

    RETURN_TYPES = (RIG_TYPE, "STRING")
    RETURN_NAMES = ("rig", "report")
    FUNCTION = "rig"
    CATEGORY = "3d/rigging"

    def rig(self, mesh, facing, symmetrize, smoothing, max_influences, min_confidence):
        # Rigging is the last stage of a pipeline that has already spent minutes
        # building and texturing a mesh, and ComfyUI aborts the whole prompt when
        # any node raises -- which would take the plain SaveGLB down with it and
        # leave the run with no model at all. A rig that cannot be fitted is a
        # result, not an error, so every failure is reported through the rig
        # itself and nothing here is allowed to propagate.
        try:
            verts, faces, _uv = _as_numpy(mesh)
            if len(faces) == 0:
                raise ValueError("the mesh has no faces")

            fit = fit_humanoid(verts, faces,
                               forward_axis=1.0 if facing == "+Z" else -1.0,
                               symmetrize=symmetrize)
            if fit["confidence"] < min_confidence:
                raise ValueError(
                    "fit confidence %.2f is below the %.2f you asked for (%s)"
                    % (fit["confidence"], min_confidence,
                       "; ".join(fit["notes"]) or "the mesh may not be humanoid"))

            ji, jw = skin_mesh(verts, faces, fit["joints"], fit["parents"],
                               iterations=smoothing, max_influences=max_influences)
        except Exception as exc:
            log.exception("[auto_rig] fit failed; the unrigged mesh is unaffected")
            report = ("Rigging failed: %s\nThe unrigged GLB is still written. "
                      "Check that the mesh is a symmetric humanoid, Y-up and facing %s."
                      % (exc, facing))
            return (_failed_rig(str(exc)), report)

        rig = {"ok": True,
               "joints": fit["joints"], "parents": fit["parents"], "names": fit["names"],
               "joint_idx": ji, "joint_w": jw, "vertex_count": len(verts),
               "confidence": fit["confidence"], "notes": fit["notes"],
               "height": fit["height"], "centre_x": fit["centre_x"]}

        lines = ["Fitted %d Mixamo joints to %d vertices." % (len(fit["joints"]), len(verts)),
                 "Confidence: %.2f" % fit["confidence"]]
        lines += ["  warning: " + n for n in fit["notes"]]
        if not fit["notes"]:
            lines.append("  proportions are within the humanoid range.")
        lines.append("Rest pose is the mesh as-is; apply Mixamo clips through a "
                     "retargeter (Unity Humanoid, Blender, Godot).")
        report = "\n".join(lines)
        log.info("[auto_rig] %s", report.replace("\n", " | "))
        return (rig, report)


class PreviewRig:
    """Render the fitted skeleton over the mesh silhouette.

    A rig that is subtly wrong -- a knee inside the shin, an elbow at the wrist --
    costs far more time downstream than it takes to look at it here, and joint
    positions are not something a numeric report makes obvious.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "rig": (RIG_TYPE,),
                "resolution": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 64}),
                "view": ("COMBO", {"default": "front+side",
                                   "options": ["front", "side", "front+side"]}),
                "mode": ("COMBO", {"default": "skeleton",
                                   "options": ["skeleton", "weights"],
                                   "tooltip": "skeleton: bones over the silhouette. "
                                              "weights: colour each vertex by the bone that "
                                              "owns it, which is how weight bleed shows up."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "3d/rigging"

    # Distinct hues so neighbouring bones never share a colour in weights mode.
    _PALETTE = np.array([
        [0.95, 0.26, 0.21], [0.91, 0.12, 0.39], [0.61, 0.15, 0.69], [0.40, 0.23, 0.72],
        [0.25, 0.32, 0.71], [0.13, 0.59, 0.95], [0.01, 0.66, 0.96], [0.00, 0.74, 0.83],
        [0.00, 0.59, 0.53], [0.30, 0.69, 0.31], [0.55, 0.76, 0.29], [0.80, 0.86, 0.22],
        [1.00, 0.92, 0.23], [1.00, 0.76, 0.03], [1.00, 0.60, 0.00], [1.00, 0.44, 0.26],
        [0.47, 0.33, 0.28], [0.62, 0.62, 0.62], [0.38, 0.49, 0.55], [0.90, 0.45, 0.45],
        [0.45, 0.90, 0.65], [0.45, 0.65, 0.90], [0.90, 0.65, 0.45], [0.65, 0.45, 0.90],
        [0.75, 0.90, 0.45],
    ], dtype=np.float32)

    def _draw(self, verts, rig, res, axis, mode):
        img = np.full((res, res, 3), 0.10, dtype=np.float32)
        joints = np.asarray(rig["joints"], dtype=np.float64)

        # One orthographic frame covering mesh and skeleton together, so a joint
        # that escaped the body is visibly outside it rather than cropped away.
        allp = np.concatenate([verts, joints], axis=0)
        h, v = (0, 1) if axis == "front" else (2, 1)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ctr = 0.5 * (lo + hi)
        half = 0.55 * max(hi[h] - lo[h], hi[1] - lo[1], 1e-6)

        def to_px(p):
            x = (p[..., h] - ctr[h]) / half * 0.5 + 0.5
            y = 0.5 - (p[..., 1] - ctr[1]) / half * 0.5
            return (np.clip(x, 0, 1) * (res - 1)).astype(np.int32), \
                   (np.clip(y, 0, 1) * (res - 1)).astype(np.int32)

        px, py = to_px(verts)
        colour = (self._PALETTE[np.asarray(rig["joint_idx"])[:, 0] % len(self._PALETTE)]
                  if mode == "weights" else np.full((len(verts), 3), 0.42, np.float32))
        # Splat each vertex over a small block. A one-pixel point cloud reads as
        # noise at preview sizes and the silhouette is the thing being judged.
        s = max(1, res // 400)
        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                img[np.clip(py + dy, 0, res - 1), np.clip(px + dx, 0, res - 1)] = colour

        jx, jy = to_px(joints)
        parents = np.asarray(rig["parents"])
        for i, p in enumerate(parents):
            if int(p) < 0:
                continue
            n = max(int(abs(jx[i] - jx[p])), int(abs(jy[i] - jy[p])), 1)
            t = np.linspace(0.0, 1.0, n * 2 + 1)
            lx = np.round(jx[p] + (jx[i] - jx[p]) * t).astype(np.int32)
            ly = np.round(jy[p] + (jy[i] - jy[p]) * t).astype(np.int32)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    img[np.clip(ly + dy, 0, res - 1), np.clip(lx + dx, 0, res - 1)] = \
                        np.array([1.0, 0.85, 0.10], dtype=np.float32)
        r = max(2, res // 220)
        for i in range(len(joints)):
            y0, y1 = max(0, jy[i] - r), min(res, jy[i] + r + 1)
            x0, x1 = max(0, jx[i] - r), min(res, jx[i] + r + 1)
            img[y0:y1, x0:x1] = np.array([0.15, 1.0, 0.55], dtype=np.float32)
        return img

    def preview(self, mesh, rig, resolution, view, mode):
        # A preview must never abort the run either; a blank frame says "no rig"
        # perfectly well and the rest of the pipeline keeps its outputs.
        try:
            verts, _faces, _uv = _as_numpy(mesh)
            if not rig.get("ok", False) or len(rig["joints"]) == 0:
                blank = np.full((resolution, resolution, 3), 0.10, dtype=np.float32)
                return (torch.from_numpy(blank)[None, ...],)
            axes = ["front", "side"] if view == "front+side" else [view]
            panes = [self._draw(verts, rig, resolution, a, mode) for a in axes]
            img = np.concatenate(panes, axis=1) if len(panes) > 1 else panes[0]
        except Exception:
            log.exception("[auto_rig] preview failed")
            img = np.full((resolution, resolution, 3), 0.10, dtype=np.float32)
        return (torch.from_numpy(img)[None, ...],)


class SaveRiggedGLB:
    """Write a skinned GLB with mixamorig:* joint names."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "rig": (RIG_TYPE,),
                "filename_prefix": ("STRING", {"default": "3d/rigged"}),
                "bone_prefix": ("STRING", {"default": "mixamorig:",
                                           "tooltip": "Mixamo clips bind by exact joint name. "
                                                      "Clear it only for a pipeline that "
                                                      "expects bare names."}),
            },
            "optional": {
                "base_color": ("IMAGE", {"tooltip": "Baked atlas. Needs the mesh to carry UVs."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "3d/rigging"

    def save(self, mesh, rig, filename_prefix, bone_prefix, base_color=None,
             prompt=None, extra_pnginfo=None):
        # Same contract as the fit: never raise. The unrigged SaveGLB in the same
        # graph must not be lost because the skinned export could not be written.
        if not rig.get("ok", False):
            log.warning("[auto_rig] no rig to save (%s); skipping the skinned GLB",
                        rig.get("error", "fit failed"))
            return {"ui": {"text": ["Rigged GLB skipped: " + str(rig.get("error", "fit failed"))]}}
        try:
            verts, faces, uvs = _as_numpy(mesh)
            if len(verts) != int(rig["vertex_count"]):
                raise ValueError(
                    "the rig was fitted to a %d-vertex mesh but this one has %d. Skin "
                    "weights are per-vertex, so the mesh must not be decimated, unwrapped "
                    "or otherwise re-indexed between rigging and saving."
                    % (int(rig["vertex_count"]), len(verts)))

            tex = None
            if base_color is not None:
                if uvs is None:
                    log.warning("[auto_rig] base colour ignored: the mesh has no UVs")
                else:
                    tex = (base_color[0].detach().cpu().numpy() * 255.0).round().astype(np.uint8)

            full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory())
            name = "%s_%05d_.glb" % (filename, counter)
            path = os.path.join(full, name)

            blob = save_rigged_glb(
                verts.astype(np.float32), faces, rig["joints"], rig["parents"], rig["names"],
                rig["joint_idx"], rig["joint_w"], uvs=uvs if tex is not None else uvs,
                base_color=tex, prefix=bone_prefix,
                metadata={"generator": "comfyui_auto_rig",
                          "rig_confidence": float(rig["confidence"]),
                          "rig_notes": list(rig["notes"])})
            with open(path, "wb") as fh:
                fh.write(blob)
        except Exception as exc:
            log.exception("[auto_rig] could not write the rigged GLB; "
                          "the unrigged mesh is unaffected")
            return {"ui": {"text": ["Rigged GLB failed: " + str(exc)]}}

        log.info("[auto_rig] wrote %s (%.1f MB)", path, len(blob) / 1e6)
        return {"ui": {"3d": [{"filename": name, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "AutoRigHumanoid": AutoRigHumanoid,
    "PreviewRig": PreviewRig,
    "SaveRiggedGLB": SaveRiggedGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoRigHumanoid": "Auto Rig Humanoid",
    "PreviewRig": "Preview Rig",
    "SaveRiggedGLB": "Save Rigged GLB",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
