"""
Multiview projection texturing for ComfyUI.

Projects a set of reference images (front / back / left / right, or any number of
arbitrary yaw+pitch viewpoints) directly onto a UV-unwrapped mesh and bakes the
result into a texture atlas.

This is what lets a Hunyuan3D multiview mesh be textured with *your own art*
rather than a generated colour field: the geometry comes from the shape model,
and every texel takes its colour from whichever reference view sees that part of
the surface most directly.

Nodes
-----
  Add Projection View      chainable; one reference image + its camera angle
  Bake Texture From Views  mesh + views -> base colour atlas + coverage mask
  Render Projection View   ortho render from one camera, for view synthesis
  Fit View Angle           recover the camera angle of an off-axis reference
  Preview Projection Views renders the mesh from each view's camera, to check
                           that a reference image lines up with the geometry

Camera convention matches ComfyUI's own CreateCameraInfo orbit helper: the world
is right-handed and Y-up, and at yaw=0 the camera sits on +Z looking at the
origin, so the subject's face points toward +Z.

Which side is which follows from that. For a subject facing +Z with up +Y, its
own right hand points along forward x up = cross(+Z, +Y) = -X, so a camera at
yaw 90 (sitting at +X) sees the subject's LEFT side:

    yaw   0 = front      yaw  90 = subject's left
    yaw 180 = back       yaw 270 = subject's right

Turnaround art is not always labelled by the subject's own left and right - some
sheets name views by where the camera sits - so the caller may swap them.
"""

import math
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import Types

# ComfyUI's own bakers use these; reusing them keeps our atlas pixel-identical in
# layout to BakeTextureFromVoxel / BakeAmbientOcclusion output.
from comfy_extras.nodes_mesh_postprocess import (
    _rasterize_uv_barycentric,
    _build_triangle_bvh,
    _any_hit_rays_bvh,
)
from comfy_extras.nodes_save_3d import get_mesh_batch_item

from .geometry import (
    vertex_normals as _vertex_normals,
    camera_basis as _camera_basis,
    frame_extents as _frame_extents,
    project_to_frame as _project_to_frame,
    mask_bbox as _mask_bbox,
    sample_bilinear as _sample_bilinear,
    fill_and_dilate as _fill_and_dilate,
    inpaint_on_surface as _inpaint_on_surface,
    palette_from_seen as _palette_from_seen,
    snap_to_palette as _snap_to_palette,
    rasterize_screen as _rasterize_screen,
    silhouette_at as _silhouette_at,
    frame_mask as _frame_mask,
    fit_view_angle as _fit_view_angle,
    rasterize_silhouette as _rasterize_silhouette,
)
from .glb_read import load_mesh as _load_glb_mesh

try:
    import scipy.ndimage as ndi
except Exception:                                            # pragma: no cover
    ndi = None


VIEWS_TYPE = "PROJECTION_VIEWS"

# Occlusion rays cast per batch. Each carries a BVH traversal stack (~256 B), so
# this bounds that spike to roughly 64 MB regardless of atlas size.
RAY_CHUNK = 262144


# --------------------------------------------------------------------------- #
# Add Projection View
# --------------------------------------------------------------------------- #
class ProjectionViewAdd:
    """One reference image plus the camera angle it was drawn from.

    Chain several of these to describe as many viewpoints as you like. The image
    is kept at full resolution together with its mask; the bake node derives the
    framing from the mask's bounding box itself, so no pre-cropping is needed
    (and must not be applied, or the framing will no longer match the mesh).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "yaw": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                                  "tooltip": "0 = front, 90 = the model's right side, "
                                             "180 = back, 270 = left."}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 1.0,
                                    "tooltip": "Camera elevation in degrees. 0 for a level view."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.005,
                                    "tooltip": "Zoom of this image against the mesh. 1.0 maps the "
                                               "subject's bounding box onto the mesh's. Raise if "
                                               "the art sits inside the silhouette, lower if it "
                                               "spills outside."}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005,
                                       "tooltip": "Nudge the image sideways, as a fraction of the "
                                                  "subject's bounding box."}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005,
                                       "tooltip": "Nudge the image vertically."}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                     "tooltip": "Relative influence of this view where several "
                                                "views overlap."}),
            },
            "optional": {
                "views": (VIEWS_TYPE, {"tooltip": "Chain from another Add Projection View."}),
                "mask": ("MASK", {"tooltip": "Subject cutout. Strongly recommended — without it "
                                             "the whole frame is treated as subject and the "
                                             "background will be projected onto the model."}),
            },
        }

    RETURN_TYPES = (VIEWS_TYPE,)
    RETURN_NAMES = ("views",)
    FUNCTION = "add"
    CATEGORY = "3d/projection"

    def add(self, image, yaw, pitch, scale, offset_x, offset_y, weight,
            views=None, mask=None):
        entry = {
            "image": image[0] if image.ndim == 4 else image,
            "mask": (mask[0] if mask.ndim == 3 else mask) if mask is not None else None,
            "yaw": float(yaw),
            "pitch": float(pitch),
            "scale": float(scale),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "weight": float(weight),
        }
        return (list(views or []) + [entry],)


# --------------------------------------------------------------------------- #
# Bake Texture From Views
# --------------------------------------------------------------------------- #
class MultiviewProjectionBake:
    """Project every view onto the mesh and blend into a UV atlas.

    Per texel: interpolate its world position and normal, project into each
    camera, sample that view's image, and weight the sample by how directly the
    surface faces that camera (cos^facing_power). Texels facing away, occluded,
    or landing outside a view's subject mask get no contribution from it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Must already be UV-unwrapped."}),
                "views": (VIEWS_TYPE,),
                "texture_size": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "facing_power": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 16.0, "step": 0.1,
                                           "tooltip": "Higher = each view dominates the surfaces "
                                                      "it faces head-on, giving crisper art but "
                                                      "harder transitions between views."}),
                "occlusion": ("BOOLEAN", {"default": True,
                                          "tooltip": "Ray-test each texel against the mesh so "
                                                     "hidden surfaces (behind an arm, inside a "
                                                     "fold) do not receive that view's colour."}),
                "mask_erode": ("INT", {"default": 6, "min": 0, "max": 64,
                                       "tooltip": "Shrink each view's mask by this many pixels "
                                                  "before sampling. Cutout mattes are soft at the "
                                                  "outline, so texels near the silhouette would "
                                                  "otherwise pick up the image background and "
                                                  "paint a pale band down the model's sides."}),
                "min_facing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.9, "step": 0.01,
                                         "tooltip": "Reject a view's sample where the surface is "
                                                    "more edge-on than this (0 = accept "
                                                    "everything). Grazing samples are badly "
                                                    "stretched; rejecting them and approximating "
                                                    "instead looks far better. Raise it if sides "
                                                    "look smeared."}),
                "fill_unseen": ("COMBO", {"default": "surface",
                                          "options": ["surface", "surface+mirror", "nearest", "none"],
                                          "tooltip": "How to colour texels no view could reach. "
                                                     "surface: blend nearby seen texels across the "
                                                     "mesh (best with few views). surface+mirror: "
                                                     "also take the mirrored side, for symmetric "
                                                     "subjects missing one side. nearest: raw "
                                                     "nearest texel. none: leave black."}),
                "palette_colors": ("INT", {"default": 0, "min": 0, "max": 64,
                                           "tooltip": "Correct the approximated texels toward this "
                                                      "many colours taken from your own art "
                                                      "(0 = off). Blending unseen surface makes "
                                                      "gradients, and the midpoint of two palette "
                                                      "colours is a hue that appears nowhere in "
                                                      "your reference. Try 8-16. Real art is never "
                                                      "touched."}),
                "palette_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                               "tooltip": "How far to pull filled texels toward "
                                                          "the palette."}),
                "palette_keep_shading": ("BOOLEAN", {"default": True,
                                                     "tooltip": "Re-light each palette colour to "
                                                                "the texel's own brightness, so "
                                                                "shading and surface detail "
                                                                "survive the correction. Turn off "
                                                                "for a flat cel-shaded look."}),
                "dilate": ("INT", {"default": 4, "min": 0, "max": 64,
                                   "tooltip": "Bleed past chart edges to avoid seams."}),
            },
            "optional": {
                "reference_mesh": ("MESH", {"tooltip": "Dense pre-decimation mesh used for the "
                                                       "occlusion test, so a low-poly silhouette "
                                                       "does not self-shadow."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("base_color", "coverage")
    FUNCTION = "bake"
    CATEGORY = "3d/projection"

    def bake(self, mesh, views, texture_size, facing_power, occlusion,
             mask_erode, min_facing, fill_unseen, palette_colors, palette_strength,
             palette_keep_shading, dilate, reference_mesh=None):
        if not views:
            raise ValueError("Bake Texture From Views: no views connected. Add at least one "
                             "Add Projection View.")
        if mesh.uvs is None:
            raise ValueError("Bake Texture From Views: the mesh has no UVs. Put an Unwrap Mesh "
                             "UVs node before this one.")

        dev = comfy.model_management.get_torch_device()
        H = W = int(texture_size)
        batch = int(mesh.vertices.shape[0])
        out_imgs, out_cov = [], []
        pbar = comfy.utils.ProgressBar(batch * max(len(views), 1))

        for bi in range(batch):
            verts, faces, _colors, uvs, _normals = get_mesh_batch_item(mesh, bi)
            if faces.numel() == 0 or uvs is None:
                out_imgs.append(torch.zeros((H, W, 3)))
                out_cov.append(torch.zeros((H, W)))
                continue

            v = verts.to(dev).float()
            f = faces.to(dev).long()
            uv = uvs.to(dev).float()

            face_idx, bary, cov = _rasterize_uv_barycentric(
                f.cpu().numpy(), uv.cpu().numpy(), H)
            if not cov.any():
                raise ValueError("Bake Texture From Views: the UV layout covers no texels.")

            vn = _vertex_normals(v, f)
            vtri = f[face_idx[cov]]                            # [K,3]
            bsel = bary[cov]                                   # [K,3]
            P = (bsel[:, :, None] * v[vtri]).sum(1)            # [K,3] world position
            N = F.normalize((bsel[:, :, None] * vn[vtri]).sum(1), dim=-1, eps=1e-6)
            K = P.shape[0]

            # Occlusion is tested against the dense mesh when one is supplied.
            bvh = tri = None
            if occlusion:
                if reference_mesh is not None:
                    rv, rf = get_mesh_batch_item(reference_mesh,
                                                 min(bi, int(reference_mesh.vertices.shape[0]) - 1))[:2]
                    tri = rv.to(dev).float()[rf.to(dev).long()]
                else:
                    tri = v[f]
                bvh = _build_triangle_bvh(tri)
            diag = float((v.amax(0) - v.amin(0)).norm().clamp_min(1e-6))
            bias = 1e-3 * diag

            acc = torch.zeros((K, 3), device=dev)
            wsum = torch.zeros((K,), device=dev)

            for vi, view in enumerate(views):
                img = view["image"].to(dev).float()[..., :3]   # [h,w,3]
                msk = view["mask"]
                msk = msk.to(dev).float() if msk is not None else None
                ih, iw = img.shape[0], img.shape[1]

                # Framing uses the true silhouette, but sampling uses an eroded
                # copy: the outermost pixels of a soft matte are part background,
                # and letting them through paints a pale rim onto the model.
                msk_sample = msk
                if msk is not None and mask_erode > 0 and ndi is not None:
                    m_np = msk.cpu().numpy()
                    core = ndi.binary_erosion(m_np > 0.5, iterations=int(mask_erode))
                    msk_sample = torch.from_numpy(
                        np.ascontiguousarray(m_np * core)).to(dev).float()

                z, right, up = _camera_basis(view["yaw"], view["pitch"], dev)

                # Facing weight: z points from the model toward the camera.
                # Grazing samples are stretched along the surface and look far
                # worse than an approximation, so drop them below a threshold and
                # let the surface fill handle those texels instead.
                facing = (N * z[None, :]).sum(-1).clamp_min(0.0)
                facing = torch.where(facing < float(min_facing),
                                     torch.zeros_like(facing), facing)
                w = facing.pow(float(facing_power)) * float(view["weight"])
                if not bool((w > 0).any()):
                    pbar.update(1)
                    continue

                # Orthographic frame sized to the mesh's own projected bounding
                # box, so pu/pv run 0..1 across the silhouette.
                cu, cv, half = _frame_extents(v, right, up, 1.0)
                pu, pv = _project_to_frame(P, right, up, cu, cv, half)

                # Map that onto the subject's bounding box in this image. scale and
                # offset act on the image side only — applying them to both sides
                # would cancel out and do nothing.
                if msk is not None:
                    bb = _mask_bbox(msk.cpu().numpy())
                    if bb is None:
                        logging.warning("Bake Texture From Views: view %d mask is empty; "
                                        "using the whole frame.", vi)
                        bb = None
                else:
                    bb = None
                if bb is None:
                    bcx, bcy, bside = (iw - 1) * 0.5, (ih - 1) * 0.5, float(max(iw, ih))
                else:
                    bcx, bcy, bside = bb

                bcx += float(view["offset_x"]) * bside
                bcy += float(view["offset_y"]) * bside
                bside *= float(view["scale"])

                px = bcx + (pu - 0.5) * bside
                py = bcy + (pv - 0.5) * bside

                inside = (px >= 0) & (px <= iw - 1) & (py >= 0) & (py <= ih - 1)
                w = w * inside.float()

                col = _sample_bilinear(img, px, py)
                if msk is not None:
                    w = w * _sample_bilinear(msk_sample[..., None], px, py)[:, 0].clamp(0.0, 1.0)

                if occlusion and bool((w > 0).any()):
                    sel = w > 0
                    o_all = P[sel] + N[sel] * bias
                    wv = w[sel]
                    # The BVH keeps a per-ray traversal stack, so casting a 2k
                    # atlas's worth of texels at once costs hundreds of MB in one
                    # go. Chunk it the way ComfyUI's own AO baker does.
                    n_rays = o_all.shape[0]
                    step = max(1, int(RAY_CHUNK))
                    for s in range(0, n_rays, step):
                        o = o_all[s:s + step]
                        d = z[None, :].expand(o.shape[0], 3).contiguous()
                        hit = _any_hit_rays_bvh(o, d, tri, bvh,
                                                tmin=bias, tmax=4.0 * diag)
                        wv[s:s + step][hit] = 0.0
                    w = w.masked_scatter(sel, wv)

                acc += col * w[:, None]
                wsum += w
                pbar.update(1)

            seen = wsum > 1e-8
            rgb = torch.zeros((K, 3), device=dev)
            rgb[seen] = acc[seen] / wsum[seen][:, None]

            pct = 100.0 * float(seen.sum()) / max(K, 1)
            logging.info("Bake Texture From Views: %d views covered %.1f%% of the "
                         "surface directly; the rest is approximated.", len(views), pct)

            # Approximate the gaps on the mesh surface, before anything touches the
            # atlas — UV charts are disconnected, so filling in atlas space would
            # blend unrelated parts of the model together.
            seen_np = seen.cpu().numpy()
            if fill_unseen in ("surface", "surface+mirror") and not seen_np.all():
                src_np = rgb.cpu().numpy()
                rgb_np = _inpaint_on_surface(
                    src_np, P.cpu().numpy(), N.cpu().numpy(), seen_np,
                    mirror_x=(fill_unseen == "surface+mirror"))
                if palette_colors >= 2 and palette_strength > 0.0:
                    # Built from the seen texels only, so the palette is exactly the
                    # set of colours your reference art put on the model.
                    pal = _palette_from_seen(src_np, seen_np, int(palette_colors))
                    rgb_np = _snap_to_palette(rgb_np, pal, ~seen_np,
                                              strength=float(palette_strength),
                                              keep_detail=bool(palette_keep_shading))
                    if pal is not None:
                        logging.info("Bake Texture From Views: pulled %d approximated texels "
                                     "toward a %d-colour palette (strength %.2f, shading %s).",
                                     int((~seen_np).sum()), int(pal.shape[0]),
                                     float(palette_strength),
                                     "kept" if palette_keep_shading else "flattened")
                rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(dev).float()

            tex = torch.zeros((H, W, 3), device=dev)
            tex[cov] = rgb
            valid = torch.zeros((H, W), dtype=torch.bool, device=dev)
            valid[cov] = seen

            tex_np = tex.cpu().numpy()
            valid_np = valid.cpu().numpy()

            # Atlas-space work is now only about seams: bleed colour a few texels
            # past each chart edge so bilinear filtering never samples empty space.
            written = cov.cpu().numpy() if fill_unseen != "none" else valid_np
            if dilate > 0 or fill_unseen == "nearest":
                src_mask = valid_np if fill_unseen == "nearest" else written
                tex_np = _fill_and_dilate(tex_np, src_mask, int(dilate))
                if ndi is not None and dilate > 0:
                    band = ndi.binary_dilation(src_mask, iterations=int(dilate))
                    tex_np = np.where(band[..., None], tex_np, 0.0)

            out_imgs.append(torch.from_numpy(np.ascontiguousarray(tex_np)).float().clamp(0, 1))
            out_cov.append(valid.float().cpu())

        dst = comfy.model_management.intermediate_device()
        return (torch.stack(out_imgs).to(dst), torch.stack(out_cov).to(dst))


# --------------------------------------------------------------------------- #
# Preview Projection Views
# --------------------------------------------------------------------------- #
class ProjectionViewPreview:
    """Render the mesh from each view's camera using the same framing the bake
    uses. Put this beside your reference art to confirm the two line up before
    committing to a bake."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "views": (VIEWS_TYPE,),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "overlay": ("BOOLEAN", {"default": True,
                                        "tooltip": "Blend the reference image under the render "
                                                   "so misalignment is obvious."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("previews",)
    FUNCTION = "preview"
    CATEGORY = "3d/projection"

    def preview(self, mesh, views, resolution, overlay):
        dev = comfy.model_management.get_torch_device()
        R = int(resolution)
        verts, faces = get_mesh_batch_item(mesh, 0)[:2]
        v = verts.to(dev).float()
        f = faces.to(dev).long()
        vn = _vertex_normals(v, f)

        frames = []
        for view in views:
            z, right, up = _camera_basis(view["yaw"], view["pitch"], dev)
            au, av = v @ right, v @ up
            cu, cv, half = _frame_extents(v, right, up, 1.0)

            # Screen-space z-buffer rasterisation of the mesh, shaded by facing.
            su = ((au - cu) / (2.0 * half) + 0.5) * (R - 1)
            sv = (0.5 - (av - cv) / (2.0 * half)) * (R - 1)
            depth = v @ z
            shade = (vn @ z).clamp(0.15, 1.0)

            buf = torch.zeros((R * R, 3), device=dev)
            zbuf = torch.full((R * R,), -1e9, device=dev)
            tu, tv_, td, ts = su[f], sv[f], depth[f], shade[f]
            # Point sampling per triangle bbox, small meshes only — this is a check, not a render.
            for k in range(3):
                xi = tu[:, k].round().long().clamp(0, R - 1)
                yi = tv_[:, k].round().long().clamp(0, R - 1)
                lin = yi * R + xi
                cur = zbuf[lin]
                better = td[:, k] > cur
                idx = lin[better]
                zbuf[idx] = td[:, k][better]
                buf[idx] = ts[:, k][better][:, None].expand(-1, 3)

            frame = buf.view(R, R, 3)
            if overlay and view["image"] is not None:
                ref = view["image"].to(dev).float()[..., :3]
                ih, iw = ref.shape[0], ref.shape[1]
                msk = view["mask"]
                if msk is not None:
                    bb = _mask_bbox(msk.to(dev).float().cpu().numpy())
                else:
                    bb = None
                bcx, bcy, bside = bb if bb else ((iw - 1) * 0.5, (ih - 1) * 0.5, float(max(iw, ih)))
                bcx += float(view["offset_x"]) * bside
                bcy += float(view["offset_y"]) * bside
                bside *= float(view["scale"])
                gy, gx = torch.meshgrid(torch.arange(R, device=dev, dtype=torch.float32),
                                        torch.arange(R, device=dev, dtype=torch.float32),
                                        indexing="ij")
                px = bcx + (gx / (R - 1) - 0.5) * bside
                py = bcy + (gy / (R - 1) - 0.5) * bside
                ref_s = _sample_bilinear(ref, px.reshape(-1), py.reshape(-1)).view(R, R, 3)
                frame = (frame * 0.55 + ref_s * 0.45).clamp(0, 1)
            frames.append(frame.cpu())

        if not frames:
            frames = [torch.zeros((R, R, 3))]
        return (torch.stack(frames).to(comfy.model_management.intermediate_device()),)


# --------------------------------------------------------------------------- #
# Render Projection View
# --------------------------------------------------------------------------- #
class ProjectionViewRender:
    """Render the mesh from one projection camera, orthographically.

    This is the bridge to the view-synthesis stage. Because it uses the same
    camera and framing as Bake Texture From Views, an image derived from this
    render — an img2img refinement, say — can be fed straight back in as a
    projection view and will land on the model exactly.

    `gap_mask` marks the pixels whose colour was approximated rather than taken
    from a real reference view, which is precisely the region an inpainting or
    img2img pass should be allowed to change.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Textured mesh (after Apply Texture to Mesh)."}),
                "yaw": ("FLOAT", {"default": 90.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 1.0}),
                "resolution": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "background": ("COLOR", {"default": "#7f7f7f"}),
                "shade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Mix in facing-angle shading. 0 keeps the texture "
                                               "flat, which is usually what you want to hand to "
                                               "an image model."}),
            },
            "optional": {
                "coverage": ("MASK", {"tooltip": "Coverage atlas from Bake Texture From Views. "
                                                 "Supply it to get a gap_mask marking the pixels "
                                                 "that were approximated."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask", "gap_mask", "depth", "normal")
    FUNCTION = "render"
    CATEGORY = "3d/projection"

    def render(self, mesh, yaw, pitch, resolution, background, shade, coverage=None):
        dev = comfy.model_management.get_torch_device()
        R = int(resolution)
        verts, faces, _c, uvs, _n = get_mesh_batch_item(mesh, 0)
        v = verts.to(dev).float()
        f = faces.to(dev).long()

        z, right, up = _camera_basis(yaw, pitch, dev)
        cu, cv, half = _frame_extents(v, right, up, 1.0)
        face_idx, bary, cov = _rasterize_screen(v, f, right, up, z, cu, cv, half, R)

        h = background.lstrip("#")
        bg = torch.tensor([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
                          if len(h) == 6 else [0.5, 0.5, 0.5], device=dev)
        img = bg[None, None, :].expand(R, R, 3).clone()
        gap = torch.zeros((R, R), device=dev)

        if cov.any():
            tex = mesh.texture
            if tex is not None and uvs is not None:
                uv = uvs.to(dev).float()
                t = tex[0].to(dev).float() if tex.ndim == 4 else tex.to(dev).float()
                vtri = f[face_idx[cov]]
                bsel = bary[cov]
                uvp = (bsel[:, :, None] * uv[vtri]).sum(1)          # [K,2]
                th, tw = t.shape[0], t.shape[1]
                px = uvp[:, 0].clamp(0, 1) * (tw - 1)
                py = uvp[:, 1].clamp(0, 1) * (th - 1)
                img[cov] = _sample_bilinear(t, px, py)

                if coverage is not None:
                    cmask = coverage[0] if coverage.ndim == 3 else coverage
                    cm = cmask.to(dev).float()[..., None]
                    seen = _sample_bilinear(cm, px, py)[:, 0]
                    gap[cov] = (1.0 - seen).clamp(0.0, 1.0)
            else:
                # No texture yet: fall back to clay shading so the render is still useful.
                vn = _vertex_normals(v, f)
                vtri = f[face_idx[cov]]
                bsel = bary[cov]
                nn = torch.nn.functional.normalize((bsel[:, :, None] * vn[vtri]).sum(1),
                                                   dim=-1, eps=1e-6)
                img[cov] = (nn @ z).clamp(0.15, 1.0)[:, None].expand(-1, 3)
                gap[cov] = 1.0

            if shade > 0:
                vn = _vertex_normals(v, f)
                vtri = f[face_idx[cov]]
                bsel = bary[cov]
                nn = torch.nn.functional.normalize((bsel[:, :, None] * vn[vtri]).sum(1),
                                                   dim=-1, eps=1e-6)
                lam = (nn @ z).clamp(0.2, 1.0)[:, None]
                img[cov] = img[cov] * (1.0 - shade) + img[cov] * lam * shade

        # Depth and normal for structural conditioning, taken from this same
        # rasterisation rather than a second render. That is the whole point: a
        # ControlNet driven by these is pinned to exactly the pixels the bake
        # will reproject, so the image model can be run at a denoise high enough
        # to invent surface detail (leather, buckles, stitching on a satchel the
        # reference art never showed) without drifting off the silhouette.
        depth = torch.zeros((R, R), device=dev)
        nrm = torch.zeros((R, R, 3), device=dev)
        if cov.any():
            vtri = f[face_idx[cov]]
            bsel = bary[cov]
            p = (bsel[:, :, None] * v[vtri]).sum(1)
            d = p @ z                       # distance along the view axis
            vn = _vertex_normals(v, f)
            nn = F.normalize((bsel[:, :, None] * vn[vtri]).sum(1), dim=-1, eps=1e-6)
            # Camera space: x right, y up, z toward the camera.
            ncam = torch.stack([nn @ right, nn @ up, nn @ z], dim=-1)
            nrm[cov] = ncam * 0.5 + 0.5
            # Near is bright, matching what depth ControlNets are trained on.
            lo_d, hi_d = d.min(), d.max()
            depth[cov] = ((d - lo_d) / (hi_d - lo_d).clamp_min(1e-6)).clamp(0, 1)

        dst = comfy.model_management.intermediate_device()
        return (img[None].clamp(0, 1).to(dst),
                cov.float()[None].to(dst),
                gap[None].to(dst),
                depth[None, ..., None].expand(-1, -1, -1, 3).contiguous().to(dst),
                nrm[None].clamp(0, 1).to(dst))


# --------------------------------------------------------------------------- #
# Fit View Angle
# --------------------------------------------------------------------------- #
class ProjectionViewFit:
    """Work out which camera angle a reference image was drawn from.

    Needed when the art is not a clean front/side/back turnaround — an isometric
    three-quarter view, say. The mesh is rendered across a grid of orbit angles
    and the pose whose silhouette best matches the image's cutout wins, then the
    search narrows around it. Because the mesh was reconstructed from that same
    image, the true pose is a strong optimum.

    Feed `yaw` and `pitch` straight into Add Projection View. `overlay` shows the
    fitted render against the image so a bad fit is obvious rather than silent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "mask": ("MASK", {"tooltip": "Subject cutout of the reference image."}),
                "yaw_center": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0,
                                         "tooltip": "Middle of the yaw search. The image that "
                                                    "conditioned the shape model sits at 0, "
                                                    "because that model puts its input at the "
                                                    "front."}),
                "yaw_range": ("FLOAT", {"default": 30.0, "min": 5.0, "max": 180.0, "step": 5.0,
                                        "tooltip": "Half-width of the yaw search, in degrees. "
                                                   "180 searches everywhere. Keep this NARROW: "
                                                   "rounded subjects have near-identical "
                                                   "silhouettes from every angle, so a wide "
                                                   "search picks a random side."}),
                "yaw_steps": ("INT", {"default": 24, "min": 4, "max": 72,
                                      "tooltip": "Coarse orbit samples across the search range."}),
                "pitch_min": ("FLOAT", {"default": -60.0, "min": -89.0, "max": 0.0, "step": 5.0}),
                "pitch_max": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 89.0, "step": 5.0,
                                        "tooltip": "Isometric game art usually sits between "
                                                   "20 and 45 degrees above the horizon."}),
                "pitch_steps": ("INT", {"default": 13, "min": 1, "max": 41}),
                "refine_passes": ("INT", {"default": 2, "min": 0, "max": 4,
                                          "tooltip": "Extra narrowing passes around the best "
                                                     "coarse hit."}),
                "search_resolution": ("INT", {"default": 160, "min": 64, "max": 512, "step": 32}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Only used to draw the overlay."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("yaw", "pitch", "score", "margin", "overlay")
    FUNCTION = "fit"
    CATEGORY = "3d/projection"

    def fit(self, mesh, mask, yaw_center, yaw_range, yaw_steps, pitch_min, pitch_max,
            pitch_steps, refine_passes, search_resolution, image=None):
        dev = comfy.model_management.get_torch_device()
        verts, faces, _c, _uv, _n = get_mesh_batch_item(mesh, 0)
        v = verts.to(dev).float()
        f = faces.to(dev).long()

        m = (mask[0] if mask.ndim == 3 else mask).to(dev).float()
        target = _frame_mask(m, int(search_resolution))
        if target is None:
            raise ValueError("Fit View Angle: the mask is empty, so there is no "
                             "silhouette to match against.")

        yaw, pitch, score, margin = _fit_view_angle(
            v, f, target,
            yaw_steps=int(yaw_steps),
            pitch_range=(float(pitch_min), float(pitch_max)),
            pitch_steps=int(pitch_steps),
            res=int(search_resolution),
            refine_passes=int(refine_passes),
            yaw_center=float(yaw_center),
            yaw_range=float(yaw_range))

        logging.info("Fit View Angle: yaw %.1f deg, pitch %.1f deg "
                     "(IoU %.3f, margin over other poses %.3f)",
                     yaw, pitch, score, margin)
        # A high IoU means nothing on its own - what matters is beating the poses
        # that point elsewhere. Below this the answer is effectively a coin toss.
        if margin < 0.03:
            logging.warning(
                "Fit View Angle: the silhouette barely distinguishes this pose "
                "(margin %.3f). This subject looks the same from many angles, so "
                "the yaw is probably wrong. Narrow yaw_range, or set the angle by "
                "hand on Add Projection View.", margin)
        elif score < 0.5:
            logging.warning("Fit View Angle: weak match (IoU %.3f) - check the overlay; "
                            "the mask or the mesh may not correspond to this image.", score)

        # Overlay: fitted silhouette in cyan over the reference, so the fit is
        # judged by eye rather than trusted blindly.
        R = 512
        sil = _silhouette_at(v, f, yaw, pitch, R)
        over = torch.zeros((R, R, 3), device=dev)
        if image is not None:
            ref = (image[0] if image.ndim == 4 else image).to(dev).float()[..., :3]
            bb = _mask_bbox(m.cpu().numpy())
            ih, iw = ref.shape[0], ref.shape[1]
            bcx, bcy, bside = bb if bb else ((iw - 1) * 0.5, (ih - 1) * 0.5, float(max(iw, ih)))
            gy, gx = torch.meshgrid(
                torch.arange(R, device=dev, dtype=torch.float32),
                torch.arange(R, device=dev, dtype=torch.float32), indexing="ij")
            px = bcx + (gx / (R - 1) - 0.5) * bside
            py = bcy + (gy / (R - 1) - 0.5) * bside
            over = _sample_bilinear(ref, px.reshape(-1), py.reshape(-1)).view(R, R, 3)
        tint = torch.tensor([0.1, 0.9, 0.8], device=dev)
        over = torch.where(sil[..., None], over * 0.45 + tint * 0.55, over)

        dst = comfy.model_management.intermediate_device()
        return (float(yaw), float(pitch), float(score),
                float(margin if margin != float("inf") else 1.0),
                over[None].clamp(0, 1).to(dst))


# --------------------------------------------------------------------------- #
# Load Mesh From GLB
# --------------------------------------------------------------------------- #
class LoadMeshGLB:
    """Read a .glb written earlier in the same session back into a MESH.

    This is what makes a multi-stage run possible. Held in one prompt, the shape
    model and the image model are resident together and a 10 GB card spends most
    of its time paging between GPU and CPU rather than computing. Split across
    prompts, each stage gets the whole card -- but only if geometry can be handed
    from one prompt to the next, and ComfyUI ships no way to read a mesh back.

    Loads positions, UVs, triangles and the base colour texture: exactly what the
    projection bake and the render node consume.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "3d/stage1_00001_.glb",
                                        "tooltip": "Path relative to the chosen "
                                                   "directory, as written by Save GLB."}),
                "directory": ("COMBO", {"default": "output",
                                        "options": ["output", "input", "temp"]}),
            },
        }

    RETURN_TYPES = ("MESH", "IMAGE")
    RETURN_NAMES = ("mesh", "base_color")
    FUNCTION = "load"
    CATEGORY = "3d/projection"

    @classmethod
    def IS_CHANGED(cls, filename, directory):
        # Staged runs write a new file per stage, but a re-run with the same name
        # must not be served from the execution cache.
        try:
            return str(os.path.getmtime(cls._resolve(filename, directory)))
        except Exception:
            return float("nan")

    @staticmethod
    def _resolve(filename, directory):
        roots = {
            "output": folder_paths.get_output_directory(),
            "input": folder_paths.get_input_directory(),
            "temp": folder_paths.get_temp_directory(),
        }
        root = roots.get(directory, roots["output"])
        full = os.path.abspath(os.path.join(root, filename))
        if not full.startswith(os.path.abspath(root)):
            raise ValueError("path escapes the %s directory" % directory)
        return full

    def load(self, filename, directory):
        path = self._resolve(filename, directory)
        if not os.path.exists(path):
            raise ValueError("Load Mesh From GLB: no such file: " + path)

        verts, faces, uvs, tex = _load_glb_mesh(path)
        logging.info("[projection] loaded %s: %d verts, %d tris, uvs=%s, texture=%s",
                     os.path.basename(path), len(verts), len(faces),
                     uvs is not None, None if tex is None else tex.shape)

        v = torch.from_numpy(np.ascontiguousarray(verts)).float()[None]
        f = torch.from_numpy(np.ascontiguousarray(faces)).long()[None]
        kw = {}
        if uvs is not None:
            kw["uvs"] = torch.from_numpy(np.ascontiguousarray(uvs)).float()[None]
        if tex is not None:
            kw["texture"] = torch.from_numpy(np.ascontiguousarray(tex)).float()[None]
        mesh = Types.MESH(v, f, **kw)

        out_tex = (kw["texture"] if tex is not None
                   else torch.zeros((1, 8, 8, 3), dtype=torch.float32))
        return (mesh, out_tex)


NODE_CLASS_MAPPINGS = {
    "LoadMeshGLB": LoadMeshGLB,
    "ProjectionViewFit": ProjectionViewFit,
    "ProjectionViewAdd": ProjectionViewAdd,
    "MultiviewProjectionBake": MultiviewProjectionBake,
    "ProjectionViewPreview": ProjectionViewPreview,
    "ProjectionViewRender": ProjectionViewRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMeshGLB": "Load Mesh From GLB",
    "ProjectionViewAdd": "Add Projection View",
    "MultiviewProjectionBake": "Bake Texture From Views",
    "ProjectionViewPreview": "Preview Projection Views",
    "ProjectionViewRender": "Render Projection View",
    "ProjectionViewFit": "Fit View Angle",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
