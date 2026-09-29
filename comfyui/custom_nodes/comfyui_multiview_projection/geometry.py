"""
Pure geometry/sampling maths for multiview projection texturing.

Deliberately free of any ComfyUI imports so the camera convention and framing
can be tested standalone (see scripts/verify-projection.py in the project repo).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    import scipy.ndimage as ndi
except Exception:                                            # pragma: no cover
    ndi = None


# Cap on elements in a single rasteriser intermediate. The tile loop builds
# [triangles, tile_h, tile_w] tensors, so on a dense mesh drawn small this is the
# difference between ~64 MB and several GB in one allocation. 16M float32 ~ 64 MB,
# and five such intermediates are live at once.
MAX_RASTER_ELEMS = 16_000_000


def vertex_normals(verts, faces):
    """Area-weighted per-vertex normals. verts [N,3], faces [M,3] -> [N,3]."""
    tri = verts[faces]
    fn = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = torch.zeros_like(verts)
    for k in range(3):
        out.index_add_(0, faces[:, k], fn)
    return F.normalize(out, dim=-1, eps=1e-8)


def camera_basis(yaw_deg, pitch_deg, device=None):
    """Orbit angles -> (eye_dir, right, up), all unit vectors.

    `eye_dir` points from the pivot toward the camera. This reproduces
    _orbit_camera_info in comfy_extras/nodes_gaussian_splat.py:

        fwd_splat = (-cp*sy, sp, cp*cy)
        eye_splat = pivot - distance * fwd_splat
        world     = (x, -y, -z)

    which yields eye_world_dir = (cp*sy, sp, cp*cy). At yaw=0 that is +Z, so the
    subject's front faces +Z, matching ComfyUI's Load3D/three.js frame.
    """
    y, p = math.radians(float(yaw_deg)), math.radians(float(pitch_deg))
    cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    z = torch.tensor([cp * sy, sp, cp * cy], device=device, dtype=torch.float32)
    z = F.normalize(z, dim=0, eps=1e-8)

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    if abs(float(torch.dot(z, world_up))) > 0.999:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    x = F.normalize(torch.linalg.cross(world_up, z), dim=0, eps=1e-8)
    yv = torch.linalg.cross(z, x)
    return z, x, yv


def frame_extents(verts, right, up, scale=1.0):
    """Orthographic framing of a mesh for one view.

    Returns (centre_u, centre_v, half_side). Matches how ImageCropToMask frames a
    subject: centred on the bounding box, square, side = max(extent) * scale.
    """
    au = verts @ right
    av = verts @ up
    cu = (au.max() + au.min()) * 0.5
    cv = (av.max() + av.min()) * 0.5
    side = torch.maximum(au.max() - au.min(), av.max() - av.min()) * float(scale)
    return cu, cv, (side * 0.5).clamp_min(1e-8)


def project_to_frame(points, right, up, cu, cv, half):
    """World points -> normalised frame coords in [0,1], y down (image order)."""
    pu = ((points @ right) - cu) / (2.0 * half) + 0.5
    pv = 0.5 - ((points @ up) - cv) / (2.0 * half)
    return pu, pv


def mask_bbox(mask_np, threshold=0.5):
    """(cx, cy, side) of a mask's foreground bbox in pixels; None when empty."""
    ys, xs = np.nonzero(mask_np > threshold)
    if ys.size == 0:
        return None
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5, max(x1 - x0 + 1.0, y1 - y0 + 1.0))


def sample_bilinear(img_hwc, px, py):
    """Sample [H,W,C] at float pixel coords -> [K,C]. Out of bounds returns 0."""
    H, W, C = img_hwc.shape
    gx = (px / max(W - 1, 1)) * 2.0 - 1.0
    gy = (py / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    src = img_hwc.permute(2, 0, 1).unsqueeze(0)
    out = F.grid_sample(src, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.view(C, -1).permute(1, 0)


def fill_and_dilate(color, valid, dilate_px):
    """Nearest-neighbour fill of unseen texels so bilinear filtering never samples
    empty space at a UV seam. color [H,W,3] float32, valid [H,W] bool."""
    if not valid.any():
        return color
    if ndi is not None:
        _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
        return color[iy, ix]

    filled = color.copy()
    cur = valid.copy()
    for _ in range(max(int(dilate_px), 16)):
        if cur.all():
            break
        t = torch.from_numpy(filled).permute(2, 0, 1).unsqueeze(0)
        m = torch.from_numpy(cur.astype(np.float32))[None, None]
        k = torch.ones((1, 1, 3, 3))
        num = F.conv2d(t * m, k.expand(t.shape[1], 1, 3, 3), padding=1, groups=t.shape[1])
        den = F.conv2d(m, k, padding=1)
        grown = (den[0, 0] > 0).numpy()
        new = grown & ~cur
        vals = (num / den.clamp_min(1e-6))[0].permute(1, 2, 0).numpy()
        filled[new] = vals[new]
        cur = cur | new
    return filled


def inpaint_on_surface(rgb, pos, nrm, seen, k=10, normal_weight=3.0,
                       mirror_x=False, mirror_tol=0.02, passes=2):
    """Approximate colour for texels no view could see, working on the mesh
    surface rather than in the UV atlas.

    Atlas-space filling is wrong here: UV charts are disconnected, so a texel's
    atlas neighbours are often unrelated parts of the model. This instead finds
    each unseen texel's nearest *seen* texels in 3D and blends them, weighted by
    distance and by how well their normals agree. Normal agreement is what keeps
    colour from bleeding through thin geometry — the front of a sleeve will not
    pull colour from the back of the same sleeve a few millimetres away.

    With only front and back views this is what fills the side band: it blends
    the front and back colours around the silhouette instead of streaking one of
    them sideways.

    `mirror_x` first tries the mirrored position on the model, which recovers a
    genuinely missing side from the opposite one on a symmetric subject.

    rgb/pos/nrm are [K,3] float arrays, seen is [K] bool. Returns filled rgb.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return rgb

    if not seen.any() or seen.all():
        return rgb

    out = rgb.copy()
    src_pos, src_rgb, src_nrm = pos[seen], rgb[seen], nrm[seen]
    tree = cKDTree(src_pos)
    kk = int(min(k, src_pos.shape[0]))

    tgt_idx = np.flatnonzero(~seen)
    tgt_pos, tgt_nrm = pos[tgt_idx], nrm[tgt_idx]
    filled = np.zeros(tgt_idx.shape[0], dtype=bool)

    def blend(query_pos, query_nrm):
        d, idx = tree.query(query_pos, k=kk)
        if kk == 1:
            d, idx = d[:, None], idx[:, None]
        w = 1.0 / (d + 1e-6)
        agree = np.clip((src_nrm[idx] * query_nrm[:, None, :]).sum(-1), 0.0, 1.0)
        w = w * (agree ** float(normal_weight) + 1e-3)
        c = (src_rgb[idx] * w[..., None]).sum(1) / w.sum(1)[:, None]
        return c, d[:, 0]

    if mirror_x:
        mpos = tgt_pos.copy()
        mpos[:, 0] *= -1.0
        mnrm = tgt_nrm.copy()
        mnrm[:, 0] *= -1.0
        mc, md = blend(mpos, mnrm)
        # Only trust the mirror where it landed on real surface.
        scale = float(np.linalg.norm(pos.max(0) - pos.min(0))) or 1.0
        good = md < (mirror_tol * scale)
        out[tgt_idx[good]] = mc[good]
        filled |= good

    rest = ~filled
    if rest.any():
        c, _ = blend(tgt_pos[rest], tgt_nrm[rest])
        out[tgt_idx[rest]] = c

    # A couple of smoothing passes over the filled region only, so the k-NN
    # patchwork relaxes toward a smooth gradient across the gap.
    if passes > 0 and tgt_idx.size:
        all_tree = cKDTree(pos)
        nb = all_tree.query(pos[tgt_idx], k=min(9, pos.shape[0]))[1]
        for _ in range(int(passes)):
            out[tgt_idx] = out[nb].mean(1)
            out[seen] = rgb[seen]                     # seen texels stay authoritative
    return out


def chromaticity(a, eps=1e-4):
    """Colour with brightness divided out: every entry has luminance 1."""
    return a / np.maximum((a @ LUMA)[..., None], eps)


def palette_from_seen(rgb, seen, k, iters=14, sample=20000, seed=0, dark_cut=0.02):
    """The distinct *hues* of the texels real reference art actually reached.

    Clustered in chromaticity rather than RGB. Clustering raw RGB sorts by
    brightness, so a single shaded body colour splits into "lit" and "shadowed"
    clusters and two genuinely different hues get merged to pay for it -- which
    defeats the point, since the whole job is to tell hues apart. Dividing
    luminance out first makes each shaded material one tight cluster however
    strongly it is lit.

    Seeded farthest-point so rare accent colours survive instead of being
    swallowed by the dominant body colour. Returns [k,3] in RGB, each at its
    cluster's mean brightness, or None when there is not enough to cluster.
    """
    src = rgb[seen]
    # Near-black texels have no reliable hue; including them drags a cluster to
    # an arbitrary direction.
    src = src[(src @ LUMA) > dark_cut]
    if src.shape[0] < k or k < 2:
        return None
    rs = np.random.RandomState(int(seed))
    if src.shape[0] > sample:
        src = src[rs.choice(src.shape[0], sample, replace=False)]

    ch = chromaticity(src)
    lum = src @ LUMA

    centres = [ch[rs.randint(ch.shape[0])]]
    for _ in range(int(k) - 1):
        d = np.min(np.linalg.norm(ch[:, None, :] - np.array(centres)[None], axis=2), axis=1)
        centres.append(ch[int(np.argmax(d))])
    C = np.array(centres, dtype=np.float64)

    lab = None
    for _ in range(int(iters)):
        lab = np.argmin(((ch[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(C.shape[0]):
            m = lab == j
            if m.any():
                C[j] = ch[m].mean(axis=0)

    # Give each hue its cluster's typical brightness, so the palette is usable
    # as-is when shading is not being preserved.
    out = np.empty_like(C)
    for j in range(C.shape[0]):
        m = lab == j
        out[j] = np.clip(C[j] * (float(np.median(lum[m])) if m.any() else 0.5), 0.0, 1.0)
    return out


LUMA = np.array([0.2126, 0.7152, 0.0722])


def snap_to_palette(rgb, palette, mask, strength=1.0, keep_detail=True,
                    shade_range=(0.55, 1.75)):
    """Pull the *approximated* texels back onto the source palette's hues,
    keeping their shading.

    Blending unseen surface from its neighbours produces gradients between
    colours, and on stylized art the midpoint of two palette colours is a hue
    that appears nowhere in the reference -- it reads as mud. The fix is to
    correct the hue without flattening: each filled texel takes the nearest
    palette colour, then that colour is re-scaled to the texel's own brightness.
    Shading, ambient occlusion, and the light-to-dark variation that makes a
    stylized model look like more than paper survive; only the wrong hue goes.

    Set `keep_detail=False` for a genuinely flat cel-shaded look.

    `strength` blends against the original, so partial correction is available.
    The real art is never in `mask` and is never touched.
    """
    if palette is None or not mask.any():
        return rgb
    out = rgb.copy()
    sel = rgb[mask]
    # Matched on hue, not on RGB distance: a shadowed patch of the body colour is
    # far from that colour in RGB and would otherwise be reassigned to whichever
    # palette entry happens to be dark.
    sc, pc = chromaticity(sel), chromaticity(palette)
    idx = np.argmin(((sc[:, None, :] - pc[None]) ** 2).sum(-1), axis=1)
    target = palette[idx]

    if keep_detail:
        # Re-light the palette colour to the texel's own luminance. Clamped so a
        # near-black crevice cannot collapse the hue to zero, and a specular
        # highlight cannot blow it past white.
        lum_src = sel @ LUMA
        lum_dst = np.maximum(target @ LUMA, 1e-4)
        ratio = np.clip(lum_src / lum_dst, shade_range[0], shade_range[1])
        target = np.clip(target * ratio[:, None], 0.0, 1.0)

    s = float(np.clip(strength, 0.0, 1.0))
    out[mask] = (1.0 - s) * sel + s * target
    return out


def raster_budget(device, coverage_only=False):
    """Elements per rasteriser intermediate, sized to VRAM actually free.

    Five intermediates are live at once in the full path (two in the coverage
    -only path), so this spends about an eighth of free memory and clamps to a
    sane range. Falls back to the static cap when the device is not CUDA.
    """
    try:
        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(device)
            live = 2 if coverage_only else 5
            elems = int((free * 0.125) / (4 * live))
            return max(1_000_000, min(MAX_RASTER_ELEMS, elems))
    except Exception:
        pass
    return MAX_RASTER_ELEMS


def rasterize_screen(verts, faces, right, up, eye_dir, cu, cv, half, res, tile=64,
                     coverage_only=False, max_elems=None):
    """Z-buffered orthographic rasterisation of a mesh.

    Returns (face_idx [R,R] long, bary [R,R,3], cov [R,R] bool). Uses the same
    camera and framing as the projection bake, so anything rendered here maps
    back onto the model exactly — that is what lets a generated view be
    reprojected without any alignment fudging.

    Depth is `verts @ eye_dir`, which increases toward the camera, so the
    nearest surface is the per-pixel maximum.
    """
    dev = verts.device
    R = int(res)
    face_idx = torch.zeros((R, R), dtype=torch.long, device=dev)
    bary = torch.zeros((R, R, 3), device=dev)
    cov = torch.zeros((R, R), dtype=torch.bool, device=dev)
    if faces.shape[0] == 0:
        return face_idx, bary, cov

    pu, pv = project_to_frame(verts, right, up, cu, cv, half)
    x = pu * (R - 1)
    y = pv * (R - 1)
    depth = verts @ eye_dir

    tx = x[faces]                                             # [F,3]
    ty = y[faces]
    td = depth[faces]
    x0, x1, x2 = tx[:, 0], tx[:, 1], tx[:, 2]
    y0, y1, y2 = ty[:, 0], ty[:, 1], ty[:, 2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegen = denom.abs() > 1e-20

    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, R - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, R - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, R - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, R - 1).long()

    eps = 1e-6
    neg_inf = float("-inf")
    for ty0 in range(0, R, tile):
        ty1 = min(ty0 + tile, R)
        th = ty1 - ty0
        for tx0 in range(0, R, tile):
            tx1 = min(tx0 + tile, R)
            tw = tx1 - tx0
            m = (nondegen & (xmin < tx1) & (xmax >= tx0) & (ymin < ty1) & (ymax >= ty0))
            if not m.any():
                continue
            idx = torch.nonzero(m, as_tuple=True)[0]

            ys = torch.arange(ty0, ty1, dtype=torch.float32, device=dev) + 0.5
            xs = torch.arange(tx0, tx1, dtype=torch.float32, device=dev) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")

            # Per-tile running z-buffer, so triangles can be processed in chunks.
            # Without this the intermediates are [K, th, tw] with K = every
            # triangle over the tile: a dense mesh rendered small puts the whole
            # model in one tile and asks for gigabytes in a single allocation.
            t_depth = torch.full((th, tw), neg_inf, device=dev)
            t_face = torch.zeros((th, tw), dtype=torch.long, device=dev)
            t_bary = torch.zeros((th, tw, 3), device=dev)

            budget = max_elems if max_elems else raster_budget(dev, coverage_only)
            chunk = max(1, int(budget // max(th * tw, 1)))
            for s in range(0, idx.numel(), chunk):
                sub = idx[s:s + chunk]
                sx0, sy0 = x0[sub][:, None, None], y0[sub][:, None, None]
                sx1, sy1 = x1[sub][:, None, None], y1[sub][:, None, None]
                sx2, sy2 = x2[sub][:, None, None], y2[sub][:, None, None]
                sden = denom[sub][:, None, None]
                b0 = ((sy1 - sy2) * (xx - sx2) + (sx2 - sx1) * (yy - sy2)) / sden
                b1 = ((sy2 - sy0) * (xx - sx2) + (sx0 - sx2) * (yy - sy2)) / sden
                b2 = 1.0 - b0 - b1
                inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)
                if not inside.any():
                    continue

                # A silhouette needs no depth ordering or barycentrics, so drop
                # them entirely - that is most of the memory and time.
                if coverage_only:
                    cov[ty0:ty1, tx0:tx1] |= inside.any(dim=0)
                    del b0, b1, b2, inside
                    continue

                sd = td[sub]
                dep = (b0 * sd[:, 0][:, None, None] + b1 * sd[:, 1][:, None, None]
                       + b2 * sd[:, 2][:, None, None])
                dep = torch.where(inside, dep, torch.full_like(dep, neg_inf))

                best = dep.argmax(dim=0)                       # [th,tw]
                bval = dep.gather(0, best[None]).squeeze(0)
                better = bval > t_depth
                if not better.any():
                    continue
                bsel = torch.stack([b0.gather(0, best[None]).squeeze(0),
                                    b1.gather(0, best[None]).squeeze(0),
                                    b2.gather(0, best[None]).squeeze(0)], dim=-1)
                t_depth = torch.where(better, bval, t_depth)
                t_face = torch.where(better, sub[best], t_face)
                t_bary = torch.where(better[..., None], bsel, t_bary)

            if coverage_only:
                continue                                       # cov filled per chunk
            hit = t_depth > neg_inf
            if not hit.any():
                continue
            face_idx[ty0:ty1, tx0:tx1][hit] = t_face[hit]
            bary[ty0:ty1, tx0:tx1][hit] = t_bary[hit]
            cov[ty0:ty1, tx0:tx1] |= hit

    return face_idx, bary, cov


def silhouette_at(verts, faces, yaw, pitch, res, scale=1.0):
    """Binary silhouette of the mesh from one orbit angle, framed like the bake."""
    z, right, up = camera_basis(yaw, pitch, verts.device)
    cu, cv, half = frame_extents(verts, right, up, scale)
    return rasterize_screen(verts, faces, right, up, z, cu, cv, half, res,
                            coverage_only=True)[2]


def frame_mask(mask, res):
    """Crop a subject mask to its bounding box and resize to a square, exactly the
    way the bake frames a view, so it can be compared against silhouette_at()."""
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return None
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    cy, cx = (y0 + y1) * 0.5, (x0 + x1) * 0.5
    side = max(y1 - y0 + 1.0, x1 - x0 + 1.0)

    gy, gx = torch.meshgrid(
        torch.arange(res, dtype=torch.float32, device=mask.device),
        torch.arange(res, dtype=torch.float32, device=mask.device), indexing="ij")
    sy = cy + (gy / (res - 1) - 0.5) * side
    sx = cx + (gx / (res - 1) - 0.5) * side
    H, W = mask.shape
    grid = torch.stack([(sx / max(W - 1, 1)) * 2 - 1,
                        (sy / max(H - 1, 1)) * 2 - 1], dim=-1)
    out = F.grid_sample(mask[None, None].float(), grid[None], mode="nearest",
                        padding_mode="zeros", align_corners=True)
    return out[0, 0] > 0.5


def silhouette_iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if int(union) else 0.0


def fit_view_angle(verts, faces, target, yaw_steps=24, pitch_range=(-60.0, 60.0),
                   pitch_steps=13, res=160, refine_passes=2,
                   yaw_center=0.0, yaw_range=180.0):
    """Recover the camera angle an image was drawn from.

    Renders the mesh across a grid of orbit angles and keeps the pose whose
    silhouette best matches `target` (an already-framed binary mask), then
    narrows around the winner.

    IMPORTANT: a silhouette does not always identify a pose. Rounded or
    near-symmetric subjects look much the same from every direction - measured on
    a low-poly creature, IoU spanned only 0.61-0.71 across all 24 yaws and the top
    two candidates differed by 0.001, i.e. pure noise. So a high score alone means
    nothing; `margin` (best minus the best clearly-different pose) is what says
    whether the answer is trustworthy. Constrain `yaw_range` when the yaw is
    already known - the image that conditioned the shape model sits at yaw 0.

    Returns (yaw, pitch, score, margin).
    """
    best = (0.0, 0.0, -1.0)
    span = max(1.0, min(180.0, float(yaw_range)))
    yaw_lo, yaw_hi = yaw_center - span, yaw_center + span
    coarse = []
    p_lo, p_hi = pitch_range
    ys, ps = int(yaw_steps), int(pitch_steps)

    for it in range(int(refine_passes) + 1):
        yaws = [yaw_lo + (yaw_hi - yaw_lo) * i / ys for i in range(ys)] if it == 0 else \
               [yaw_lo + (yaw_hi - yaw_lo) * i / max(ys - 1, 1) for i in range(ys)]
        pitches = [p_lo + (p_hi - p_lo) * i / max(ps - 1, 1) for i in range(ps)]
        for yw in yaws:
            for pt in pitches:
                sil = silhouette_at(verts, faces, yw % 360.0, pt, res)
                s = silhouette_iou(sil, target)
                if it == 0:
                    coarse.append((yw % 360.0, s))
                if s > best[2]:
                    best = (yw % 360.0, pt, s)
        # Narrow the window around the current winner for the next pass.
        yspan = (yaw_hi - yaw_lo) / ys
        pspan = (p_hi - p_lo) / max(ps - 1, 1)
        yaw_lo, yaw_hi = best[0] - yspan, best[0] + yspan
        p_lo, p_hi = max(-89.0, best[1] - pspan), min(89.0, best[1] + pspan)
        ys, ps = 9, 7

    # Confidence: how far the winner beats the best pose pointing somewhere else.
    # Poses within 45 degrees of the winner are the same answer, not rivals.
    rival = -1.0
    for yw, s in coarse:
        d = abs((yw - best[0] + 180.0) % 360.0 - 180.0)
        if d > 45.0:
            rival = max(rival, s)
    margin = best[2] - rival if rival >= 0.0 else float("inf")
    return best[0], best[1], best[2], margin


def rasterize_silhouette(verts, faces, right, up, eye_dir, cu, cv, half, res):
    """Small z-buffered point rasteriser used for view previews and for offline
    verification that a reference image lines up with the mesh."""
    pu, pv = project_to_frame(verts, right, up, cu, cv, half)
    su = (pu * (res - 1)).round().long().clamp(0, res - 1)
    sv = (pv * (res - 1)).round().long().clamp(0, res - 1)
    depth = verts @ eye_dir
    vn = vertex_normals(verts, faces)
    shade = (vn @ eye_dir).clamp(0.12, 1.0)

    buf = torch.zeros((res * res,), dtype=torch.float32, device=verts.device)
    zbuf = torch.full((res * res,), -1e9, device=verts.device)
    lin_all = sv * res + su
    for k in range(3):
        idx = faces[:, k]
        lin = lin_all[idx]
        d = depth[idx]
        better = d > zbuf[lin]
        sel = lin[better]
        zbuf[sel] = d[better]
        buf[sel] = shade[idx][better]
    return buf.view(res, res)
