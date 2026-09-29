"""
Determine empirically which yaw corresponds to the 'left' and 'right' reference
images, by silhouette matching rather than by eye.

A character's left and right silhouettes differ (asymmetric pose, props, hair),
so intersection-over-union between the rendered silhouette and the reference
cutout discriminates the two. Whichever pairing scores higher is the truth.

    python scripts/check-view-mapping.py <mesh.glb> <ref_dir>

ref_dir holds front/left/back/right .png with a plain background.
"""

import json
import struct
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _nodepath                                   # noqa: E402
_nodepath.add("comfyui_multiview_projection")
import geometry as G  # noqa: E402


def load_glb(path):
    buf = Path(path).read_bytes()
    off, js, bin_ = 12, None, None
    while off < len(buf):
        ln, ty = struct.unpack_from("<II", buf, off)
        ch = buf[off + 8: off + 8 + ln]
        if ty == 0x4E4F534A:
            js = json.loads(ch)
        elif ty == 0x004E4942:
            bin_ = ch
        off += 8 + ln + ((4 - ln % 4) % 4)

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        st = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "<u1"}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.frombuffer(bin_, dtype=np.dtype(dt),
                             count=a["count"] * nc, offset=st).reshape(a["count"], nc)

    vs, fs, base = [], [], 0
    for m in js["meshes"]:
        for p in m["primitives"]:
            v = acc(p["attributes"]["POSITION"]).astype(np.float32)
            f = acc(p["indices"]).reshape(-1, 3).astype(np.int64) + base
            vs.append(v); fs.append(f); base += v.shape[0]
    return torch.from_numpy(np.concatenate(vs)), torch.from_numpy(np.concatenate(fs))


def ref_silhouette(path, res):
    """Subject mask from a reference image, cropped to its bbox and resized to a
    square exactly the way the bake frames a view."""
    from torchvision.io import read_image
    img = read_image(str(path)).float() / 255.0
    rgb = img[:3].permute(1, 2, 0)
    # Plain studio background: near-uniform and brighter than the subject edges.
    corners = torch.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg = corners.median(0).values
    mask = ((rgb - bg[None, None, :]).abs().sum(-1) > 0.12).float()

    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return None
    y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    side = max(y1 - y0 + 1, x1 - x0 + 1)
    half = side / 2.0

    gy, gx = torch.meshgrid(torch.arange(res, dtype=torch.float32),
                            torch.arange(res, dtype=torch.float32), indexing="ij")
    sy = cy + (gy / (res - 1) - 0.5) * side
    sx = cx + (gx / (res - 1) - 0.5) * side
    H, W = mask.shape
    gs = torch.stack([(sx / max(W - 1, 1)) * 2 - 1, (sy / max(H - 1, 1)) * 2 - 1], dim=-1)
    out = F.grid_sample(mask[None, None], gs[None], mode="nearest",
                        padding_mode="zeros", align_corners=True)
    return (out[0, 0] > 0.5)


def iou(a, b):
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 0.0


def main():
    mesh_path, ref_dir = sys.argv[1], Path(sys.argv[2])
    res = 320
    v, f = load_glb(mesh_path)

    rendered = {}
    for yaw in (90.0, 270.0):
        z, right, up = G.camera_basis(yaw, 0.0)
        cu, cv, half = G.frame_extents(v, right, up, 1.0)
        _, _, cov = G.rasterize_screen(v, f, right, up, z, cu, cv, half, res)
        rendered[yaw] = cov

    refs = {}
    for name in ("left", "right"):
        s = ref_silhouette(ref_dir / f"{name}.png", res)
        if s is None:
            print(f"could not build a silhouette for {name}.png")
            return
        refs[name] = s

    print(f"{'':>10}" + "".join(f"{('yaw ' + str(int(y))):>12}" for y in rendered))
    scores = {}
    for name, rs in refs.items():
        row = f"{name:>10}"
        for yaw, cov in rendered.items():
            s = iou(cov, rs)
            scores[(name, yaw)] = s
            row += f"{s:>12.3f}"
        print(row)

    # Compare the two consistent pairings.
    a = scores[("right", 90.0)] + scores[("left", 270.0)]     # current mapping
    b = scores[("left", 90.0)] + scores[("right", 270.0)]     # swapped
    print()
    print(f"  right=90, left=270  (current) : {a:.3f}")
    print(f"  left=90,  right=270 (swapped) : {b:.3f}")
    print()
    print("  => " + ("CURRENT mapping is correct" if a > b else
                     "SWAPPED mapping is correct - left and right are reversed"))


if __name__ == "__main__":
    main()
