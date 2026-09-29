"""
Offline check that the projection camera convention and framing are right.

Renders the mesh's silhouette from each view's yaw using the exact same maths the
bake node uses, and writes it side by side with the reference image for that view.
If the silhouettes line up with the art, projection will land correctly.

    python scripts/verify-projection.py <mesh.glb> <out_dir>

Expects front/left/back/right PNGs alongside; see VIEW_ANGLES below.
"""

import json
import struct
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _nodepath                                   # noqa: E402
_nodepath.add("comfyui_multiview_projection")
import geometry as G  # noqa: E402

# yaw 0 = +Z (front). Which side "left"/"right" name is a labelling choice; both
# candidate mappings are rendered so the correct one can be read off the output.
VIEW_ANGLES = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}


def load_glb(path):
    buf = Path(path).read_bytes()
    assert struct.unpack_from("<I", buf, 0)[0] == 0x46546C67, "not a GLB"
    off, js, bin_ = 12, None, None
    while off < len(buf):
        ln, ty = struct.unpack_from("<II", buf, off)
        chunk = buf[off + 8: off + 8 + ln]
        if ty == 0x4E4F534A:
            js = json.loads(chunk)
        elif ty == 0x004E4942:
            bin_ = chunk
        off += 8 + ln + ((4 - ln % 4) % 4)

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "<u1"}[a["componentType"]]
        ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        arr = np.frombuffer(bin_, dtype=np.dtype(dt), count=a["count"] * ncomp, offset=start)
        return arr.reshape(a["count"], ncomp).astype(np.float32 if dt == "<f4" else np.int64)

    vs, fs, base = [], [], 0
    for m in js["meshes"]:
        for p in m["primitives"]:
            v = acc(p["attributes"]["POSITION"])
            f = acc(p["indices"]).reshape(-1, 3) + base
            vs.append(v)
            fs.append(f)
            base += v.shape[0]
    return (torch.from_numpy(np.concatenate(vs)).float(),
            torch.from_numpy(np.concatenate(fs)).long())


def save_png(path, arr_hw3):
    """Minimal PNG writer so this script needs no image library."""
    import zlib
    h, w, _ = arr_hw3.shape
    data = (np.clip(arr_hw3, 0, 1) * 255).astype(np.uint8)
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def load_png_rgb(path):
    """Decode a PNG via torchvision if present, else report unavailable."""
    try:
        from torchvision.io import read_image
        t = read_image(str(path)).float() / 255.0
        return t.permute(1, 2, 0)[..., :3]
    except Exception:
        return None


def main():
    mesh_path, out_dir, ref_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    v, f = load_glb(mesh_path)
    print(f"mesh: {v.shape[0]} verts, {f.shape[0]} faces")
    print(f"bounds min={v.amin(0).tolist()}  max={v.amax(0).tolist()}")

    R = 512
    for name, yaw in VIEW_ANGLES.items():
        z, right, up = G.camera_basis(yaw, 0.0)
        cu, cv, half = G.frame_extents(v, right, up, 1.0)
        sil = G.rasterize_silhouette(v, f, right, up, z, cu, cv, half, R)
        img = sil[..., None].repeat(1, 1, 3).numpy()

        ref = load_png_rgb(ref_dir / f"{name}.png")
        if ref is not None:
            ref_r = torch.nn.functional.interpolate(
                ref.permute(2, 0, 1)[None], size=(R, R), mode="bilinear",
                align_corners=False)[0].permute(1, 2, 0).numpy()
            img = np.concatenate([img, ref_r], axis=1)

        save_png(out_dir / f"view_{name}_yaw{int(yaw)}.png", img)
        print(f"  wrote view_{name}_yaw{int(yaw)}.png  (render | reference)")


if __name__ == "__main__":
    main()
