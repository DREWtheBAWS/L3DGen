"""
Render a finished GLB offline, with its embedded texture.

Uses the same orthographic camera as the projection bake, so what you see here is
what the pipeline actually produced. Costs no GPU time and never touches ComfyUI
— unlike render-check.py, which rebuilds the whole graph just to append renders.

    python scripts/render-glb.py model.glb out_dir [--yaws 0,90,180,270] [--res 640]
"""

import json
import struct
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _nodepath                                   # noqa: E402
_nodepath.add("comfyui_multiview_projection")
import geometry as G  # noqa: E402


def parse_glb(path):
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
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        return np.frombuffer(bin_, dtype=np.dtype(dt),
                             count=a["count"] * nc, offset=start).reshape(a["count"], nc)

    verts, faces, uvs, base = [], [], [], 0
    tex_index = None
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            v = acc(prim["attributes"]["POSITION"]).astype(np.float32)
            f = acc(prim["indices"]).reshape(-1, 3).astype(np.int64) + base
            uv = (acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
                  if "TEXCOORD_0" in prim["attributes"]
                  else np.zeros((v.shape[0], 2), np.float32))
            verts.append(v); faces.append(f); uvs.append(uv); base += v.shape[0]
            mi = prim.get("material")
            if tex_index is None and mi is not None:
                pbr = js["materials"][mi].get("pbrMetallicRoughness", {})
                if "baseColorTexture" in pbr:
                    tex_index = js["textures"][pbr["baseColorTexture"]["index"]]["source"]

    tex = None
    if tex_index is not None:
        im = js["images"][tex_index]
        bv = js["bufferViews"][im["bufferView"]]
        start = bv.get("byteOffset", 0)
        data = bin_[start:start + bv["byteLength"]]
        try:
            from torchvision.io import decode_image
            t = decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8))
            tex = t.float().permute(1, 2, 0)[..., :3] / 255.0
        except Exception as e:                                # pragma: no cover
            print(f"  (could not decode texture: {e})")

    return (torch.from_numpy(np.concatenate(verts)),
            torch.from_numpy(np.concatenate(faces)),
            torch.from_numpy(np.concatenate(uvs)),
            tex)


def write_png(path, rgb):
    h, w, _ = rgb.shape
    data = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b""))


def main():
    src, out_dir = sys.argv[1], Path(sys.argv[2])
    arg = lambda f, d: (sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d)  # noqa: E731
    yaws = [float(y) for y in arg("--yaws", "0,90,180,270").split(",")]
    res = int(arg("--res", "640"))
    pitch = float(arg("--pitch", "0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    v, f, uv, tex = parse_glb(src)
    print(f"{Path(src).name}: {v.shape[0]} verts, {f.shape[0]} faces, "
          f"texture {'yes ' + str(tuple(tex.shape)) if tex is not None else 'none'}")

    bg = torch.tensor([0.125, 0.14, 0.18])
    for yaw in yaws:
        z, right, up = G.camera_basis(yaw, pitch)
        cu, cv, half = G.frame_extents(v, right, up, 1.05)
        face_idx, bary, cov = G.rasterize_screen(v, f, right, up, z, cu, cv, half, res)

        img = bg[None, None, :].expand(res, res, 3).clone()
        if cov.any():
            vtri = f[face_idx[cov]]
            bsel = bary[cov]
            if tex is not None:
                uvp = (bsel[:, :, None] * uv[vtri]).sum(1)
                th, tw = tex.shape[0], tex.shape[1]
                px = uvp[:, 0].clamp(0, 1) * (tw - 1)
                py = uvp[:, 1].clamp(0, 1) * (th - 1)
                col = G.sample_bilinear(tex, px, py)
            else:
                col = torch.full((int(cov.sum()), 3), 0.75)
            # Gentle facing shade so form reads even on a flat albedo.
            vn = G.vertex_normals(v, f)
            nn = torch.nn.functional.normalize((bsel[:, :, None] * vn[vtri]).sum(1),
                                               dim=-1, eps=1e-6)
            img[cov] = col * (0.55 + 0.45 * (nn @ z).clamp(0, 1))[:, None]

        name = out_dir / f"{Path(src).stem}_yaw{int(yaw)}.png"
        write_png(name, img.numpy())
        print(f"  wrote {name.name}  ({cov.float().mean() * 100:.1f}% of frame)")


if __name__ == "__main__":
    main()
