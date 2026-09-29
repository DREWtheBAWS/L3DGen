"""
Fit the Mixamo rig to an existing .glb and report, offline.

Lets a real generated mesh be exercised against the fitter without occupying the
GPU or waiting on a pipeline run -- which is the difference between iterating on
the fit in seconds and in minutes.

    <comfy venv python> scripts/rig-glb.py <file.glb> [--preview out.png]
"""

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _nodepath                                   # noqa: E402
_nodepath.add("comfyui_auto_rig")

from rig_fit import J, fit_humanoid, skin_mesh   # noqa: E402

COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
        5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path):
    """Positions and triangle indices of the first primitive."""
    blob = open(path, "rb").read()
    magic, _ver, _total = struct.unpack("<4sII", blob[:12])
    if magic != b"glTF":
        raise ValueError("not a GLB")
    off, gltf, bin_chunk = 12, None, b""
    while off < len(blob):
        length, kind = struct.unpack("<II", blob[off:off + 8])
        data = blob[off + 8:off + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(data.decode("utf-8"))
        elif kind == 0x004E4942:
            bin_chunk = data
        off += 8 + length

    def read(acc_i):
        acc = gltf["accessors"][acc_i]
        bv = gltf["bufferViews"][acc["bufferView"]]
        fmt, size = COMP[acc["componentType"]]
        n = NCOMP[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride") or (size * n)
        out = np.empty((acc["count"], n), dtype=np.float64 if fmt == "f" else np.int64)
        for i in range(acc["count"]):
            o = start + i * stride
            out[i] = struct.unpack_from("<" + fmt * n, bin_chunk, o)
        return out

    prim = gltf["meshes"][0]["primitives"][0]
    verts = read(prim["attributes"]["POSITION"]).astype(np.float64)
    faces = read(prim["indices"]).reshape(-1, 3).astype(np.int64)
    return verts, faces


def main():
    path = sys.argv[1]
    verts, faces = load_glb(path)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    print("%s\n  %d verts, %d tris" % (os.path.basename(path), len(verts), len(faces)))
    print("  bbox X %.3f..%.3f  Y %.3f..%.3f  Z %.3f..%.3f" %
          (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    ext = hi - lo
    print("  extents  X %.3f  Y %.3f  Z %.3f   (tallest axis: %s)"
          % (ext[0], ext[1], ext[2], "XYZ"[int(np.argmax(ext))]))

    fit = fit_humanoid(verts, faces)
    Jp, H = fit["joints"], fit["height"]
    print("  confidence %.2f  centre_x %.4f" % (fit["confidence"], fit["centre_x"]))
    for n in fit["notes"]:
        print("  note: " + n)

    def rel(name):
        p = Jp[J[name]]
        return "%-14s x=%+.3f y=%+.3f z=%+.3f   (%.0f%% up)" % (
            name, p[0], p[1], p[2], 100 * (p[1] - lo[1]) / max(ext[1], 1e-9))

    for n in ("Hips", "Spine2", "Neck", "HeadTop_End", "LeftShoulder", "LeftArm",
              "LeftForeArm", "LeftHand", "RightHand", "LeftUpLeg", "LeftLeg",
              "LeftFoot", "LeftToe_End"):
        print("  " + rel(n))

    d = np.linalg.norm(verts[None, :, :] - Jp[:, None, :], axis=2).min(axis=1)
    worst = int(np.argmax(d))
    print("  furthest joint from any surface: %s at %.1f%% of height"
          % (fit["names"][worst], 100 * d.max() / max(ext[1], 1e-9)))

    ji, jw = skin_mesh(verts, faces, Jp, fit["parents"])
    print("  skin: weights sum to 1 -> %s, influences %d"
          % (bool(np.allclose(jw.sum(axis=1), 1.0, atol=1e-4)), ji.shape[1]))

    if "--preview" in sys.argv:
        out = sys.argv[sys.argv.index("--preview") + 1]
        render_preview(verts, {"joints": Jp, "parents": fit["parents"],
                               "joint_idx": ji}, out)
        print("  preview -> " + out)


PALETTE = np.array([
    [.95, .26, .21], [.91, .12, .39], [.61, .15, .69], [.40, .23, .72], [.25, .32, .71],
    [.13, .59, .95], [.01, .66, .96], [.00, .74, .83], [.00, .59, .53], [.30, .69, .31],
    [.55, .76, .29], [.80, .86, .22], [1, .92, .23], [1, .76, .03], [1, .60, 0],
    [1, .44, .26], [.47, .33, .28], [.62, .62, .62], [.38, .49, .55], [.90, .45, .45],
    [.45, .90, .65], [.45, .65, .90], [.90, .65, .45], [.65, .45, .90], [.75, .90, .45],
], dtype=np.float32)


def render_preview(verts, rig, out, res=620):
    from PIL import Image

    def draw(axis, mode):
        img = np.full((res, res, 3), 0.10, np.float32)
        jt = np.asarray(rig["joints"], float)
        allp = np.concatenate([verts, jt], 0)
        h = 0 if axis == "front" else 2
        lo, hi = allp.min(0), allp.max(0)
        ctr = .5 * (lo + hi)
        half = .55 * max(hi[h] - lo[h], hi[1] - lo[1], 1e-6)

        def px(p):
            x = (p[..., h] - ctr[h]) / half * .5 + .5
            y = .5 - (p[..., 1] - ctr[1]) / half * .5
            return ((np.clip(x, 0, 1) * (res - 1)).astype(int),
                    (np.clip(y, 0, 1) * (res - 1)).astype(int))

        vx, vy = px(verts)
        col = (PALETTE[np.asarray(rig["joint_idx"])[:, 0] % len(PALETTE)]
               if mode == "weights" else np.full((len(verts), 3), 0.42, np.float32))
        s = max(1, res // 500)
        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                img[np.clip(vy + dy, 0, res - 1), np.clip(vx + dx, 0, res - 1)] = col
        jx, jy = px(jt)
        for i, p in enumerate(np.asarray(rig["parents"])):
            if p < 0:
                continue
            n = max(abs(int(jx[i] - jx[p])), abs(int(jy[i] - jy[p])), 1)
            t = np.linspace(0, 1, n * 2 + 1)
            lx = np.round(jx[p] + (jx[i] - jx[p]) * t).astype(int)
            ly = np.round(jy[p] + (jy[i] - jy[p]) * t).astype(int)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    img[np.clip(ly + dy, 0, res - 1), np.clip(lx + dx, 0, res - 1)] = [1, .85, .10]
        r = max(2, res // 200)
        for i in range(len(jt)):
            img[max(0, jy[i] - r):jy[i] + r + 1, max(0, jx[i] - r):jx[i] + r + 1] = [.15, 1, .55]
        return img

    img = np.concatenate([draw("front", "skeleton"), draw("side", "skeleton"),
                          draw("front", "weights")], axis=1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out)


if __name__ == "__main__":
    main()
