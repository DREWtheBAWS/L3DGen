"""
Offline check for the auto-rig fit and skinning, with no ComfyUI running.

Builds a blocky humanoid from a capsule field (the same shape class the pipeline
produces after decimation), fits the Mixamo skeleton to it and asserts the joints
land where anatomy says they should. Run it after touching rig_fit.py:

    <comfy venv python> scripts/test-autorig.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _nodepath                                   # noqa: E402
_nodepath.add("comfyui_auto_rig")

from rig_fit import J, MIXAMO_JOINTS, fit_humanoid, skin_mesh   # noqa: E402
from glb_skin import save_rigged_glb                            # noqa: E402


# ------------------------------------------------------------ test geometry --
def capsule_sdf(p, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab[None, :]), axis=1) - r


def build_humanoid(res=72, arm_drop=0.45, stylized=False):
    """Voxelised humanoid, surfaced as its own boundary quads.

    Y up, facing +Z. `arm_drop` moves the hands from a T-pose (0.0) toward the
    hips, which is what actually stresses the geodesic extremity search: a hand
    close to a thigh in space is still far along the surface.

    `stylized` builds a game character instead of a mannequin: an oversized head,
    stubby limbs, ears, a snout and a prop hanging off one hip. Every one of
    those is a surface extremity competing with the real limbs, which is the
    failure mode a bare humanoid never exposes.
    """
    lo = np.array([-0.75, -0.05, -0.55])
    hi = np.array([0.75, 1.90, 0.55])
    steps = np.array([res, int(res * 1.35), int(res * 0.75)])
    grids = [np.linspace(lo[i], hi[i], steps[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*grids, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    hand_y = 1.40 - arm_drop
    if stylized:
        # Stubby limbs, a head a quarter of the body, and the appendages a
        # creature character has. Proportions roughly match a low-poly
        # anthropomorphic animal in a T-pose.
        parts = [
            (( 0.15, 0.05, 0.0), ( 0.13, 0.72, 0.0), 0.13),    # left leg
            ((-0.15, 0.05, 0.0), (-0.13, 0.72, 0.0), 0.13),    # right leg
            (( 0.00, 0.78, 0.0), ( 0.00, 1.24, 0.0), 0.23),    # barrel torso / tunic
            (( 0.00, 1.28, 0.0), ( 0.00, 1.60, 0.0), 0.26),    # oversized head
            (( 0.00, 1.42, 0.20), ( 0.00, 1.40, 0.34), 0.10),  # snout (forward)
            (( 0.17, 1.58, 0.0), ( 0.30, 1.86, 0.0), 0.055),   # left ear
            ((-0.17, 1.58, 0.0), (-0.30, 1.86, 0.0), 0.055),   # right ear
            (( 0.21, 1.18, 0.0), ( 0.62, 1.18 - arm_drop, 0.0), 0.085),   # left arm
            ((-0.21, 1.18, 0.0), (-0.62, 1.18 - arm_drop, 0.0), 0.085),   # right arm
            (( 0.15, 0.05, 0.02), ( 0.15, 0.05, 0.22), 0.10),  # left foot
            ((-0.15, 0.05, 0.02), (-0.15, 0.05, 0.22), 0.10),  # right foot
            (( 0.00, 0.70, -0.22), ( 0.00, 0.62, -0.36), 0.05),  # tail
            (( 0.26, 0.78, 0.10), ( 0.30, 0.52, 0.14), 0.075),   # prop on one hip
        ]
    else:
        parts = [
            (( 0.13, 0.05, 0.0), ( 0.11, 0.92, 0.0), 0.10),    # left leg  (+X = left)
            ((-0.13, 0.05, 0.0), (-0.11, 0.92, 0.0), 0.10),    # right leg
            (( 0.00, 0.85, 0.0), ( 0.00, 1.46, 0.0), 0.17),    # torso
            (( 0.00, 1.50, 0.0), ( 0.00, 1.62, 0.0), 0.145),   # head
            (( 0.16, 1.42, 0.0), ( 0.56, hand_y, 0.0), 0.065),  # left arm
            ((-0.16, 1.42, 0.0), (-0.56, hand_y, 0.0), 0.065),  # right arm
            (( 0.13, 0.05, 0.02), ( 0.13, 0.05, 0.17), 0.075),  # left foot (forward +Z)
            ((-0.13, 0.05, 0.02), (-0.13, 0.05, 0.17), 0.075),  # right foot
        ]
    sdf = np.full(len(pts), 1e9)
    for a, b, r in parts:
        np.minimum(sdf, capsule_sdf(pts, a, b, r), out=sdf)
    occ = (sdf <= 0.0).reshape(steps)

    # Surface as the boundary quads of the occupied voxels, vertices welded on
    # the integer lattice so the mesh comes out connected.
    vmap, verts, faces = {}, [], []

    def vid(i, j, k):
        key = (i, j, k)
        got = vmap.get(key)
        if got is None:
            got = len(verts)
            vmap[key] = got
            verts.append([lo[0] + i * (hi[0] - lo[0]) / (steps[0] - 1),
                          lo[1] + j * (hi[1] - lo[1]) / (steps[1] - 1),
                          lo[2] + k * (hi[2] - lo[2]) / (steps[2] - 1)])
        return got

    padded = np.zeros(np.array(steps) + 2, dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = occ
    quads = [((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
             ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
             ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
             ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
             ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
             ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])]
    idx = np.argwhere(occ)
    for (i, j, k) in idx:
        for (d, corners) in quads:
            if padded[i + 1 + d[0], j + 1 + d[1], k + 1 + d[2]]:
                continue
            c = [vid(i + o[0], j + o[1], k + o[2]) for o in corners]
            faces.append([c[0], c[1], c[2]])
            faces.append([c[0], c[2], c[3]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


# ------------------------------------------------------------------- checks --
def main():
    failures = []

    def check(label, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + label + (("  -- " + detail) if detail else ""))
        if not cond:
            failures.append(label)

    for arm_drop, stylized in ((0.0, False), (0.45, False), (0.0, True), (0.35, True)):
        pose = ("stylized " if stylized else "") + ("T-pose" if arm_drop == 0.0 else "A-pose")
        print("\n=== %s ===" % pose)
        t0 = time.time()
        verts, faces = build_humanoid(arm_drop=arm_drop, stylized=stylized)
        print("mesh: %d verts, %d tris (%.1fs)" % (len(verts), len(faces), time.time() - t0))

        t0 = time.time()
        fit = fit_humanoid(verts, faces)
        Jp = fit["joints"]
        H = fit["height"]
        print("fit: %.1fs  confidence %.2f  notes=%s"
              % (time.time() - t0, fit["confidence"], fit["notes"] or "none"))

        # The subject faces +Z with up +Y, so its own left is +X.
        check("LeftHand on +X", Jp[J["LeftHand"]][0] > 0.1,
              "x=%.3f" % Jp[J["LeftHand"]][0])
        check("RightHand on -X", Jp[J["RightHand"]][0] < -0.1,
              "x=%.3f" % Jp[J["RightHand"]][0])
        check("LeftFoot on +X", Jp[J["LeftFoot"]][0] > 0.02,
              "x=%.3f" % Jp[J["LeftFoot"]][0])
        y0 = verts[:, 1].min()
        hips_f = (Jp[J["Hips"]][1] - y0) / H
        check("hips in the mid-body band", 0.33 <= hips_f <= 0.62,
              "%.0f%% of height (y=%.3f)" % (100 * hips_f, Jp[J["Hips"]][1]))
        check("head above neck above chest",
              Jp[J["Head"]][1] > Jp[J["Neck"]][1] > Jp[J["Spine2"]][1],
              "head=%.2f neck=%.2f chest=%.2f" % (Jp[J["Head"]][1], Jp[J["Neck"]][1],
                                                  Jp[J["Spine2"]][1]))
        check("knee between hip and ankle",
              Jp[J["LeftUpLeg"]][1] > Jp[J["LeftLeg"]][1] > Jp[J["LeftFoot"]][1],
              "hip=%.2f knee=%.2f ankle=%.2f" % (Jp[J["LeftUpLeg"]][1],
                                                 Jp[J["LeftLeg"]][1], Jp[J["LeftFoot"]][1]))
        check("elbow between shoulder and hand",
              abs(Jp[J["LeftForeArm"]][0]) > abs(Jp[J["LeftArm"]][0])
              and abs(Jp[J["LeftHand"]][0]) > abs(Jp[J["LeftForeArm"]][0]),
              "sh=%.2f el=%.2f hand=%.2f" % (Jp[J["LeftArm"]][0],
                                             Jp[J["LeftForeArm"]][0], Jp[J["LeftHand"]][0]))
        check("toe forward of ankle (+Z)",
              Jp[J["LeftToe_End"]][2] > Jp[J["LeftFoot"]][2],
              "toe=%.3f ankle=%.3f" % (Jp[J["LeftToe_End"]][2], Jp[J["LeftFoot"]][2]))
        # Symmetry is enforced about the mesh's own symmetry plane, which is not
        # necessarily x=0, so measure against the plane the fit reports.
        cx = fit["centre_x"]
        check("rig is symmetric about the fitted plane",
              abs((Jp[J["LeftHand"]][0] - cx) + (Jp[J["RightHand"]][0] - cx)) < 1e-4
              and abs(Jp[J["LeftHand"]][1] - Jp[J["RightHand"]][1]) < 1e-4,
              "cx=%.4f" % cx)
        check("spine on the symmetry plane",
              all(abs(Jp[J[n]][0] - cx) < 1e-4 for n in ("Hips", "Spine", "Neck", "Head")))
        # Spine2 is the upper chest: it should land in the top half of the
        # hips-to-neck run, not at the waist and not merged into the neck.
        trunk = Jp[J["Neck"]][1] - Jp[J["Hips"]][1]
        chest_f = (Jp[J["Spine2"]][1] - Jp[J["Hips"]][1]) / max(trunk, 1e-6)
        check("chest in the upper trunk", 0.45 <= chest_f <= 0.92,
              "%.0f%% of hips->neck (hips=%.2f chest=%.2f neck=%.2f)"
              % (100 * chest_f, Jp[J["Hips"]][1], Jp[J["Spine2"]][1], Jp[J["Neck"]][1]))
        check("neck between chest and crown",
              Jp[J["Spine2"]][1] < Jp[J["Neck"]][1] < Jp[J["HeadTop_End"]][1] - 0.05 * H,
              "chest=%.2f neck=%.2f top=%.2f" % (Jp[J["Spine2"]][1], Jp[J["Neck"]][1],
                                                 Jp[J["HeadTop_End"]][1]))
        check("thighs offset from the centre line",
              abs(Jp[J["LeftUpLeg"]][0] - cx) > 0.02 * H,
              "x offset %.3f" % abs(Jp[J["LeftUpLeg"]][0] - cx))
        check("shoulder between chest and elbow",
              abs(Jp[J["LeftShoulder"]][0] - cx) < abs(Jp[J["LeftArm"]][0] - cx)
              < abs(Jp[J["LeftForeArm"]][0] - cx),
              "clav=%.2f sh=%.2f el=%.2f" % (Jp[J["LeftShoulder"]][0], Jp[J["LeftArm"]][0],
                                             Jp[J["LeftForeArm"]][0]))
        # A joint outside the mesh means the bone rotates about empty space.
        d = np.linalg.norm(verts[None, :, :] - Jp[:, None, :], axis=2).min(axis=1)
        check("every joint sits inside the surface", float(d.max()) < 0.12 * H,
              "worst gap %.3f (%.0f%% of height)" % (d.max(), 100 * d.max() / H))

        t0 = time.time()
        ji, jw = skin_mesh(verts, faces, Jp, fit["parents"])
        print("skin: %.1fs" % (time.time() - t0))
        check("weights normalised", np.allclose(jw.sum(axis=1), 1.0, atol=1e-4),
              "min=%.4f max=%.4f" % (jw.sum(axis=1).min(), jw.sum(axis=1).max()))
        check("at most 4 influences", ji.shape[1] == 4)

        # The decisive skinning test: a hand vertex must not be driven by the
        # leg it is sitting next to in space.
        hand = Jp[J["LeftHand"]]
        near_hand = np.linalg.norm(verts - hand[None, :], axis=1) < 0.09
        leg_joints = {J["LeftUpLeg"], J["LeftLeg"], J["RightUpLeg"], J["RightLeg"]}
        bleed = 0.0
        if near_hand.sum():
            mask = np.isin(ji[near_hand], list(leg_joints))
            bleed = float((jw[near_hand] * mask).sum(axis=1).max())
        check("no hand->leg weight bleed", bleed < 0.05,
              "%d hand verts, worst leg weight %.3f" % (int(near_hand.sum()), bleed))

        arm_joints = {J["LeftShoulder"], J["LeftArm"], J["LeftForeArm"], J["LeftHand"]}
        if near_hand.sum():
            m = np.isin(ji[near_hand], list(arm_joints))
            own = float((jw[near_hand] * m).sum(axis=1).min())
            check("hand verts owned by the arm chain", own > 0.8,
                  "weakest %.3f" % own)

    # -------------------------------------------------------- GLB export ----
    print("\n=== rigged GLB ===")
    verts, faces = build_humanoid(arm_drop=0.45)
    fit = fit_humanoid(verts, faces)
    ji, jw = skin_mesh(verts, faces, fit["joints"], fit["parents"])
    uvs = np.stack([(verts[:, 0] - verts[:, 0].min()) / np.ptp(verts[:, 0]),
                    (verts[:, 1] - verts[:, 1].min()) / np.ptp(verts[:, 1])], axis=1)
    tex = np.zeros((64, 64, 3), dtype=np.uint8)
    tex[:, :, 0] = 200
    blob = save_rigged_glb(verts, faces, fit["joints"], fit["parents"], fit["names"],
                           ji, jw, uvs=uvs, base_color=tex)
    print("glb: %.2f MB" % (len(blob) / 1e6))

    gltf, bin_chunk = parse_glb(blob)
    check("glb header and chunk sizes agree", gltf is not None)
    check("declared buffer matches the binary chunk",
          gltf["buffers"][0]["byteLength"] <= len(bin_chunk),
          "%d vs %d" % (gltf["buffers"][0]["byteLength"], len(bin_chunk)))

    for i, bv in enumerate(gltf["bufferViews"]):
        end = bv["byteOffset"] + bv["byteLength"]
        if end > len(bin_chunk) or bv["byteOffset"] % 4:
            failures.append("bufferView %d out of range or misaligned" % i)
    check("every bufferView is in range and 4-byte aligned",
          not any("bufferView" in f for f in failures))

    sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    comp = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
    ok = True
    for i, a in enumerate(gltf["accessors"]):
        need = a["count"] * sizes[a["type"]] * comp[a["componentType"]]
        if need > gltf["bufferViews"][a["bufferView"]]["byteLength"]:
            print("  accessor %d overruns its bufferView" % i)
            ok = False
    check("every accessor fits its bufferView", ok)

    prim = gltf["meshes"][0]["primitives"][0]
    check("primitive has skin attributes",
          "JOINTS_0" in prim["attributes"] and "WEIGHTS_0" in prim["attributes"])
    check("JOINTS_0 is an integer type",
          gltf["accessors"][prim["attributes"]["JOINTS_0"]]["componentType"] == 5123)
    check("index count is a multiple of 3",
          gltf["accessors"][prim["indices"]]["count"] % 3 == 0)

    skin = gltf["skins"][0]
    check("skin joint count matches the skeleton",
          len(skin["joints"]) == len(MIXAMO_JOINTS),
          "%d joints" % len(skin["joints"]))
    check("inverse bind matrices present and sized",
          gltf["accessors"][skin["inverseBindMatrices"]]["count"] == len(MIXAMO_JOINTS))
    check("mesh node references the skin",
          any(n.get("skin") == 0 and "mesh" in n for n in gltf["nodes"]))

    names = [gltf["nodes"][j]["name"] for j in skin["joints"]]
    check("all joints carry the mixamorig prefix",
          all(n.startswith("mixamorig:") for n in names))
    expected = {"mixamorig:" + n for n, _ in MIXAMO_JOINTS}
    check("joint names match the Mixamo set exactly", set(names) == expected,
          "missing=%s" % (expected - set(names)))

    # A cycle or an orphan in the hierarchy makes the skeleton unusable.
    child_of = {}
    for i, n in enumerate(gltf["nodes"]):
        for c in n.get("children", []):
            child_of[c] = i
    roots = [j for j in skin["joints"] if j not in child_of]
    check("exactly one skeleton root", len(roots) == 1, "roots=%s" % roots)
    depth_ok = True
    for j in skin["joints"]:
        seen, cur = set(), j
        while cur in child_of:
            if cur in seen:
                depth_ok = False
                break
            seen.add(cur)
            cur = child_of[cur]
    check("hierarchy is acyclic", depth_ok)

    # The inverse bind matrix must undo the accumulated local translations,
    # otherwise the mesh explodes the moment a skin is applied.
    ibm_acc = gltf["accessors"][skin["inverseBindMatrices"]]
    bv = gltf["bufferViews"][ibm_acc["bufferView"]]
    ibm = np.frombuffer(bin_chunk, dtype=np.float32, count=len(skin["joints"]) * 16,
                        offset=bv["byteOffset"]).reshape(-1, 16)
    world = {}
    for k, j in enumerate(skin["joints"]):
        t = np.array(gltf["nodes"][j]["translation"], dtype=np.float64)
        p = child_of.get(j)
        world[j] = t + (world[p] if p is not None else 0.0)
    err = max(float(np.abs(ibm[k, 12:15] + world[j]).max())
              for k, j in enumerate(skin["joints"]))
    check("inverse bind matrices invert the bind pose", err < 1e-5, "max error %.2e" % err)

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


def parse_glb(blob):
    """Minimal GLB reader, so the export is validated rather than assumed."""
    import json
    import struct
    magic, version, total = struct.unpack("<4sII", blob[:12])
    if magic != b"glTF" or version != 2 or total != len(blob):
        return None, b""
    off, gltf, bin_chunk = 12, None, b""
    while off < len(blob):
        length, kind = struct.unpack("<II", blob[off:off + 8])
        data = blob[off + 8:off + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(data.decode("utf-8"))
        elif kind == 0x004E4942:
            bin_chunk = data
        off += 8 + length
    return gltf, bin_chunk


if __name__ == "__main__":
    sys.exit(main())
