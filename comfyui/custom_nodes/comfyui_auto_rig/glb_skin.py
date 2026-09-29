"""
Write a skinned glTF 2.0 binary (.glb).

ComfyUI's own save_glb writes static geometry only -- it has no skin, no joint
nodes and no inverse bind matrices -- so a rigged export needs its own writer.
This one is deliberately minimal and dependency-free: one buffer, one mesh, one
skin, one material.

Bind pose convention
--------------------
Every joint node carries a translation and an identity rotation, so a joint's
global bind transform is a pure translation to its fitted position and the
inverse bind matrix is just the negated translation. That keeps the rest pose
identical to the mesh as generated, which is what a retargeter (Unity Humanoid,
Blender, Godot) expects to be handed.
"""

import io
import json
import struct

import numpy as np

GL_USHORT = 5123
GL_UINT = 5125
GL_FLOAT = 5126
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def _vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


class _Buffer:
    """Accumulates the binary chunk and hands back bufferView indices."""

    def __init__(self):
        self.blob = bytearray()
        self.views = []

    def add(self, array, target=None):
        # glTF requires each accessor's data to start on a 4-byte boundary.
        while len(self.blob) % 4:
            self.blob.append(0)
        offset = len(self.blob)
        raw = np.ascontiguousarray(array).tobytes()
        self.blob.extend(raw)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1


def save_rigged_glb(vertices, faces, joints, parents, names, joint_idx, joint_w,
                    uvs=None, normals=None, base_color=None,
                    prefix="mixamorig:", metadata=None):
    """Serialise a skinned mesh to GLB bytes.

    vertices [N,3], faces [M,3], joints [B,3], parents [B] (-1 for the root),
    names [B] str, joint_idx [N,4] int, joint_w [N,4] float.
    `base_color` is an optional HxWx3/4 uint8 texture, requiring `uvs`.
    """
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)
    ji = np.ascontiguousarray(joint_idx, dtype=np.uint16)
    jw = np.ascontiguousarray(joint_w, dtype=np.float32)
    nb = len(joints)

    if ji.shape[0] != len(verts) or jw.shape[0] != len(verts):
        raise ValueError("skin weights must be 1:1 with vertices")
    if int(ji.max(initial=0)) >= nb:
        raise ValueError("skin references joint %d but the rig has %d"
                         % (int(ji.max()), nb))

    if normals is None:
        normals = _vertex_normals(verts.astype(np.float64), faces.astype(np.int64))
    normals = np.ascontiguousarray(normals, dtype=np.float32)

    buf = _Buffer()
    accessors = []

    def accessor(view, ctype, count, atype, minmax=None, normalized=False):
        a = {"bufferView": view, "componentType": ctype, "count": int(count),
             "type": atype}
        if minmax is not None:
            a["min"], a["max"] = minmax
        if normalized:
            a["normalized"] = True
        accessors.append(a)
        return len(accessors) - 1

    a_pos = accessor(buf.add(verts, ARRAY_BUFFER), GL_FLOAT, len(verts), "VEC3",
                     [verts.min(axis=0).tolist(), verts.max(axis=0).tolist()])
    a_nrm = accessor(buf.add(normals, ARRAY_BUFFER), GL_FLOAT, len(verts), "VEC3")
    a_jnt = accessor(buf.add(ji, ARRAY_BUFFER), GL_USHORT, len(verts), "VEC4")
    a_wgt = accessor(buf.add(jw, ARRAY_BUFFER), GL_FLOAT, len(verts), "VEC4")
    a_idx = accessor(buf.add(faces.reshape(-1), ELEMENT_ARRAY_BUFFER),
                     GL_UINT, faces.size, "SCALAR")

    attributes = {"POSITION": a_pos, "NORMAL": a_nrm,
                  "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt}
    a_uv = None
    if uvs is not None:
        uv = np.ascontiguousarray(uvs, dtype=np.float32)[:, :2]
        if len(uv) != len(verts):
            raise ValueError("uvs must be 1:1 with vertices")
        a_uv = accessor(buf.add(uv, ARRAY_BUFFER), GL_FLOAT, len(verts), "VEC2")
        attributes["TEXCOORD_0"] = a_uv

    # Inverse bind matrices. Rotation is identity in the bind pose, so this is
    # just the negated world translation, column-major.
    ibm = np.zeros((nb, 16), dtype=np.float32)
    ibm[:, 0] = ibm[:, 5] = ibm[:, 10] = ibm[:, 15] = 1.0
    ibm[:, 12:15] = -joints
    a_ibm = accessor(buf.add(ibm), GL_FLOAT, nb, "MAT4")

    # -- nodes -----------------------------------------------------------------
    nodes = []
    children = {i: [] for i in range(nb)}
    for i, p in enumerate(parents):
        if int(p) >= 0:
            children[int(p)].append(i)
    for i in range(nb):
        p = int(parents[i])
        local = joints[i] - (joints[p] if p >= 0 else np.zeros(3, dtype=np.float32))
        node = {"name": prefix + str(names[i]),
                "translation": [float(local[0]), float(local[1]), float(local[2])]}
        if children[i]:
            node["children"] = [int(c) for c in children[i]]
        nodes.append(node)
    roots = [i for i in range(nb) if int(parents[i]) < 0]

    mesh_node = len(nodes)
    nodes.append({"name": "Mesh", "mesh": 0, "skin": 0})

    # -- material / texture ----------------------------------------------------
    images, samplers, textures, materials = [], [], [], []
    pbr = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],
           "metallicFactor": 0.0, "roughnessFactor": 1.0}
    if base_color is not None:
        if a_uv is None:
            raise ValueError("a base colour texture needs uvs")
        from PIL import Image
        img = np.asarray(base_color)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        png = io.BytesIO()
        Image.fromarray(img[..., :3], "RGB").save(png, format="PNG")
        view = buf.add(np.frombuffer(png.getvalue(), dtype=np.uint8))
        images.append({"bufferView": view, "mimeType": "image/png"})
        samplers.append({"magFilter": 9729, "minFilter": 9987,
                         "wrapS": 10497, "wrapT": 10497})
        textures.append({"sampler": 0, "source": 0})
        pbr["baseColorTexture"] = {"index": 0}
    materials.append({"name": "Material", "pbrMetallicRoughness": pbr,
                      "doubleSided": True})

    gltf = {
        "asset": {"version": "2.0", "generator": "comfyui_auto_rig"},
        "scene": 0,
        "scenes": [{"nodes": roots + [mesh_node]}],
        "nodes": nodes,
        "meshes": [{"name": "Mesh", "primitives": [
            {"attributes": attributes, "indices": a_idx, "material": 0}]}],
        "skins": [{"name": "Armature", "joints": list(range(nb)),
                   "inverseBindMatrices": a_ibm, "skeleton": roots[0]}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.blob)}],
    }
    if images:
        gltf["images"], gltf["samplers"], gltf["textures"] = images, samplers, textures
    if metadata:
        gltf["asset"]["extras"] = metadata

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    blob = bytes(buf.blob)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)

    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(blob))
    return b"".join([header,
                     struct.pack("<II", len(js), 0x4E4F534A), js,
                     struct.pack("<II", len(blob), 0x004E4942), blob])
