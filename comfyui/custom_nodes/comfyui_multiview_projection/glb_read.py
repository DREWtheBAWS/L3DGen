"""
Read a .glb back into vertices / faces / uvs / base colour.

ComfyUI can write a GLB but has no way to read one back into a MESH, which makes
a multi-stage pipeline impossible: a run that wants to hand geometry from one
prompt to the next has nowhere to put it. This closes that loop.

Deliberately minimal -- positions, texture coordinates, triangle indices and the
base colour image, which is exactly what the projection bake consumes.
"""

import json
import struct

import numpy as np

# glTF component types -> (struct code, byte size)
COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
        5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NP_OF = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
         5123: np.uint16, 5125: np.uint32, 5126: np.float32}

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def parse_glb(path):
    """Split a GLB into its JSON and binary chunks."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 12:
        raise ValueError("not a GLB: file is too short")
    magic, version, total = struct.unpack("<4sII", blob[:12])
    if magic != b"glTF":
        raise ValueError("not a GLB: bad magic")
    if version != 2:
        raise ValueError("unsupported glTF version %d" % version)
    if total != len(blob):
        # Truncated downloads are common enough to be worth naming precisely.
        raise ValueError("GLB is truncated: header says %d bytes, file has %d"
                         % (total, len(blob)))

    off, gltf, binary = 12, None, b""
    while off + 8 <= len(blob):
        length, kind = struct.unpack("<II", blob[off:off + 8])
        data = blob[off + 8:off + 8 + length]
        if kind == JSON_CHUNK:
            gltf = json.loads(data.decode("utf-8"))
        elif kind == BIN_CHUNK:
            binary = data
        off += 8 + length
    if gltf is None:
        raise ValueError("GLB has no JSON chunk")
    return gltf, binary


def read_accessor(gltf, binary, index):
    """One accessor as a [count, components] array."""
    acc = gltf["accessors"][index]
    n = NCOMP[acc["type"]]
    ctype = acc["componentType"]
    count = acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, n), dtype=NP_OF[ctype])

    bv = gltf["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    itemsize = COMP[ctype][1] * n
    stride = bv.get("byteStride")

    if not stride or stride == itemsize:
        # Tightly packed: one read, no per-element work.
        out = np.frombuffer(binary, dtype=NP_OF[ctype],
                            count=count * n, offset=base).reshape(count, n)
        return np.array(out)
    # Interleaved: gather each element at its own stride.
    out = np.empty((count, n), dtype=NP_OF[ctype])
    for i in range(count):
        out[i] = np.frombuffer(binary, dtype=NP_OF[ctype], count=n,
                               offset=base + i * stride)
    return out


def read_image(gltf, binary, index):
    """A glTF image as HxWx3 float in 0..1, or None."""
    try:
        from PIL import Image
    except Exception:
        return None
    import io

    img = gltf["images"][index]
    if "bufferView" in img:
        bv = gltf["bufferViews"][img["bufferView"]]
        raw = binary[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        pil = Image.open(io.BytesIO(raw))
    elif "uri" in img and img["uri"].startswith("data:"):
        import base64
        raw = base64.b64decode(img["uri"].split(",", 1)[1])
        pil = Image.open(io.BytesIO(raw))
    else:
        return None                      # external file: not supported
    return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def load_mesh(path):
    """Everything the projection bake needs from a GLB.

    Returns (vertices [N,3], faces [M,3], uvs [N,2] or None,
             base_colour [H,W,3] or None).

    All primitives are concatenated, with indices offset, so a model split by
    material still comes back as one mesh -- the bake works on the whole surface
    and would otherwise silently texture only the first material's share of it.
    """
    gltf, binary = parse_glb(path)
    meshes = gltf.get("meshes") or []
    if not meshes:
        raise ValueError("GLB contains no meshes")

    vs, fs, us = [], [], []
    base = 0
    texture = None
    have_uv = True

    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "POSITION" not in prim.get("attributes", {}):
                continue
            v = read_accessor(gltf, binary, attrs["POSITION"]).astype(np.float32)
            if "indices" in prim:
                idx = read_accessor(gltf, binary, prim["indices"]).reshape(-1)
            else:
                idx = np.arange(len(v), dtype=np.uint32)
            f = idx.reshape(-1, 3).astype(np.int64) + base

            if "TEXCOORD_0" in attrs:
                us.append(read_accessor(gltf, binary, attrs["TEXCOORD_0"])[:, :2]
                          .astype(np.float32))
            else:
                have_uv = False
                us.append(np.zeros((len(v), 2), dtype=np.float32))

            vs.append(v)
            fs.append(f)
            base += len(v)

            if texture is None and "material" in prim:
                mat = (gltf.get("materials") or [])[prim["material"]]
                pbr = mat.get("pbrMetallicRoughness") or {}
                bct = pbr.get("baseColorTexture")
                if bct is not None:
                    tex = (gltf.get("textures") or [])[bct["index"]]
                    if "source" in tex:
                        texture = read_image(gltf, binary, tex["source"])

    if not vs:
        raise ValueError("GLB has no drawable primitives")

    return (np.concatenate(vs, axis=0),
            np.concatenate(fs, axis=0),
            np.concatenate(us, axis=0) if have_uv else None,
            texture)
