"""
Convert a textured GLB to FBX, ready to upload to Mixamo's auto-rigger.

    "<blender.exe>" --background --python scripts/glb-to-fbx.py -- \
        <in.glb> <out.fbx> [--embed] [--scale 1.0]

Mixamo reads FBX and OBJ, not glTF, so the pipeline's output has to be
converted before it can be rigged there. Textures are embedded by default
(`--embed`) so the upload is a single self-contained file.

Mixamo does not preserve material bindings reliably, which is the whole reason
for the return trip in apply-textures.py: the atlas written alongside the FBX
here is what gets reattached to the rigged mesh afterwards. UVs *are* preserved
by Mixamo, so reattaching is a material assignment rather than a re-bake.
"""

import os
import sys

import bpy


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main():
    args = argv_after_dashes()
    if len(args) < 2:
        print("usage: <in.glb> <out.fbx> [--embed] [--scale N]")
        return 1
    src, dst = args[0], args[1]
    arg = lambda f, d: (args[args.index(f) + 1] if f in args else d)  # noqa: E731
    scale = float(arg("--scale", "1.0"))
    embed = "--embed" in args

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        print("ERROR: no mesh in " + src)
        return 1
    total_v = sum(len(o.data.vertices) for o in meshes)
    total_f = sum(len(o.data.polygons) for o in meshes)
    uv_ok = all(len(o.data.uv_layers) > 0 for o in meshes)
    print("imported %d mesh(es): %d verts, %d faces, uvs=%s"
          % (len(meshes), total_v, total_f, uv_ok))
    if not uv_ok:
        print("WARNING: a mesh has no UVs; textures cannot be reattached later")

    # Save the atlas next to the FBX. This is the copy the return trip uses, so
    # it does not matter what Mixamo does to the material.
    saved = []
    for img in bpy.data.images:
        if img.size[0] == 0 or img.name.lower().startswith("render result"):
            continue
        out_png = os.path.splitext(dst)[0] + "_" + \
            "".join(c if c.isalnum() else "_" for c in img.name) + ".png"
        img.file_format = "PNG"
        try:
            img.save(filepath=out_png)
            saved.append(out_png)
        except Exception as exc:
            print("WARNING: could not save %s: %s" % (img.name, exc))
    print("saved %d texture(s): %s" % (len(saved), ", ".join(os.path.basename(p) for p in saved)))

    if scale != 1.0:
        for o in meshes:
            o.scale = (scale, scale, scale)
        bpy.context.view_layer.update()

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=dst,
        use_selection=False,
        # Mixamo wants a Y-up, metre-scaled mesh; these are the settings its own
        # documentation asks for.
        axis_forward="-Z", axis_up="Y",
        apply_unit_scale=True, global_scale=1.0,
        path_mode="COPY" if embed else "AUTO",
        embed_textures=embed,
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        bake_anim=False,
    )
    print("wrote " + dst + " (%.1f MB)" % (os.path.getsize(dst) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
