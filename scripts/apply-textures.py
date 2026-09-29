"""
Put the pipeline's textures back on a mesh that came back rigged from Mixamo.

    "<blender.exe>" --background --python scripts/apply-textures.py -- \
        <rigged.fbx> <original.glb> <out.glb> [--fbx out.fbx] [--report]

Mixamo rigs the mesh but does not bring the material through, so the download
arrives grey. It does preserve the UV layout, which is the part that matters:
reattaching the atlas is then a material assignment, not a re-bake, and is exact.

That assumption is checked rather than trusted -- `--report` compares the UV
layouts of the two meshes and says how far apart they are. If Mixamo ever starts
re-unwrapping, this is where it would show up, and a proximity transfer would be
needed instead.
"""

import os
import sys

import bpy


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def uv_bounds(mesh):
    """(min_u, min_v, max_u, max_v) of a mesh's active UV layer."""
    uvs = mesh.uv_layers.active
    if uvs is None or len(uvs.data) == 0:
        return None
    us = [d.uv[0] for d in uvs.data]
    vs = [d.uv[1] for d in uvs.data]
    return (min(us), min(vs), max(us), max(vs))


def main():
    args = argv_after_dashes()
    if len(args) < 3:
        print("usage: <rigged.fbx> <original.glb> <out.glb> [--fbx out.fbx] [--report]")
        return 1
    rigged, original, out_glb = args[0], args[1], args[2]
    arg = lambda f, d: (args[args.index(f) + 1] if f in args else d)  # noqa: E731
    out_fbx = arg("--fbx", None)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # -- the rigged mesh, as Mixamo returned it --------------------------------
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=rigged, automatic_bone_orientation=True)
    rig_objs = [o for o in bpy.data.objects if o not in before]
    rig_meshes = [o for o in rig_objs if o.type == "MESH"]
    arm = next((o for o in rig_objs if o.type == "ARMATURE"), None)
    if not rig_meshes:
        print("ERROR: no mesh in " + rigged)
        return 1
    print("rigged: %d mesh(es), armature=%s (%d bones)"
          % (len(rig_meshes), arm.name if arm else "NONE",
             len(arm.data.bones) if arm else 0))
    if arm is None:
        print("WARNING: no armature found - is this really the rigged download?")

    # -- the original, for its texture -----------------------------------------
    # Either format: the donor is whatever was uploaded to Mixamo, and that may
    # have been an FBX rather than the pipeline's GLB.
    before = set(bpy.data.objects)
    if original.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=original)
    else:
        bpy.ops.import_scene.gltf(filepath=original)
    orig_objs = [o for o in bpy.data.objects if o not in before]
    orig_meshes = [o for o in orig_objs if o.type == "MESH"]
    if not orig_meshes:
        print("ERROR: no mesh in " + original)
        return 1

    # Find the base colour image on the original.
    image = None
    for o in orig_meshes:
        for slot in o.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    # The base colour is the one feeding Base Color; take the
                    # first image only as a fallback.
                    for link in mat.node_tree.links:
                        if (link.from_node is node
                                and link.to_socket.name == "Base Color"):
                            image = node.image
                            break
                    if image is None:
                        image = node.image
        if image:
            break
    if image is None:
        print("ERROR: the donor has no texture image to reapply")
        return 1
    print("texture: %s %dx%d" % (image.name, image.size[0], image.size[1]))
    if image.size[0] == 0 or image.size[1] == 0:
        # An FBX can name a texture it does not actually carry -- Mixamo's
        # download does exactly that. Assigning it would produce a model that
        # looks correct in the log and renders untextured, so refuse instead.
        print("ERROR: the donor's texture has no pixel data (%s). It names an "
              "image but does not contain one -- point --original at the "
              "pipeline GLB that produced this mesh." % (image.filepath or "no path"))
        return 1

    if "--report" in args:
        a = uv_bounds(orig_meshes[0].data)
        b = uv_bounds(rig_meshes[0].data)
        # A bare tuple on the right of % is unpacked as the argument list, so it
        # has to be wrapped to be printed as one value.
        fmt = lambda t: (tuple(round(x, 4) for x in t) if t else None)  # noqa: E731
        print("uv bounds  original %s" % (fmt(a),))
        print("uv bounds  rigged   %s" % (fmt(b),))
        print("vertex count  original %d  rigged %d"
              % (len(orig_meshes[0].data.vertices), len(rig_meshes[0].data.vertices)))
        if a and b:
            drift = max(abs(x - y) for x, y in zip(a, b))
            print("uv bound drift %.5f -- %s" % (
                drift,
                "layout preserved, a plain material assignment is exact" if drift < 1e-3
                else "LAYOUT CHANGED: a proximity transfer would be needed"))

    # -- build the material and assign it --------------------------------------
    mat = bpy.data.materials.new("Rigged")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Closest" if max(image.size) <= 256 else "Linear"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 1.0

    for o in rig_meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
        if not o.data.uv_layers:
            print("WARNING: %s has no UVs; it will render untextured" % o.name)
    print("assigned the atlas to %d rigged mesh(es)" % len(rig_meshes))

    # The original is only a texture donor; it must not reach the export.
    for o in orig_objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for o in rig_objs:
        if o.name in bpy.data.objects:
            o.select_set(True)
    if arm:
        bpy.context.view_layer.objects.active = arm

    # Bake the import transform into the data before exporting.
    #
    # Mixamo's FBX is centimetre-scaled and Z-up, so Blender's importer leaves a
    # 90-degree X rotation and a 0.01 scale on the armature. Exported as-is, that
    # becomes a root node transform in the glTF: the model is technically valid
    # but arrives lying on its side and a hundred times too small, and every
    # consumer has to know to undo it. Applying it here means the file is plain
    # metre-scaled Y-up geometry with an identity root, like the pipeline's own
    # output.
    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        print("applied the importer's root rotation and scale")
    except Exception as exc:
        print("WARNING: could not apply the root transform (%s); the export may "
              "carry a rotation and scale on its root node" % exc)

    # Drop whatever action came in with the FBX.
    #
    # A Mixamo "rigged, no animation" download still carries a two-frame
    # mixamo.com stub, and its bone translations are in Mixamo's centimetre
    # space. Applying the root transform above rescaled the objects but not the
    # keyframe values, so exporting that clip ships an animation whose
    # translations are a hundred times the rig it belongs to -- and a viewer that
    # autoplays the first clip shows a mangled mesh instead of the bind pose.
    # This step's job is a rigged, textured mesh; animation comes later from the
    # retarget, which builds its own clip against the corrected rest pose.
    dropped = len(bpy.data.actions)
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)
    for o in rig_objs:
        if o.name in bpy.data.objects and o.animation_data:
            o.animation_data_clear()
    if dropped:
        print("dropped %d imported action(s); exporting the bind pose" % dropped)

    # Give the atlas a real file on disk before any FBX export.
    #
    # The image arrived inside a GLB, so it lives only in memory with an empty
    # filepath. Blender still embeds its bytes, but it writes a blank
    # RelativeFilename alongside them -- and Unity uses that name to bind the
    # embedded media to a material slot. The result is an FBX that genuinely
    # contains a 2048px PNG and still imports untextured. Saving it first gives
    # the exporter a name to write.
    out_dir = os.path.dirname(os.path.abspath(out_glb)) or "."
    os.makedirs(out_dir, exist_ok=True)
    atlas_png = os.path.join(out_dir, os.path.splitext(os.path.basename(out_glb))[0] + "_atlas.png")
    try:
        image.filepath_raw = atlas_png
        image.file_format = "PNG"
        image.save()
        image.filepath = atlas_png
        image.source = "FILE"
        image.reload()
        print("wrote atlas %s (%dx%d)" % (os.path.basename(atlas_png), *image.size))
    except Exception as exc:
        print("WARNING: could not write the atlas to disk (%s); the FBX may "
              "import without textures" % exc)

    bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB",
                              use_selection=True, export_skins=True,
                              export_animations=False, export_yup=True)
    print("wrote " + out_glb + " (%.1f MB)" % (os.path.getsize(out_glb) / 1e6))

    if out_fbx:
        bpy.ops.export_scene.fbx(filepath=out_fbx, use_selection=True,
                                 axis_forward="-Z", axis_up="Y",
                                 path_mode="COPY", embed_textures=True,
                                 add_leaf_bones=False, bake_anim=False)
        print("wrote " + out_fbx + " (%.1f MB)" % (os.path.getsize(out_fbx) / 1e6))

    # A zip of the FBX beside its .fbm texture folder.
    #
    # Embedded textures depend on the importer choosing to extract them, and
    # that is out of our hands. Loose textures in the .fbm folder next to the
    # FBX is the oldest and most widely understood convention there is: Unity
    # resolves them on import with nothing to click. Shipping both means the
    # single file is there when it works and the zip is there when it does not.
    out_zip = arg("--zip", None)
    if out_zip:
        import shutil
        import tempfile
        import zipfile

        stem = os.path.splitext(os.path.basename(out_zip))[0]
        tmp = tempfile.mkdtemp(prefix="fbxzip_")
        try:
            fbx_path = os.path.join(tmp, stem + ".fbx")
            # embed_textures=False with COPY writes the textures into
            # "<stem>.fbm/" and references them relatively, which is what an
            # importer follows without being asked.
            bpy.ops.export_scene.fbx(filepath=fbx_path, use_selection=True,
                                     axis_forward="-Z", axis_up="Y",
                                     path_mode="COPY", embed_textures=False,
                                     add_leaf_bones=False, bake_anim=False)
            os.makedirs(os.path.dirname(os.path.abspath(out_zip)) or ".", exist_ok=True)
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _dirs, files in os.walk(tmp):
                    for name in files:
                        full = os.path.join(root, name)
                        z.write(full, os.path.relpath(full, tmp))
            listing = []
            with zipfile.ZipFile(out_zip) as z:
                listing = z.namelist()
            print("wrote %s (%.1f MB): %s"
                  % (out_zip, os.path.getsize(out_zip) / 1e6, ", ".join(listing)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
