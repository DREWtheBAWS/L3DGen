"""
Retarget a Mixamo animation onto a rigged GLB from this pipeline.

    "<blender.exe>" --background --python scripts/retarget-mixamo.py -- \
        <rigged.glb> <mixamo.fbx> <out.glb> [--fps 30]

Why a retarget and not a straight copy
-------------------------------------
Both rigs use the same `mixamorig:*` names, but a Mixamo clip stores each bone's
rotation *relative to Mixamo's own rest pose*, and this pipeline's rest pose is
whatever the generated mesh happened to be. Copying local rotations across would
apply Mixamo's deltas to differently-oriented bones and fold the character up.

So the transfer goes through world space: take the source bone's rotation
relative to its own rest, and apply that same world-space delta to the target
bone's rest. Bones are walked parents-first so each one is posed on top of an
already-posed parent.
"""

import sys
from pathlib import Path

import bpy
from mathutils import Matrix

MIXAMO_PREFIX = "mixamorig:"


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_armature(objs):
    for o in objs:
        if o.type == "ARMATURE":
            return o
    return None


def bone_order(arm):
    """Bone names, parents before children."""
    out = []

    def walk(b):
        out.append(b.name)
        for c in b.children:
            walk(c)

    for b in arm.data.bones:
        if b.parent is None:
            walk(b)
    return out


def normalise(name):
    """Match names across rigs that may or may not carry the Mixamo prefix."""
    n = name
    if n.startswith(MIXAMO_PREFIX):
        n = n[len(MIXAMO_PREFIX):]
    return n.lower().replace(" ", "").replace("_", "")


def main():
    args = argv_after_dashes()
    if len(args) < 3:
        print("usage: <rigged.glb> <mixamo.fbx> <out.glb> [--fps N]")
        return 1
    glb_path, fbx_path, out_path = args[0], args[1], args[2]
    fps = int(args[args.index("--fps") + 1]) if "--fps" in args else 30

    clear_scene()
    scene = bpy.context.scene
    scene.render.fps = fps

    # -- target: our rigged GLB -------------------------------------------------
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    dst_objs = [o for o in bpy.data.objects if o not in before]
    dst = find_armature(dst_objs)
    if dst is None:
        print("ERROR: no armature in " + glb_path)
        return 1
    dst_meshes = [o for o in dst_objs if o.type == "MESH"]
    print("target armature '%s': %d bones, %d mesh(es)"
          % (dst.name, len(dst.data.bones), len(dst_meshes)))

    # -- source: the Mixamo FBX -------------------------------------------------
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=fbx_path, automatic_bone_orientation=True)
    src_objs = [o for o in bpy.data.objects if o not in before]
    src = find_armature(src_objs)
    if src is None:
        print("ERROR: no armature in " + fbx_path)
        return 1
    # The Mixamo mesh itself is not wanted, only its motion.
    for o in src_objs:
        if o.type == "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    print("source armature '%s': %d bones" % (src.name, len(src.data.bones)))

    action = src.animation_data.action if src.animation_data else None
    if action is None:
        print("ERROR: the FBX carries no animation")
        return 1
    f0, f1 = (int(round(x)) for x in action.frame_range)
    print("clip '%s': frames %d..%d" % (action.name, f0, f1))

    # -- pair the bones ---------------------------------------------------------
    src_by_key = {normalise(b.name): b.name for b in src.data.bones}
    pairs = []
    for name in bone_order(dst):
        key = normalise(name)
        if key in src_by_key:
            pairs.append((name, src_by_key[key]))
    print("matched %d of %d target bones" % (len(pairs), len(dst.data.bones)))
    if not pairs:
        print("ERROR: no bone names in common")
        return 1
    missing = [n for n in bone_order(dst) if normalise(n) not in src_by_key]
    if missing:
        print("  unmatched (left at rest): " + ", ".join(missing))

    # Scale the root's translation between rigs, so a character half the height
    # of the Mixamo actor does not stride twice as far as its own legs allow.
    def height_of(arm):
        zs = [(arm.matrix_world @ b.head_local).z for b in arm.data.bones]
        zs += [(arm.matrix_world @ b.tail_local).z for b in arm.data.bones]
        return max(zs) - min(zs)

    scale = height_of(dst) / max(height_of(src), 1e-6)
    print("height ratio target/source = %.3f" % scale)

    root_name = next((n for n, _ in pairs if normalise(n) == "hips"), pairs[0][0])

    # Rest matrices, captured once.
    src_rest = {sn: (src.matrix_world @ src.pose.bones[sn].bone.matrix_local)
                for _, sn in pairs}
    dst_rest = {dn: (dst.matrix_world @ dst.pose.bones[dn].bone.matrix_local)
                for dn, _ in pairs}

    dst.animation_data_create()
    dst_action = bpy.data.actions.new("Retargeted")
    dst.animation_data.action = dst_action
    for pb in dst.pose.bones:
        pb.rotation_mode = "QUATERNION"

    scene.frame_start, scene.frame_end = f0, f1

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        for dn, sn in pairs:
            sb = src.pose.bones[sn]
            db = dst.pose.bones[dn]

            src_pose = (src.matrix_world @ sb.matrix).to_3x3().normalized()
            delta = src_pose @ src_rest[sn].to_3x3().normalized().inverted()
            new_rot = delta @ dst_rest[dn].to_3x3().normalized()

            # db.matrix already reflects the parent posed this same frame, so its
            # translation is the correct place for this bone to sit; only the
            # orientation is being replaced.
            loc = db.matrix.to_translation()
            if dn == root_name:
                src_off = (src.matrix_world @ sb.matrix).to_translation() \
                    - src_rest[sn].to_translation()
                loc = dst_rest[dn].to_translation() + src_off * scale

            db.matrix = Matrix.Translation(loc) @ new_rot.to_4x4()
            bpy.context.view_layer.update()

        for dn, _ in pairs:
            db = dst.pose.bones[dn]
            db.keyframe_insert("rotation_quaternion", frame=f)
            if dn == root_name:
                db.keyframe_insert("location", frame=f)

    print("baked %d frames onto %d bones" % (f1 - f0 + 1, len(pairs)))

    # Drop the source rig so only the animated character is exported.
    bpy.data.objects.remove(src, do_unlink=True)

    # ...and its action with it. The glTF exporter emits every action that could
    # apply to the armature, so leaving the imported Mixamo one behind ships two
    # clips and the player has no way to know which is the retargeted one.
    for a in list(bpy.data.actions):
        if a is not dst_action:
            bpy.data.actions.remove(a)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in dst_objs:
        if o.name in bpy.data.objects:
            o.select_set(True)
    bpy.context.view_layer.objects.active = dst

    bpy.ops.export_scene.gltf(
        filepath=out_path, export_format="GLB", use_selection=True,
        export_animations=True, export_frame_range=True,
        export_skins=True, export_yup=True,
    )
    print("wrote " + out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
