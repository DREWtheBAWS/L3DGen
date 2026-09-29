"""
Render frames of an animated GLB, to see whether a retarget actually worked.

    "<blender.exe>" --background --python scripts/render-anim.py -- \
        <animated.glb> <out_dir> [--frames 1,30,60,90] [--res 480] [--yaw 0]

Numbers cannot tell you a rig is bound correctly -- a skeleton can move exactly
as intended while the skin folds through itself. Rendering a few frames is the
only check that answers the question actually being asked.
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main():
    args = argv_after_dashes()
    if len(args) < 2:
        print("usage: <animated.glb> <out_dir> [--frames a,b,c] [--res N] [--yaw D]")
        return 1
    src, out_dir = args[0], Path(args[1])
    arg = lambda f, d: (args[args.index(f) + 1] if f in args else d)  # noqa: E731
    res = int(arg("--res", "480"))
    yaw = float(arg("--yaw", "0"))
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)

    scene = bpy.context.scene
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        print("ERROR: no mesh in " + src)
        return 1

    acts = list(bpy.data.actions)
    print("actions: " + (", ".join(a.name for a in acts) or "none"))
    if acts:
        f0, f1 = (int(round(x)) for x in acts[0].frame_range)
    else:
        f0, f1 = 1, 1
    frames = ([int(x) for x in arg("--frames", "").split(",") if x]
              or [f0, f0 + (f1 - f0) // 3, f0 + 2 * (f1 - f0) // 3, f1])
    print("frame range %d..%d, rendering %s" % (f0, f1, frames))

    # Frame the camera on the character's bounds in world space.
    pts = []
    for m in meshes:
        for c in m.bound_box:
            pts.append(m.matrix_world @ Vector(c))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    ctr = (lo + hi) / 2.0
    size = max((hi - lo).x, (hi - lo).y, (hi - lo).z)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = size * 1.35
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    a = math.radians(yaw)
    cam.location = ctr + Vector((math.sin(a), -math.cos(a), 0.0)) * size * 3.0
    cam.rotation_euler = (math.radians(90), 0.0, a)
    scene.camera = cam

    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(35 + yaw))
    scene.collection.objects.link(sun)

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("w")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.12, .14, .18, 1)

    for f in frames:
        scene.frame_set(f)
        scene.render.filepath = str(out_dir / ("frame_%04d.png" % f))
        bpy.ops.render.render(write_still=True)
        print("  wrote frame %d" % f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
