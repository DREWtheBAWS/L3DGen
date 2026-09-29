# animations/

Drop Mixamo `.fbx` downloads here and the panel lists them automatically under
**Mixamo clip**, ready to retarget onto a rigged model.

The files themselves are not in this repository. Adobe's Mixamo licence permits
using their animations in your own projects but not redistributing the asset
files, so `.gitignore` keeps them out.

## Getting a clip

1. Sign in at [mixamo.com](https://www.mixamo.com).
2. Pick any animation.
3. Download with **Format: FBX Binary**, **Skin: Without Skin**, and no
   character — the clip alone is all that is needed.
4. Save it here. The panel strips Mixamo's naming, so
   `X Bot@Standing Idle.fbx` is listed as **Standing Idle**.

Any FBX containing an armature animation works; it does not have to come from
Mixamo. Bones are matched by name, so the skeleton has to use the same naming as
the rig you are applying it to — `mixamorig:*` for anything this pipeline or
Mixamo's auto-rigger produced.
