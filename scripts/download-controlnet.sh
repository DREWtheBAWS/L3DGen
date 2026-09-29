#!/usr/bin/env bash
# Structural conditioning for the "paint the geometry" texture path.
#
# Why this is needed
# -----------------
# Filling unreached surface by blending or mirroring produces plausible *colour*,
# never *content*. When the shape model invents a satchel on a side no reference
# drew, that satchel needs leather, buckles and stitching painted onto it -- and
# nothing in the mesh's own colours can supply those.
#
# The image model has to invent them, which means running at a denoise high
# enough to add detail rather than merely recolour. At that strength an
# unconstrained img2img also wanders off the silhouette, and the result no longer
# reprojects. A depth/normal ControlNet is what pins the generation to the exact
# geometry while leaving it free to invent surface detail.
#
# Safe to re-run: resumes, skips complete files.
set -u
M="/d/ComfyUI-Models"

get () {
  local dir="$M/$1" url="$2" name; name="$(basename "$url")"
  mkdir -p "$dir"; local dest="$dir/$name" want
  want=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  if [ -f "$dest" ] && [ -n "${want:-}" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then
    echo "[skip] $name"; return 0; fi
  echo "[get ] $name -> $1/  (${want:-?} bytes)"
  curl -sL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
  local have; have=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  [ -n "${want:-}" ] && [ "$have" != "$want" ] && { echo "[FAIL] $name got=$have want=$want"; return 1; }
  echo "[ok  ] $name"
}

# ControlNet Union for SDXL: one model covering depth, normal, canny and more,
# selected at runtime with SetUnionControlNetType. Preferred over separate
# per-type models because the pipeline wants depth AND normal from the same
# render and a 10 GB card should load one adapter, not two.
get controlnet https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors

echo "=== ALL DONE ==="
echo "Restart ComfyUI so the new controlnet is listed, then pick it in the panel."
