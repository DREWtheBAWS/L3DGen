#!/usr/bin/env bash
# Downloads every model the image->3D pipeline needs into the ComfyUI shared models dir.
# Safe to re-run: curl -C - resumes, and finished files are skipped.
set -u
# Models live on D: (see ComfyUI/extra_model_paths.yaml, which adds this root).
M="/d/ComfyUI-Models"

get () { # get <subdir> <url>
  local dir="$M/$1" url="$2" name
  name="$(basename "$url")"
  mkdir -p "$dir"
  local dest="$dir/$name"
  local want
  want=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  if [ -f "$dest" ] && [ -n "${want:-}" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then
    echo "[skip] $name (complete)"; return 0
  fi
  echo "[get ] $name -> $1/"
  curl -L --retry 5 --retry-delay 5 -C - -o "$dest" "$url" 2>&1 | tail -2
  local have; have=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  if [ -n "${want:-}" ] && [ "$have" != "$want" ]; then
    echo "[FAIL] $name  got=$have want=$want"; return 1
  fi
  echo "[ok  ] $name"
}

get checkpoints        https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2-mv_fp16.safetensors
get diffusion_models   https://huggingface.co/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_int8_convrot.safetensors
get vae                https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors
get vae                https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors
get clip_vision        https://huggingface.co/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors
get background_removal https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors
get geometry_estimation https://huggingface.co/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors
echo "=== ALL DONE ==="

# --- optional: 2D model for the view-synthesis stage -------------------------
# SDXL base carries its own VAE and CLIP, so this single file is the whole stage.
# Swap in any stylized SDXL finetune by dropping it in checkpoints/ and picking
# it in the panel.
get checkpoints https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
