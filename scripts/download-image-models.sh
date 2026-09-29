#!/usr/bin/env bash
# Models for the view-synthesis stage. Safe to re-run: resumes, skips complete.
set -u
# Models live on D: (see ComfyUI/extra_model_paths.yaml, which adds this root).
M="/d/ComfyUI-Models"

get () {
  local dir="$M/$1" url="$2" name; name="$(basename "$url")"
  mkdir -p "$dir"; local dest="$dir/$name" want
  want=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
  if [ -f "$dest" ] && [ -n "${want:-}" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then
    echo "[skip] $name"; return 0; fi
  echo "[get ] $name -> $1/"
  curl -sL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
  local have; have=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  [ -n "${want:-}" ] && [ "$have" != "$want" ] && { echo "[FAIL] $name got=$have want=$want"; return 1; }
  echo "[ok  ] $name"
}

# Flux Kontext dev: reference-image conditioned editing, core ComfyUI nodes.
get diffusion_models https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors
get text_encoders    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors
get text_encoders    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
get vae              https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors

# Stylized SDXL finetune: lighter/faster alternative to Kontext.
get checkpoints      https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_v2_1.safetensors
echo "=== ALL DONE ==="
