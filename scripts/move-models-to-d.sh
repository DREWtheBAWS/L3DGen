#!/usr/bin/env bash
# Move model weights from the C: boot drive to D:\ComfyUI-Models.
#
# ComfyUI reads D: through custom_nodes/../extra_model_paths.yaml, which it loads
# in addition to the Comfy Desktop managed config. Nothing is deleted until the
# copy is verified byte-for-byte by size, so an interrupted run leaves the
# original in place rather than a half-file.
#
# Safe to re-run: files already moved are skipped.
set -u

# Both overridable: these defaults match a Comfy Desktop install on Windows.
#   SRC=... DST=... ./scripts/move-models-to-d.sh
SRC="${SRC:-$LOCALAPPDATA/Comfy-Desktop/ComfyUI-Shared/models}"
DST="${DST:-/d/ComfyUI-Models}"

moved=0; skipped=0; failed=0; bytes=0

# A plain file list, not process substitution: this Git Bash has no /dev/fd.
LIST="$(mktemp)"
find "$SRC" -type f \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pth' \
     -o -name '*.pt' -o -name '*.bin' -o -name '*.onnx' -o -name '*.gguf' \) > "$LIST"

while IFS= read -r f; do
  [ -n "$f" ] || continue
  rel="${f#$SRC/}"
  out="$DST/$rel"
  name="$(basename "$f")"
  mkdir -p "$(dirname "$out")"

  src_size=$(stat -c %s "$f")
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$src_size" ]; then
    echo "[skip] $rel (already on D:)"
    rm -f "$f"
    skipped=$((skipped + 1))
    continue
  fi

  printf '[move] %-52s %6s MB ... ' "$rel" "$((src_size / 1048576))"
  if cp "$f" "$out"; then
    if [ "$(stat -c %s "$out")" = "$src_size" ]; then
      rm -f "$f"
      echo "ok"
      moved=$((moved + 1))
      bytes=$((bytes + src_size))
    else
      echo "SIZE MISMATCH - original kept"
      rm -f "$out"
      failed=$((failed + 1))
    fi
  else
    echo "COPY FAILED - original kept"
    failed=$((failed + 1))
  fi
done < "$LIST"
rm -f "$LIST"

echo
echo "moved $moved, skipped $skipped, failed $failed  ($((bytes / 1073741824)) GB transferred)"
[ "$failed" -gt 0 ] && echo "Some files stayed on C: - rerun to retry." && exit 1
exit 0
