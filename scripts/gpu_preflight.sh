#!/usr/bin/env bash

# Shared-server guard for one baseline process. It never kills or resets a GPU.
# Use `--run COMMAND ...` to select a currently roomy GPU and launch exactly one
# command with that physical GPU made visible as CUDA device 0.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate a virtual environment before running this script." >&2
  exit 2
fi

minimum_free_mib="${QAQ_MIN_FREE_MIB:-20000}"
mode="report"
if [[ "${1:-}" == "--run" ]]; then
  mode="run"
  shift
fi
if [[ "${1:-}" == "--index-only" ]]; then
  mode="index-only"
  shift
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available." >&2
  exit 2
fi

query="nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits"
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
if (( ${#gpu_rows[@]} == 0 )); then
  echo "No NVIDIA GPUs were returned." >&2
  exit 2
fi

best_index=""
best_free="-1"
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r index name total used free util <<< "$row"
  index="${index// /}"
  free="${free// /}"
  if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= minimum_free_mib )) && (( free > best_free )); then
    best_index="$index"
    best_free="$free"
  fi
done

if [[ -z "$best_index" ]]; then
  echo "No GPU has at least ${minimum_free_mib} MiB free." >&2
  printf '%s\n' "${gpu_rows[@]}" >&2
  exit 3
fi

# Print current state for a human audit. The process list is informational only.
if [[ "$mode" != "index-only" ]]; then
  echo "Current GPU state:"
  printf '%s\n' "${gpu_rows[@]}"
  echo
  echo "Current compute processes (do not interrupt other users):"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
  echo
fi

# Recheck the selected physical device immediately before a possible launch.
selected_free="$(nvidia-smi --id="$best_index" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ' | head -n 1)"
if [[ ! "$selected_free" =~ ^[0-9]+$ ]] || (( selected_free < minimum_free_mib )); then
  echo "Selected GPU ${best_index} no longer has ${minimum_free_mib} MiB free; stop and recheck." >&2
  exit 4
fi

if [[ "$mode" == "index-only" ]]; then
  printf '%s\n' "$best_index"
  exit 0
fi

echo "Selected physical GPU ${best_index} with ${selected_free} MiB free."
echo "Inside the launched process it will appear as CUDA device 0."

if [[ "$mode" == "run" ]]; then
  if (( $# == 0 )); then
    echo "--run requires a command." >&2
    exit 2
  fi
  exec env CUDA_VISIBLE_DEVICES="$best_index" "$@"
fi

echo "To launch one command after reviewing this state:"
printf '  env CUDA_VISIBLE_DEVICES=%q <command>\n' "$best_index"
