#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT="${1:-/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$FORMAL_ROOT/logs/orchestrator"
mkdir -p "$LOG_ROOT"

# Wait until no compute process is using a GPU. Xorg-only devices are accepted.
wait_gpu() {
  local gpu="$1"
  while true; do
    local count
    count="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "$count" -eq 0 ]]; then
      return 0
    fi
    echo "[WAIT] GPU $gpu has $count compute process(es)"
    sleep 60
  done
}

run_dataset() {
  local gpu="$1"; shift
  local ds="$1"
  wait_gpu "$gpu"
  echo "[QUEUE] GPU=$gpu dataset=$ds"
  bash "$HERE/run_dataset_full.sh" "$ds" "$gpu" "$FORMAL_ROOT"
}

run_h50() {
  local gpu="$1"
  wait_gpu "$gpu"
  echo "[QUEUE] GPU=$gpu task=tg3k_h50"
  bash "$HERE/run_tg3k_h50_ablation.sh" "$gpu" "$FORMAL_ROOT"
}

# Balanced queues. TG3K main method is already complete and is not repeated.
(
  run_dataset 0 tn3k
  run_dataset 0 btcv
) > "$LOG_ROOT/gpu0.log" 2>&1 & P0=$!

(
  run_h50 1
  run_dataset 1 otu_2d
  run_dataset 1 synapse
) > "$LOG_ROOT/gpu1.log" 2>&1 & P1=$!

(
  run_dataset 2 kvasirseg
  run_dataset 2 ph2
  run_dataset 2 acdc
) > "$LOG_ROOT/gpu2.log" 2>&1 & P2=$!

(
  run_dataset 3 cvc_clinicdb
  run_dataset 3 ddti
  run_dataset 3 prostate158
) > "$LOG_ROOT/gpu3.log" 2>&1 & P3=$!

printf 'GPU0=%s\nGPU1=%s\nGPU2=%s\nGPU3=%s\n' "$P0" "$P1" "$P2" "$P3" \
  | tee "$LOG_ROOT/queue_pids.txt"

status=0
for pair in "0:$P0" "1:$P1" "2:$P2" "3:$P3"; do
  gpu="${pair%%:*}"; pid="${pair##*:}"
  if wait "$pid"; then
    echo "[PASS] GPU $gpu queue finished"
  else
    echo "[FAIL] GPU $gpu queue failed"
    status=1
  fi
done
exit "$status"
