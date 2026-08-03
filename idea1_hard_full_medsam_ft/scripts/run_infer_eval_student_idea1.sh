#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <dataset> <idea_fold_root> <swin_repo> <out_root> [gpu]" >&2
  exit 2
fi

DATASET="$1"
FOLD_ROOT="$2"
SWIN_REPO="$3"
OUT_ROOT="$4"
GPU="${5:-0}"
METHOD="idea1_hard_full_medsam_ft"
RUN_ROOT="$OUT_ROOT/$METHOD/$DATASET/fold_0"
CKPT="$RUN_ROOT/last.pth"
PRED="$RUN_ROOT/pred_test"

rm -rf "$PRED"
mkdir -p "$PRED"
CUDA_VISIBLE_DEVICES="$GPU" python -u "$SWIN_REPO/pipeline/infer_student.py" \
  --fold_root "$FOLD_ROOT" \
  --dataset "$DATASET" \
  --mode baseline \
  --deep_supervision \
  --ckpt "$CKPT" \
  --out_dir "$PRED" \
  --amp \
  2>&1 | tee "$RUN_ROOT/run_infer_idea1.log"

case "$DATASET" in
  btcv|synapse|acdc|prostate158)
    SAVE_DIR="$RUN_ROOT/eval_3d"
    rm -rf "$SAVE_DIR"
    python -u "$SWIN_REPO/pipeline/eval_3d.py" \
      --fold_root "$FOLD_ROOT" \
      --pred_dir "$PRED" \
      --save_dir "$SAVE_DIR" \
      --split test \
      --require_native_gt \
      2>&1 | tee "$RUN_ROOT/run_eval_idea1.log"
    ;;
  *)
    SAVE_DIR="$RUN_ROOT/eval_2d"
    rm -rf "$SAVE_DIR"
    EXTRA=()
    if [[ "$DATASET" == "kvasirseg" || "$DATASET" == "cvc_clinicdb" ]]; then
      EXTRA+=(--report_mae)
    fi
    python -u "$SWIN_REPO/pipeline/eval_2d.py" \
      --fold_root "$FOLD_ROOT" \
      --pred_dir "$PRED" \
      --save_dir "$SAVE_DIR" \
      --split test \
      --require_native_gt \
      "${EXTRA[@]}" \
      2>&1 | tee "$RUN_ROOT/run_eval_idea1.log"
    ;;
esac
