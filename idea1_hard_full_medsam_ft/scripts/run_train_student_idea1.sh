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

case "$DATASET" in
  btcv|synapse) BS=1; NW=0 ;;
  acdc|prostate158) BS=2; NW=2 ;;
  kvasirseg|cvc_clinicdb|tn3k|tg3k|ddti|otu_2d|ph2) BS=8; NW=4 ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

OUT_DIR="$OUT_ROOT/$METHOD/$DATASET/fold_0"
mkdir -p "$OUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python -u "$SWIN_REPO/pipeline/train_student.py" \
  --fold_root "$FOLD_ROOT" \
  --dataset "$DATASET" \
  --mode baseline \
  --epochs 50 \
  --batch_size "$BS" \
  --num_workers "$NW" \
  --lr 1e-4 \
  --weight_decay 0.05 \
  --freeze_encoder_epochs 10 \
  --amp \
  --deep_supervision \
  --pretrained_ckpt "$SWIN_REPO/data/pretrained/vmamba/vmamba_tiny_e292.pth" \
  --out_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run_train_idea1.log"
