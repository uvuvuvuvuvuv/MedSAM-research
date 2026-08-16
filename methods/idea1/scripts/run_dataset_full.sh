#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dataset> <gpu> [formal_root]" >&2
  exit 2
fi

DATASET="$1"
GPU="$2"
FORMAL_ROOT="${3:-/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft}"

METHOD="idea1_hard_full_medsam_ft"
MEDSAM_REPO="${MEDSAM_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/MedSAM-main}"
IDEA_CODE="$MEDSAM_REPO/methods/idea1"
SWIN_REPO="${SWIN_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/Swin-UMamba-main}"
FROZEN_PROCESSED="${FROZEN_PROCESSED:-/storage/baiyuting/data/MedSAM-main/data/processed}"
BASE_CKPT="${BASE_CKPT:-/storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth}"
AUDIT="${AUDIT:-/storage/baiyuting/data/out_data_idea1/baseline_reference/baseline_contract_audit.json}"
IDEA_PROCESSED="$FORMAL_ROOT/MedSAM-main/data/processed"
VIEW_ROOT="$FORMAL_ROOT/student_views"
WORK_ROOT="$FORMAL_ROOT/Swin-UMamba-main/work_dir"
FOLD_ROOT="$IDEA_PROCESSED/$DATASET/fold_0"
STUDENT_VIEW="$VIEW_ROOT/$METHOD/$DATASET/fold_0"
OUT_DIR="$WORK_ROOT/$METHOD/$DATASET/fold_0"
LOG_DIR="$FORMAL_ROOT/logs/$DATASET"
LOCK_DIR="$FORMAL_ROOT/locks"
mkdir -p "$LOG_DIR" "$LOCK_DIR" "$OUT_DIR"

exec 9>"$LOCK_DIR/${DATASET}.lock"
if ! flock -n 9; then
  echo "[STOP] another process holds dataset lock: $DATASET" >&2
  exit 9
fi

case "$DATASET" in
  btcv|synapse) BS=1; NW=0; EVAL_KIND=3d ;;
  acdc|prostate158) BS=2; NW=2; EVAL_KIND=3d ;;
  kvasirseg|cvc_clinicdb|tn3k|tg3k|ddti|otu_2d|ph2) BS=8; NW=4; EVAL_KIND=2d ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

for P in "$MEDSAM_REPO" "$SWIN_REPO" "$FROZEN_PROCESSED" "$BASE_CKPT" "$AUDIT"; do
  [[ -e "$P" ]] || { echo "[MISSING] $P" >&2; exit 3; }
done

export PYTHONUNBUFFERED=1

echo "[STAGE 1] MedSAM iteration + final labels: $DATASET"
CUDA_VISIBLE_DEVICES="$GPU" conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$IDEA_CODE:$MEDSAM_REPO:${PYTHONPATH:-}" \
  python -u "$IDEA_CODE/run_dataset_full.py" \
    --repo_root "$MEDSAM_REPO" \
    --frozen_processed_root "$FROZEN_PROCESSED" \
    --idea_processed_root "$IDEA_PROCESSED" \
    --view_root "$VIEW_ROOT" \
    --dataset "$DATASET" \
    --fold fold_0 \
    --method "$METHOD" \
    --base_checkpoint "$BASE_CKPT" \
    --baseline_audit "$AUDIT" \
    --device cuda \
    --seed 2026 \
    --steps_2d 300 \
    --steps_3d 1000 \
    --batch_size 2 \
  2>&1 | tee "$LOG_DIR/01_medsam_and_labels.log"

echo "[STAGE 2] Student training: $DATASET"
LAST="$OUT_DIR/last.pth"
LATEST="$OUT_DIR/checkpoint_latest.pth"
TRAIN_DONE=0
if [[ -f "$OUT_DIR/train_log.csv" ]]; then
  LAST_EPOCH="$(tail -1 "$OUT_DIR/train_log.csv" | cut -d, -f1 || true)"
  [[ "$LAST_EPOCH" == "50" ]] && TRAIN_DONE=1
fi
if [[ "$TRAIN_DONE" -eq 0 ]]; then
  rm -f "$OUT_DIR/epoch_15.pth" "$OUT_DIR/epoch_20.pth" "$OUT_DIR/epoch_30.pth" "$OUT_DIR/epoch_40.pth" "$OUT_DIR/epoch_50.pth"
  (
    CUDA_VISIBLE_DEVICES="$GPU" conda run -n swin_umamba --no-capture-output \
      env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
      python -u "$SWIN_REPO/pipeline/train_student.py" \
        --fold_root "$STUDENT_VIEW" \
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
  ) &
  TRAIN_PID=$!

  (
    targets=(15 20 30 40 50)
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      if [[ -f "$OUT_DIR/train_log.csv" && -f "$LATEST" ]]; then
        ep="$(tail -1 "$OUT_DIR/train_log.csv" | cut -d, -f1 || true)"
        for t in "${targets[@]}"; do
          if [[ "$ep" =~ ^[0-9]+$ ]] && (( ep >= t )) && [[ ! -e "$OUT_DIR/epoch_${t}.pth" ]]; then
            cp --reflink=auto "$LATEST" "$OUT_DIR/epoch_${t}.pth"
            echo "[CHECKPOINT] saved epoch_${t}.pth"
          fi
        done
      fi
      sleep 5
    done
  ) &
  WATCH_PID=$!

  wait "$TRAIN_PID"
  wait "$WATCH_PID" || true
else
  echo "[SKIP] student train_log already ends at epoch 50"
fi

if [[ ! -f "$LAST" && -f "$LATEST" ]]; then
  ln -sfn checkpoint_latest.pth "$LAST"
fi
CKPT="$LAST"
[[ -f "$CKPT" ]] || CKPT="$LATEST"
[[ -f "$CKPT" ]] || { echo "[MISSING] student checkpoint" >&2; exit 4; }

echo "[STAGE 3] Inference + native evaluation: $DATASET"
SUMMARY="$OUT_DIR/eval_2d/eval_summary.json"
[[ "$EVAL_KIND" == "3d" ]] && SUMMARY="$OUT_DIR/eval_3d/eval_3d_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  rm -rf "$OUT_DIR/pred_test" "$OUT_DIR/eval_2d" "$OUT_DIR/eval_3d"
  mkdir -p "$OUT_DIR/pred_test" "$SWIN_REPO/work_dir/baseline/$DATASET/fold_0"

  CUDA_VISIBLE_DEVICES="$GPU" conda run -n swin_umamba --no-capture-output \
    env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
    python -u "$SWIN_REPO/pipeline/infer_student.py" \
      --fold_root "$STUDENT_VIEW" \
      --dataset "$DATASET" \
      --mode baseline \
      --deep_supervision \
      --ckpt "$CKPT" \
      --out_dir "$OUT_DIR/pred_test" \
      --amp \
    2>&1 | tee "$OUT_DIR/run_infer_idea1.log"

  if [[ "$EVAL_KIND" == "3d" ]]; then
    conda run -n swin_umamba --no-capture-output \
      env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
      python -u "$SWIN_REPO/pipeline/eval_3d.py" \
        --fold_root "$STUDENT_VIEW" \
        --pred_dir "$OUT_DIR/pred_test" \
        --save_dir "$OUT_DIR/eval_3d" \
        --split test \
        --require_native_gt \
      2>&1 | tee "$OUT_DIR/run_eval_idea1.log"
  else
    EXTRA=()
    [[ "$DATASET" == "kvasirseg" || "$DATASET" == "cvc_clinicdb" ]] && EXTRA+=(--report_mae)
    conda run -n swin_umamba --no-capture-output \
      env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
      python -u "$SWIN_REPO/pipeline/eval_2d.py" \
        --fold_root "$STUDENT_VIEW" \
        --pred_dir "$OUT_DIR/pred_test" \
        --save_dir "$OUT_DIR/eval_2d" \
        --split test \
        --require_native_gt \
        "${EXTRA[@]}" \
      2>&1 | tee "$OUT_DIR/run_eval_idea1.log"
  fi
else
  echo "[SKIP] evaluation summary exists: $SUMMARY"
fi

[[ -f "$SUMMARY" ]] || { echo "[MISSING] final evaluation summary: $SUMMARY" >&2; exit 5; }
python - "$DATASET" "$SUMMARY" "$OUT_DIR/DONE.json" <<'PY'
import json, sys, time
from pathlib import Path
dataset, summary, out = sys.argv[1:]
obj = {"dataset": dataset, "summary": summary, "completed_at_unix": time.time(), "status": "DONE"}
Path(out).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
echo "[DONE] $DATASET"
