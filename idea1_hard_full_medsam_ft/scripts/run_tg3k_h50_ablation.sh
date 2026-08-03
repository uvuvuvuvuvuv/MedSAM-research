#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
FORMAL_ROOT="${2:-/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft}"
MEDSAM_REPO="${MEDSAM_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/MedSAM-main}"
IDEA_CODE="$MEDSAM_REPO/idea1_hard_full_medsam_ft"
SWIN_REPO="${SWIN_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/Swin-UMamba-main}"
FOLD_ROOT="$FORMAL_ROOT/MedSAM-main/data/processed/tg3k/fold_0"
VIEW_ROOT="$FORMAL_ROOT/student_views"
H_VIEW="$VIEW_ROOT/hard20_fullonly/tg3k/fold_0"
OUT="$FORMAL_ROOT/Swin-UMamba-main/work_dir/hard20_fullonly/tg3k/fold_0"
A_OUT="$FORMAL_ROOT/Swin-UMamba-main/work_dir/idea1_hard_full_medsam_ft/tg3k/fold_0"
SELECTION="$FOLD_ROOT/rounds/idea1_hard_full_medsam_ft/round_03/selection/selection.json"
SUMMARY_DIR="$FORMAL_ROOT/summaries/tg3k/ablation_h50"
mkdir -p "$OUT" "$SUMMARY_DIR" "$FORMAL_ROOT/locks"

exec 9>"$FORMAL_ROOT/locks/tg3k_h50.lock"
flock -n 9 || { echo "[STOP] tg3k_h50 already running"; exit 9; }

conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$IDEA_CODE:$MEDSAM_REPO:${PYTHONPATH:-}" \
  python -u "$IDEA_CODE/ablation/build_hard20_fullonly.py" \
    --fold_root "$FOLD_ROOT" \
    --selection "$SELECTION" \
    --overwrite

conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$IDEA_CODE:$MEDSAM_REPO:${PYTHONPATH:-}" \
  python -u "$IDEA_CODE/09_build_student_view.py" \
    --fold_root "$FOLD_ROOT" \
    --view_root "$VIEW_ROOT" \
    --dataset tg3k \
    --fold fold_0 \
    --method hard20_fullonly \
    --overwrite

TRAIN_DONE=0
if [[ -f "$OUT/train_log.csv" ]] && [[ "$(tail -1 "$OUT/train_log.csv" | cut -d, -f1)" == "50" ]]; then
  TRAIN_DONE=1
fi
if [[ "$TRAIN_DONE" -eq 0 ]]; then
  (
    CUDA_VISIBLE_DEVICES="$GPU" conda run -n swin_umamba --no-capture-output \
      env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
      python -u "$SWIN_REPO/pipeline/train_student.py" \
        --fold_root "$H_VIEW" \
        --dataset tg3k \
        --mode baseline \
        --epochs 50 \
        --batch_size 8 \
        --num_workers 4 \
        --lr 1e-4 \
        --weight_decay 0.05 \
        --freeze_encoder_epochs 10 \
        --amp \
        --deep_supervision \
        --pretrained_ckpt "$SWIN_REPO/data/pretrained/vmamba/vmamba_tiny_e292.pth" \
        --out_dir "$OUT" \
      2>&1 | tee "$OUT/run_train_h50.log"
  ) &
  PID=$!
  (
    for_target=(15 20 30 40 50)
    while kill -0 "$PID" 2>/dev/null; do
      if [[ -f "$OUT/train_log.csv" && -f "$OUT/checkpoint_latest.pth" ]]; then
        ep="$(tail -1 "$OUT/train_log.csv" | cut -d, -f1 || true)"
        for t in "${for_target[@]}"; do
          if [[ "$ep" =~ ^[0-9]+$ ]] && (( ep >= t )) && [[ ! -e "$OUT/epoch_${t}.pth" ]]; then
            cp --reflink=auto "$OUT/checkpoint_latest.pth" "$OUT/epoch_${t}.pth"
          fi
        done
      fi
      sleep 5
    done
  ) &
  WPID=$!
  wait "$PID"
  wait "$WPID" || true
fi

[[ -f "$OUT/last.pth" ]] || ln -sfn checkpoint_latest.pth "$OUT/last.pth"
rm -rf "$OUT/pred_test" "$OUT/eval_2d"
mkdir -p "$OUT/pred_test" "$SWIN_REPO/work_dir/baseline/tg3k/fold_0"

CUDA_VISIBLE_DEVICES="$GPU" conda run -n swin_umamba --no-capture-output \
  env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
  python -u "$SWIN_REPO/pipeline/infer_student.py" \
    --fold_root "$H_VIEW" \
    --dataset tg3k \
    --mode baseline \
    --deep_supervision \
    --ckpt "$OUT/last.pth" \
    --out_dir "$OUT/pred_test" \
    --amp \
  2>&1 | tee "$OUT/run_infer_h50.log"

conda run -n swin_umamba --no-capture-output \
  env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
  python -u "$SWIN_REPO/pipeline/eval_2d.py" \
    --fold_root "$H_VIEW" \
    --pred_dir "$OUT/pred_test" \
    --save_dir "$OUT/eval_2d" \
    --split test \
    --require_native_gt \
  2>&1 | tee "$OUT/run_eval_h50.log"

conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$IDEA_CODE:$MEDSAM_REPO:${PYTHONPATH:-}" \
  python -u "$IDEA_CODE/ablation/compare_tg3k_h50.py" \
    --a_summary "$A_OUT/eval_2d/eval_summary.json" \
    --h_summary "$OUT/eval_2d/eval_summary.json" \
    --output_dir "$SUMMARY_DIR"

echo "[DONE] TG3K H50 ablation"
