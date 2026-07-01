#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Default SAC clean reproduction: full_pipeline_gpu2_2d scheduler
# - Does not touch frozen baseline pseudo/results.
# - Uses physical files only; no symlinks.
# - Waits for genuinely idle GPUs and starts one dataset pipeline per GPU.
# - Each dataset runs:
#   split copy -> 02 template -> 03 SAC train -> 04 pseudo -> 06 audit
#   -> Student dataset check -> Student train -> test infer -> test eval
# ============================================================

CONDA_BIN="/home/baiyuting/anaconda3/bin/conda"
MEDSAM_CODE="/storage/baiyuting/data/MedSAM-main/idea1_sac_medsam_final"
SWIN_ROOT="/storage/baiyuting/data/Swin-UMamba-main"
PROC="/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed"

METHOD_OLD="idea1_sac_medsam_final"
METHOD_REPRO="idea1_sac_medsam_final_repro"
PSEUDO_NAME="tri_train_${METHOD_REPRO}"

BASE_CKPT="/storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
PRETRAIN="${SWIN_ROOT}/data/pretrained/vmamba/vmamba_tiny_e292.pth"
MEDSAM_FT_ROOT="/storage/baiyuting/data/out_data_idea1/MedSAM-main/work_dir/MedSAM_ft/${METHOD_REPRO}"
STUDENT_ROOT="/storage/baiyuting/data/out_data_idea1/Swin-UMamba-main/work_dir/${METHOD_REPRO}"
VERIFY_ROOT="/storage/baiyuting/data/out_data_idea1/verification/${METHOD_REPRO}/full_pipeline_gpu2_2d"

# Some GPUs may already be occupied. The scheduler will not kill or pre-empt them.
GPU_POOL=(2)
MIN_FREE_MB=20000
MAX_UTIL=10
POLL_SECONDS=60

# CVC has already passed the complete clean reproduction.
# Synapse is intentionally excluded here because its current Student/eval task may still be running.
# Queue long 3D jobs first to reduce the overall makespan.
DEFAULT_DATASETS=(
  kvasirseg
  tn3k
  tg3k
  ddti
  otu_2d
  ph2
)

if (( $# > 0 )); then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

declare -A BATCH_SIZE=(
  [btcv]=1 [synapse]=1 [acdc]=2 [prostate158]=2
  [kvasirseg]=8 [cvc_clinicdb]=8 [tn3k]=8 [tg3k]=8
  [ddti]=8 [otu_2d]=8 [ph2]=8
)

declare -A NUM_WORKERS=(
  [btcv]=0 [synapse]=0 [acdc]=2 [prostate158]=2
  [kvasirseg]=4 [cvc_clinicdb]=4 [tn3k]=4 [tg3k]=4
  [ddti]=4 [otu_2d]=4 [ph2]=4
)

is_3d_dataset() {
  case "$1" in
    btcv|synapse|acdc|prostate158) return 0 ;;
    *) return 1 ;;
  esac
}

report_mae_dataset() {
  case "$1" in
    kvasirseg|cvc_clinicdb) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "[FATAL] missing file: $path" >&2; return 1; }
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || { echo "[FATAL] missing directory: $path" >&2; return 1; }
}

run_stage() {
  local stage="$1"
  local log="$2"
  shift 2

  echo "[$(date '+%F %T')] [START] ${stage}" | tee -a "${PIPELINE_LOG}"
  if "$@" >"${log}" 2>&1; then
    echo "[$(date '+%F %T')] [DONE ] ${stage}" | tee -a "${PIPELINE_LOG}"
  else
    local status=$?
    echo "[$(date '+%F %T')] [FAIL ] ${stage}, status=${status}" | tee -a "${PIPELINE_LOG}"
    echo "----- tail ${log} -----" >>"${PIPELINE_LOG}"
    tail -120 "${log}" >>"${PIPELINE_LOG}" 2>/dev/null || true
    return "${status}"
  fi
}

check_pseudo_count() {
  local dataset="$1"
  local fold_root="$2"
  local split_path="${fold_root}/meta/full_box_split_${METHOD_REPRO}.json"
  local teacher_dir="${fold_root}/pseudo_teacher/tri_train_${METHOD_REPRO}"
  local student_dir="${fold_root}/pseudo_student/tri_train_${METHOD_REPRO}"

  "$CONDA_BIN" run -n medsam310 --no-capture-output python - "$split_path" "$teacher_dir" "$student_dir" <<'PY'
import json
import sys
from pathlib import Path

split_path = Path(sys.argv[1])
teacher_dir = Path(sys.argv[2])
student_dir = Path(sys.argv[3])
obj = json.loads(split_path.read_text(encoding="utf-8"))
if isinstance(obj, list):
    rows = obj
elif isinstance(obj, dict):
    rows = next((obj[k] for k in ("records", "items", "samples", "data") if isinstance(obj.get(k), list)), None)
    if rows is None:
        raise RuntimeError(f"Cannot find split rows in {split_path}")
else:
    raise TypeError(type(obj))
expected = len(rows)
teacher = len(list(teacher_dir.glob("*.npy")))
student = len(list(student_dir.glob("*.npy")))
print(f"expected={expected} teacher={teacher} student={student}")
if teacher != expected or student != expected:
    raise SystemExit(2)
PY
}

gpu_is_free() {
  local gpu="$1"
  local free_mb util pids

  # Optional per-GPU reservation. While HOLD_GPU_<id> exists, this GPU
  # is not assigned even if nvidia-smi reports it idle.
  [[ -f "${VERIFY_ROOT}/HOLD_GPU_${gpu}" ]] && return 1

  free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' | tr '\n' ' ' || true)

  [[ "$free_mb" =~ ^[0-9]+$ ]] || return 1
  [[ "$util" =~ ^[0-9]+$ ]] || return 1
  (( free_mb >= MIN_FREE_MB )) || return 1
  (( util <= MAX_UTIL )) || return 1
  [[ -z "${pids// /}" ]] || return 1
  return 0
}

run_dataset_pipeline() {
  local dataset="$1"
  local gpu="$2"
  local fold_root="${PROC}/${dataset}/fold_0"
  local medsam_out="${MEDSAM_FT_ROOT}/${dataset}/fold_0"
  local student_out="${STUDENT_ROOT}/${dataset}/fold_0"
  local run_root="${VERIFY_ROOT}/${dataset}"
  local log_root="${run_root}/logs"
  local done_marker="${run_root}/DONE"
  local fail_marker="${run_root}/FAILED"

  mkdir -p "$log_root"
  PIPELINE_LOG="${run_root}/pipeline.log"
  export PIPELINE_LOG

  if [[ -f "$done_marker" ]]; then
    echo "[$(date '+%F %T')] [SKIP] ${dataset}: DONE marker exists"
    return 0
  fi

  rm -f "$fail_marker"
  echo "[$(date '+%F %T')] dataset=${dataset} gpu=${gpu}" >"$PIPELINE_LOG"

  require_dir "$fold_root"
  require_file "${fold_root}/meta/full_box_split_${METHOD_OLD}.json"
  require_file "${fold_root}/meta/full_box_split_summary_${METHOD_OLD}.json"
  require_file "$BASE_CKPT"
  require_file "$PRETRAIN"

  # Start this dataset from a clean repro state. Historical METHOD_OLD and frozen tri_train are untouched.
  rm -rf \
    "${fold_root}/pseudo_teacher/tri_train_${METHOD_REPRO}" \
    "${fold_root}/pseudo_student/tri_train_${METHOD_REPRO}" \
    "${MEDSAM_FT_ROOT:?}/${dataset}" \
    "${STUDENT_ROOT:?}/${dataset}"

  rm -f \
    "${fold_root}/meta/support_template_${METHOD_REPRO}.npz" \
    "${fold_root}/meta/support_template_stats_${METHOD_REPRO}.json" \
    "${fold_root}/meta/pseudo_generation_config_${METHOD_REPRO}.json" \
    "${fold_root}/meta/pseudo_quality_stats_${METHOD_REPRO}.csv"

  mkdir -p "$medsam_out" "$student_out" "${run_root}/06"

  cp -a \
    "${fold_root}/meta/full_box_split_${METHOD_OLD}.json" \
    "${fold_root}/meta/full_box_split_${METHOD_REPRO}.json"
  cp -a \
    "${fold_root}/meta/full_box_split_summary_${METHOD_OLD}.json" \
    "${fold_root}/meta/full_box_split_summary_${METHOD_REPRO}.json"

  local old_hash new_hash
  old_hash=$(sha256sum "${fold_root}/meta/full_box_split_${METHOD_OLD}.json" | awk '{print $1}')
  new_hash=$(sha256sum "${fold_root}/meta/full_box_split_${METHOD_REPRO}.json" | awk '{print $1}')
  [[ "$old_hash" == "$new_hash" ]] || { echo "split hash mismatch" >>"$PIPELINE_LOG"; return 10; }

  run_stage "02_support_template" "${log_root}/02_support_template.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n medsam310 --no-capture-output \
    python -u "${MEDSAM_CODE}/02_build_support_template.py" \
      --processed_root "$PROC" \
      --checkpoint "$BASE_CKPT" \
      --datasets "$dataset" \
      --fold fold_0 \
      --method "$METHOD_REPRO" \
      --device cuda \
      --seed 2026 \
      --k_fg 3 \
      --k_bg 5 \
      --max_fg_per_image 512 \
      --max_bg_per_image 512 \
      --max_total_features 200000 \
      --overwrite

  run_stage "03_sac_train" "${log_root}/03_sac_train.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n medsam310 --no-capture-output \
    python -u "${MEDSAM_CODE}/03_train_medsam_sac.py" \
      --processed_root "$PROC" \
      --checkpoint "$BASE_CKPT" \
      --datasets "$dataset" \
      --fold fold_0 \
      --method "$METHOD_REPRO" \
      --out_root "$MEDSAM_FT_ROOT" \
      --device cuda \
      --seed 2026 \
      --epochs 1 \
      --max_steps 0 \
      --lr 1e-5 \
      --weight_decay 0.01 \
      --ema_decay 0.99 \
      --max_grad_norm 1.0 \
      --lambda_out 1.0 \
      --lambda_seed 0.5 \
      --lambda_wac 0.5 \
      --lambda_proto 0.1 \
      --lambda_smooth 0.03 \
      --seed_p_threshold 0.65 \
      --seed_qf_threshold 0.55 \
      --proto_p_low 0.15 \
      --proto_p_high 0.85 \
      --weak1_low 0.30 \
      --weak1_high 0.40 \
      --weak2_low 0.40 \
      --weak2_high 0.50 \
      --weak2_weight 0.5 \
      --strong_p_threshold 0.50 \
      --strong_qf_threshold 0.55 \
      --wac_kernel 31 \
      --log_every 20

  require_file "${medsam_out}/medsam_sac_ema.pth"

  run_stage "04_generate_pseudo" "${log_root}/04_generate_pseudo.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n medsam310 --no-capture-output \
    python -u "${MEDSAM_CODE}/04_generate_pseudo_sac.py" \
      --processed_root "$PROC" \
      --base_checkpoint "$BASE_CKPT" \
      --sac_checkpoint "${medsam_out}/medsam_sac_ema.pth" \
      --datasets "$dataset" \
      --fold fold_0 \
      --method "$METHOD_REPRO" \
      --device cuda \
      --p_weight 0.65 \
      --qf_weight 0.35 \
      --p_threshold 0.50 \
      --q_threshold 0.58 \
      --fg_q_threshold 0.45 \
      --shape_threshold 0.10 \
      --bg_p_threshold 0.15 \
      --bg_qf_threshold 0.25 \
      --log_every 100 \
      --overwrite

  run_stage "04_count_audit" "${log_root}/04_count_audit.log" \
    check_pseudo_count "$dataset" "$fold_root"

  require_file "${fold_root}/meta/pseudo_generation_config_${METHOD_REPRO}.json"
  grep -q '"box_samples_load_gt": false' "${fold_root}/meta/pseudo_generation_config_${METHOD_REPRO}.json" \
    || { echo "box_samples_load_gt is not false" >>"$PIPELINE_LOG"; return 11; }

  if find \
      "${fold_root}/pseudo_teacher/tri_train_${METHOD_REPRO}" \
      "${fold_root}/pseudo_student/tri_train_${METHOD_REPRO}" \
      -type l -print -quit | grep -q .; then
    echo "symlink detected in repro pseudo directories" >>"$PIPELINE_LOG"
    return 12
  fi

  run_stage "06_box_only_audit" "${log_root}/06_box_only_audit.log" \
    "$CONDA_BIN" run -n medsam310 --no-capture-output \
    python -u "${MEDSAM_CODE}/06_compare_pseudo_quality.py" \
      --fold_root "$fold_root" \
      --baseline_tri_dir "${fold_root}/pseudo_student/tri_train" \
      --sac_tri_dir "${fold_root}/pseudo_student/tri_train_${METHOD_REPRO}" \
      --method "$METHOD_REPRO" \
      --space student \
      --only_box \
      --out_csv "${run_root}/06/compare_box_full.csv"

  run_stage "student_dataset_check" "${log_root}/student_dataset_check.log" \
    "$CONDA_BIN" run -n swin_umamba --no-capture-output \
    python -u "${SWIN_ROOT}/pipeline/test_student_patch_dataset.py" \
      --fold_root "$fold_root" \
      --dataset "$dataset" \
      --split train \
      --student_pseudo_name "$PSEUDO_NAME"

  run_stage "student_train" "${log_root}/student_train.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n swin_umamba --no-capture-output \
    python -u "${SWIN_ROOT}/pipeline/train_student.py" \
      --fold_root "$fold_root" \
      --dataset "$dataset" \
      --mode baseline \
      --student_pseudo_name "$PSEUDO_NAME" \
      --epochs 50 \
      --batch_size "${BATCH_SIZE[$dataset]}" \
      --num_workers "${NUM_WORKERS[$dataset]}" \
      --lr 1e-4 \
      --weight_decay 0.05 \
      --freeze_encoder_epochs 10 \
      --amp \
      --deep_supervision \
      --pretrained_ckpt "$PRETRAIN" \
      --out_dir "$student_out"

  require_file "${student_out}/last.pth"

  rm -rf "${student_out}/pred_test"
  run_stage "student_infer_test" "${log_root}/student_infer_test.log" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -n swin_umamba --no-capture-output \
    python -u "${SWIN_ROOT}/pipeline/infer_student.py" \
      --fold_root "$fold_root" \
      --dataset "$dataset" \
      --mode baseline \
      --deep_supervision \
      --ckpt "${student_out}/last.pth" \
      --out_dir "${student_out}/pred_test" \
      --amp

  if is_3d_dataset "$dataset"; then
    rm -rf "${student_out}/eval_3d"
    run_stage "student_eval_test_3d" "${log_root}/student_eval_test_3d.log" \
      "$CONDA_BIN" run -n swin_umamba --no-capture-output \
      python -u "${SWIN_ROOT}/pipeline/eval_3d.py" \
        --fold_root "$fold_root" \
        --pred_dir "${student_out}/pred_test" \
        --save_dir "${student_out}/eval_3d" \
        --split test \
        --require_native_gt
  else
    rm -rf "${student_out}/eval_2d"
    local -a eval_cmd=(
      "$CONDA_BIN" run -n swin_umamba --no-capture-output
      python -u "${SWIN_ROOT}/pipeline/eval_2d.py"
      --fold_root "$fold_root"
      --pred_dir "${student_out}/pred_test"
      --save_dir "${student_out}/eval_2d"
      --split test
      --require_native_gt
    )
    if report_mae_dataset "$dataset"; then
      eval_cmd+=(--report_mae)
    fi
    run_stage "student_eval_test_2d" "${log_root}/student_eval_test_2d.log" "${eval_cmd[@]}"
  fi

  date '+%F %T' >"$done_marker"
  echo "[$(date '+%F %T')] [SUCCESS] ${dataset}" | tee -a "$PIPELINE_LOG"
}

mkdir -p "$VERIFY_ROOT" "$MEDSAM_FT_ROOT" "$STUDENT_ROOT"
require_file "$CONDA_BIN"
require_dir "$MEDSAM_CODE"
require_dir "$SWIN_ROOT"
require_dir "$PROC"

SCHEDULER_LOG="${VERIFY_ROOT}/scheduler.log"
FAILED_LIST="${VERIFY_ROOT}/FAILED_DATASETS.txt"
: >"$FAILED_LIST"

echo "[$(date '+%F %T')] scheduler started" | tee -a "$SCHEDULER_LOG"
echo "datasets: ${DATASETS[*]}" | tee -a "$SCHEDULER_LOG"
echo "gpu pool: ${GPU_POOL[*]}" | tee -a "$SCHEDULER_LOG"

declare -A JOB_PID=()
declare -A JOB_DATASET=()
next_index=0

while (( next_index < ${#DATASETS[@]} || ${#JOB_PID[@]} > 0 )); do
  # Reap finished jobs.
  for gpu in "${!JOB_PID[@]}"; do
    pid="${JOB_PID[$gpu]}"
    dataset="${JOB_DATASET[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        status=0
      else
        status=$?
      fi
      echo "[$(date '+%F %T')] finished dataset=${dataset} gpu=${gpu} status=${status}" | tee -a "$SCHEDULER_LOG"
      if (( status != 0 )); then
        echo "$dataset" >>"$FAILED_LIST"
        date '+%F %T' >"${VERIFY_ROOT}/${dataset}/FAILED"
      fi
      unset 'JOB_PID[$gpu]'
      unset 'JOB_DATASET[$gpu]'
    fi
  done

  # Creating ${VERIFY_ROOT}/PAUSE pauses only new launches; running datasets continue.
  if [[ -f "${VERIFY_ROOT}/PAUSE" ]]; then
    echo "[$(date '+%F %T')] queue paused by ${VERIFY_ROOT}/PAUSE" >>"$SCHEDULER_LOG"
    sleep "$POLL_SECONDS"
    continue
  fi

  # Launch queued datasets on externally and internally idle GPUs.
  for gpu in "${GPU_POOL[@]}"; do
    (( next_index < ${#DATASETS[@]} )) || break
    [[ -z "${JOB_PID[$gpu]:-}" ]] || continue
    gpu_is_free "$gpu" || continue

    dataset="${DATASETS[$next_index]}"
    ((next_index += 1))

    echo "[$(date '+%F %T')] launch dataset=${dataset} gpu=${gpu}" | tee -a "$SCHEDULER_LOG"
    run_dataset_pipeline "$dataset" "$gpu" \
      >"${VERIFY_ROOT}/${dataset}.launcher.log" 2>&1 &
    JOB_PID[$gpu]=$!
    JOB_DATASET[$gpu]="$dataset"
  done

  sleep "$POLL_SECONDS"
done

if [[ -s "$FAILED_LIST" ]]; then
  echo "[$(date '+%F %T')] completed with failures:" | tee -a "$SCHEDULER_LOG"
  cat "$FAILED_LIST" | tee -a "$SCHEDULER_LOG"
  exit 1
fi

echo "[$(date '+%F %T')] all queued datasets completed successfully" | tee -a "$SCHEDULER_LOG"
