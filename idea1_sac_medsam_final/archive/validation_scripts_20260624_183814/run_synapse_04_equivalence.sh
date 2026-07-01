#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Compare current clean 04 output with the frozen original output,
# using the same original split, template, checkpoint, and thresholds.
#
# The original pseudo-label directories are never overwritten.
#
# Usage:
#   conda activate medsam310
#   bash run_synapse_04_equivalence.sh
#
# Optional:
#   DATASET=btcv MAX_SAMPLES=32 RESET=1 bash run_synapse_04_equivalence.sh

SCRIPT_ROOT="${SCRIPT_ROOT:-/storage/baiyuting/data/MedSAM-main/idea1_sac_medsam_final}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed}"
BASE_CKPT="${BASE_CKPT:-/storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${SCRIPT_ROOT}/compare_pseudo_dirs.py}"

DATASET="${DATASET:-synapse}"
FOLD="${FOLD:-fold_0}"
OLD_METHOD="${OLD_METHOD:-idea1_sac_medsam_final}"
VERIFY_METHOD="${VERIFY_METHOD:-idea1_sac_medsam_clean_04eq_${DATASET}}"
MAX_SAMPLES="${MAX_SAMPLES:-24}"
RESET="${RESET:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FOLD_ROOT="${PROCESSED_ROOT}/${DATASET}/${FOLD}"
META_ROOT="${FOLD_ROOT}/meta"

OLD_TEACHER="${FOLD_ROOT}/pseudo_teacher/tri_train_${OLD_METHOD}"
OLD_STUDENT="${FOLD_ROOT}/pseudo_student/tri_train_${OLD_METHOD}"
NEW_TEACHER="${FOLD_ROOT}/pseudo_teacher/tri_train_${VERIFY_METHOD}"
NEW_STUDENT="${FOLD_ROOT}/pseudo_student/tri_train_${VERIFY_METHOD}"

OUT_ROOT="${OUT_ROOT:-/storage/baiyuting/data/out_data_idea1/verification/04_equivalence/${DATASET}}"

EXPECTED_CKPT="/storage/baiyuting/data/MedSAM-main/MedSAM_ft/${OLD_METHOD}/${DATASET}/${FOLD}/medsam_sac_ema.pth"
OLD_SAC_CKPT="${OLD_SAC_CKPT:-${EXPECTED_CKPT}}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 1; }; }

if [[ "${CONDA_DEFAULT_ENV:-}" != "medsam310" ]]; then
  echo "[ERROR] activate medsam310 first." >&2
  exit 1
fi

if [[ ! -f "${OLD_SAC_CKPT}" ]]; then
  OLD_SAC_CKPT="$(find /storage/baiyuting/data/MedSAM-main/MedSAM_ft \
    -type f -path "*/${DATASET}/${FOLD}/medsam_sac_ema.pth" \
    | grep "/${OLD_METHOD}/" | head -n 1 || true)"
fi

require_file "${SCRIPT_ROOT}/04_generate_pseudo_sac.py"
require_file "${COMPARE_SCRIPT}"
require_file "${BASE_CKPT}"
require_file "${OLD_SAC_CKPT}"
require_dir "${OLD_TEACHER}"
require_dir "${OLD_STUDENT}"
require_file "${META_ROOT}/full_box_split_${OLD_METHOD}.json"
require_file "${META_ROOT}/support_template_${OLD_METHOD}.npz"

echo "OLD_SAC_CKPT=${OLD_SAC_CKPT}"
echo "OLD_TEACHER=${OLD_TEACHER}"
echo "OLD_STUDENT=${OLD_STUDENT}"
echo "VERIFY_METHOD=${VERIFY_METHOD}"

if [[ "${RESET}" == "1" ]]; then
  rm -rf -- "${NEW_TEACHER}" "${NEW_STUDENT}" "${OUT_ROOT}"
  rm -f -- \
    "${META_ROOT}/full_box_split_${VERIFY_METHOD}.json" \
    "${META_ROOT}/full_box_split_summary_${VERIFY_METHOD}.json" \
    "${META_ROOT}/support_template_${VERIFY_METHOD}.npz" \
    "${META_ROOT}/support_template_stats_${VERIFY_METHOD}.json" \
    "${META_ROOT}/pseudo_quality_stats_${VERIFY_METHOD}.csv" \
    "${META_ROOT}/pseudo_generation_config_${VERIFY_METHOD}.json"
fi

# Create method aliases so the current 04 writes to an isolated directory
# while reading exactly the frozen original split/template.
cp -f \
  "${META_ROOT}/full_box_split_${OLD_METHOD}.json" \
  "${META_ROOT}/full_box_split_${VERIFY_METHOD}.json"

if [[ -f "${META_ROOT}/full_box_split_summary_${OLD_METHOD}.json" ]]; then
  cp -f \
    "${META_ROOT}/full_box_split_summary_${OLD_METHOD}.json" \
    "${META_ROOT}/full_box_split_summary_${VERIFY_METHOD}.json"
fi

cp -f \
  "${META_ROOT}/support_template_${OLD_METHOD}.npz" \
  "${META_ROOT}/support_template_${VERIFY_METHOD}.npz"

if [[ -f "${META_ROOT}/support_template_stats_${OLD_METHOD}.json" ]]; then
  cp -f \
    "${META_ROOT}/support_template_stats_${OLD_METHOD}.json" \
    "${META_ROOT}/support_template_stats_${VERIFY_METHOD}.json"
fi

# Reuse the exact saved generation thresholds when available.
CFG="${META_ROOT}/pseudo_generation_config_${OLD_METHOD}.json"
read -r P_WEIGHT QF_WEIGHT P_THRESHOLD Q_THRESHOLD FG_Q_THRESHOLD SHAPE_THRESHOLD BG_P_THRESHOLD BG_QF_THRESHOLD < <(
  CFG="${CFG}" python - <<'PY'
import json
import os
from pathlib import Path

defaults = {
    "p_weight": 0.65,
    "qf_weight": 0.35,
    "p_threshold": 0.50,
    "q_threshold": 0.58,
    "fg_q_threshold": 0.45,
    "shape_threshold": 0.10,
    "bg_p_threshold": 0.15,
    "bg_qf_threshold": 0.25,
}
path = Path(os.environ["CFG"])
if path.exists():
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for k in defaults:
        if k in cfg:
            defaults[k] = cfg[k]
print(*(defaults[k] for k in [
    "p_weight", "qf_weight", "p_threshold", "q_threshold",
    "fg_q_threshold", "shape_threshold", "bg_p_threshold",
    "bg_qf_threshold"
]))
PY
)

echo "generation parameters:"
echo "  p_weight=${P_WEIGHT}"
echo "  qf_weight=${QF_WEIGHT}"
echo "  p_threshold=${P_THRESHOLD}"
echo "  q_threshold=${Q_THRESHOLD}"
echo "  fg_q_threshold=${FG_Q_THRESHOLD}"
echo "  shape_threshold=${SHAPE_THRESHOLD}"
echo "  bg_p_threshold=${BG_P_THRESHOLD}"
echo "  bg_qf_threshold=${BG_QF_THRESHOLD}"

python -u "${SCRIPT_ROOT}/04_generate_pseudo_sac.py" \
  --processed_root "${PROCESSED_ROOT}" \
  --base_checkpoint "${BASE_CKPT}" \
  --sac_checkpoint "${OLD_SAC_CKPT}" \
  --datasets "${DATASET}" \
  --fold "${FOLD}" \
  --method "${VERIFY_METHOD}" \
  --device cuda \
  --max_samples "${MAX_SAMPLES}" \
  --p_weight "${P_WEIGHT}" \
  --qf_weight "${QF_WEIGHT}" \
  --p_threshold "${P_THRESHOLD}" \
  --q_threshold "${Q_THRESHOLD}" \
  --fg_q_threshold "${FG_Q_THRESHOLD}" \
  --shape_threshold "${SHAPE_THRESHOLD}" \
  --bg_p_threshold "${BG_P_THRESHOLD}" \
  --bg_qf_threshold "${BG_QF_THRESHOLD}" \
  --log_every 1 \
  --overwrite

mkdir -p "${OUT_ROOT}"

echo "[compare] teacher space"
python "${COMPARE_SCRIPT}" \
  --old-dir "${OLD_TEACHER}" \
  --new-dir "${NEW_TEACHER}" \
  --out-dir "${OUT_ROOT}" \
  --tag teacher

echo "[compare] student space"
python "${COMPARE_SCRIPT}" \
  --old-dir "${OLD_STUDENT}" \
  --new-dir "${NEW_STUDENT}" \
  --out-dir "${OUT_ROOT}" \
  --tag student

echo
echo "Review:"
echo "  ${OUT_ROOT}/teacher_summary.json"
echo "  ${OUT_ROOT}/student_summary.json"
echo
echo "Exact-equivalence acceptance:"
echo "  common_file_count = ${MAX_SAMPLES}"
echo "  non_exact_count = 0"
echo "  overall_diff_ratio = 0"
echo
echo "Do not add --require-exact until the summaries have been reviewed."
