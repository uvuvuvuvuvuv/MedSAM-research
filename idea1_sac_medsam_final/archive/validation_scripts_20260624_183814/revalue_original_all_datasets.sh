#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Re-evaluate frozen original pseudo labels with the current 05 and 06.
# This script does not train and does not generate/overwrite pseudo labels.
#
# Run only after 2D + 3D smoke and 04 equivalence have passed.
#
# Usage:
#   conda activate medsam310
#   bash revalue_original_all_datasets.sh
#
# Optional explicit dataset list:
#   DATASETS_CSV="kvasirseg,cvc_clinicdb,..." bash revalue_original_all_datasets.sh

SCRIPT_ROOT="${SCRIPT_ROOT:-/storage/baiyuting/data/MedSAM-main/idea1_sac_medsam_final}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed}"
OLD_METHOD="${OLD_METHOD:-idea1_sac_medsam_final}"
FOLD="${FOLD:-fold_0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/baiyuting/data/out_data_idea1/visualization/pseudo_quality/revalue_${OLD_METHOD}_clean_metrics}"
DATASETS_CSV="${DATASETS_CSV:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "medsam310" ]]; then
  echo "[ERROR] activate medsam310 first." >&2
  exit 1
fi

python -m py_compile "${SCRIPT_ROOT}/05_visualize_pseudo_sac.py"
python -m py_compile "${SCRIPT_ROOT}/06_compare_pseudo_quality.py"

if [[ -z "${DATASETS_CSV}" ]]; then
  mapfile -t DATASETS < <(
    find "${PROCESSED_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
    | while read -r ds; do
        if [[ -d "${PROCESSED_ROOT}/${ds}/${FOLD}/pseudo_student/tri_train_${OLD_METHOD}" ]]; then
          echo "${ds}"
        fi
      done \
    | sort
  )
else
  IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
fi

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "[ERROR] no datasets with frozen SAC pseudo labels were found." >&2
  exit 1
fi

printf 'Datasets (%d):\n' "${#DATASETS[@]}"
printf '  %s\n' "${DATASETS[@]}"

mkdir -p "${OUTPUT_ROOT}/logs"

for DATASET in "${DATASETS[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  [[ -n "${DATASET}" ]] || continue

  FOLD_ROOT="${PROCESSED_ROOT}/${DATASET}/${FOLD}"
  BASELINE_DIR="${FOLD_ROOT}/pseudo_student/tri_train"
  SAC_DIR="${FOLD_ROOT}/pseudo_student/tri_train_${OLD_METHOD}"
  DATASET_OUT="${OUTPUT_ROOT}/${DATASET}"
  COMPARE_OUT="${DATASET_OUT}/compare_box_full.csv"

  echo
  echo "===== ${DATASET}: 05 visualization ====="

  rm -rf -- "${DATASET_OUT}"

  python -u "${SCRIPT_ROOT}/05_visualize_pseudo_sac.py" \
    --processed-root "${PROCESSED_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --datasets "${DATASET}" \
    --fold "${FOLD}" \
    --method "${OLD_METHOD}" \
    --baseline-pseudo-name tri_train \
    --sac-pseudo-name "tri_train_${OLD_METHOD}" \
    --prompt-space teacher \
    --display-mode native \
    --per-group 5 \
    --max-per-case 3 \
    --max-samples-scan -1 \
    --summary-scope box \
    --overlay-alpha 0.58 \
    --error-alpha 0.62 \
    --dpi 220 \
    --sheet-dpi 180 \
    --include-full \
    --render-individual \
    --render-contact-sheets \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${DATASET}_05.log"

  echo "===== ${DATASET}: 06 paired comparison ====="
  python -u "${SCRIPT_ROOT}/06_compare_pseudo_quality.py" \
    --fold_root "${FOLD_ROOT}" \
    --baseline_tri_dir "${BASELINE_DIR}" \
    --sac_tri_dir "${SAC_DIR}" \
    --method "${OLD_METHOD}" \
    --space student \
    --only_box \
    --out_csv "${COMPARE_OUT}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${DATASET}_06.log"

  DATASET_OUT="${DATASET_OUT}" COMPARE_OUT="${COMPARE_OUT}" python - <<'PY'
import json
import os
from pathlib import Path
import pandas as pd

dataset_out = Path(os.environ["DATASET_OUT"])
compare_out = Path(os.environ["COMPARE_OUT"])

summary = json.loads((dataset_out / "summary.json").read_text(encoding="utf-8"))
paired = pd.read_csv(compare_out.with_name(compare_out.stem + "_paired_delta.csv"))
samples = pd.read_csv(compare_out)

missing = samples.groupby("method")["missing"].sum().to_dict()
expected_box = int(summary["num_box_visualized"])

assert summary["num_samples_in_summary"] == expected_box, summary
assert summary["warnings_count"] == 0, summary
assert missing.get("baseline_box_only") == 0, missing
assert missing.get("sac_medsam_final") == 0, missing
assert set(paired["paired_samples"].astype(int).tolist()) == {expected_box}, paired

print(
    f"[OK] {summary['dataset']}: "
    f"visualized={summary['num_samples_visualized']} "
    f"box_paired={expected_box}"
)
PY
done

echo
echo "[OK] Frozen original results were re-evaluated without retraining."
echo "Output root: ${OUTPUT_ROOT}"
