#!/usr/bin/env bash
set -Eeuo pipefail

# Synapse/BTCV 3D multi-class clean smoke:
# 01 split -> 02 support template -> 03 short training ->
# 04 foreground-bearing class-stratified pseudo generation ->
# 05 visualization -> 06 paired comparison
#
# No additional helper script is required.
#
# Fresh run:
#   conda activate medsam310
#   RESET=1 bash run_synapse_clean_smoke.sh
#
# Resume from the already completed 01-03 stages:
#   START_STAGE=4 bash run_synapse_clean_smoke.sh
#
# Optional:
#   DATASET=btcv RESET=1 bash run_synapse_clean_smoke.sh
#   MAX_STEPS=96 MAX_SAMPLES=32 RESET=1 bash run_synapse_clean_smoke.sh

SCRIPT_ROOT="${SCRIPT_ROOT:-/storage/baiyuting/data/MedSAM-main/idea1_sac_medsam_final}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed}"
BASE_CKPT="${BASE_CKPT:-/storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth}"

DATASET="${DATASET:-synapse}"
FOLD="${FOLD:-fold_0}"
METHOD="${METHOD:-idea1_sac_medsam_clean_${DATASET}_smoke}"
TRAIN_ROOT="${TRAIN_ROOT:-/storage/baiyuting/data/MedSAM-main/MedSAM_ft/smoke_clean_${DATASET}}"
VIS_OUT="${VIS_OUT:-/storage/baiyuting/data/out_data_idea1/visualization/pseudo_quality/${METHOD}}"

MAX_STEPS="${MAX_STEPS:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-24}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-6}"
RESET="${RESET:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FOLD_ROOT="${PROCESSED_ROOT}/${DATASET}/${FOLD}"
META_ROOT="${FOLD_ROOT}/meta"
TRAIN_DS_ROOT="${TRAIN_ROOT}/${DATASET}/${FOLD}"
SAC_CKPT="${TRAIN_DS_ROOT}/medsam_sac_ema.pth"
BASELINE_DIR="${FOLD_ROOT}/pseudo_student/tri_train"
TEACHER_DIR="${FOLD_ROOT}/pseudo_teacher/tri_train_${METHOD}"
SAC_DIR="${FOLD_ROOT}/pseudo_student/tri_train_${METHOD}"
COMPARE_OUT="${VIS_OUT}/${DATASET}/compare_box_smoke.csv"
LOG_ROOT="${TRAIN_ROOT}/pipeline_logs"

FULL_SPLIT="${META_ROOT}/full_box_split_${METHOD}.json"
FULL_SPLIT_SUMMARY="${META_ROOT}/full_box_split_summary_${METHOD}.json"
FULL_SPLIT_BACKUP="${META_ROOT}/full_box_split_${METHOD}.before_fg_smoke.json"
SUBSET_SUMMARY="${META_ROOT}/foreground_smoke_subset_${METHOD}.json"

mkdir -p "${LOG_ROOT}"

SUBSET_ACTIVE=0

restore_full_split() {
  if [[ "${SUBSET_ACTIVE}" == "1" && -f "${FULL_SPLIT_BACKUP}" ]]; then
    cp -f "${FULL_SPLIT_BACKUP}" "${FULL_SPLIT}"
    rm -f "${FULL_SPLIT_BACKUP}"
    SUBSET_ACTIVE=0
    echo "[CLEANUP] restored full split: ${FULL_SPLIT}"
  fi
}

on_error() {
  local exit_code=$?
  echo "[ERROR] line ${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
  restore_full_split
  exit "${exit_code}"
}

trap on_error ERR
trap restore_full_split EXIT

require_file() {
  [[ -f "$1" ]] || {
    echo "[ERROR] missing file: $1" >&2
    exit 1
  }
}

require_dir() {
  [[ -d "$1" ]] || {
    echo "[ERROR] missing directory: $1" >&2
    exit 1
  }
}

if [[ "${CONDA_DEFAULT_ENV:-}" != "medsam310" ]]; then
  echo "[ERROR] activate medsam310 first." >&2
  exit 1
fi

require_dir "${FOLD_ROOT}"
require_file "${META_ROOT}/manifest.json"
require_file "${BASE_CKPT}"
require_dir "${BASELINE_DIR}"

for s in \
  01_build_full_box_split.py \
  02_build_support_template.py \
  03_train_medsam_sac.py \
  04_generate_pseudo_sac.py \
  05_visualize_pseudo_sac.py \
  06_compare_pseudo_quality.py
do
  require_file "${SCRIPT_ROOT}/${s}"
  python -m py_compile "${SCRIPT_ROOT}/${s}"
done

# Recover from a previous interrupted run that temporarily replaced the split.
if [[ -f "${FULL_SPLIT_BACKUP}" ]]; then
  echo "[RECOVER] stale split backup found; restoring it first."
  cp -f "${FULL_SPLIT_BACKUP}" "${FULL_SPLIT}"
  rm -f "${FULL_SPLIT_BACKUP}"
fi

if [[ "${RESET}" == "1" ]]; then
  rm -rf -- "${TRAIN_DS_ROOT}"
  rm -rf -- "${VIS_OUT}/${DATASET}"
  rm -rf -- "${TEACHER_DIR}" "${SAC_DIR}"
  rm -f -- \
    "${FULL_SPLIT}" \
    "${FULL_SPLIT_SUMMARY}" \
    "${FULL_SPLIT_BACKUP}" \
    "${SUBSET_SUMMARY}" \
    "${META_ROOT}/support_template_${METHOD}.npz" \
    "${META_ROOT}/support_template_stats_${METHOD}.json" \
    "${META_ROOT}/pseudo_quality_stats_${METHOD}.csv" \
    "${META_ROOT}/pseudo_generation_config_${METHOD}.json"
fi

if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
  echo "[1/6] split"
  python -u "${SCRIPT_ROOT}/01_build_full_box_split.py" \
    --processed_root "${PROCESSED_ROOT}" \
    --datasets "${DATASET}" \
    --fold "${FOLD}" \
    --method "${METHOD}" \
    --full_ratio 0.10 \
    --seed 2026 \
    --overwrite | tee "${LOG_ROOT}/01_split.log"
fi

require_file "${FULL_SPLIT}"

if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
  echo "[2/6] support template"
  python -u "${SCRIPT_ROOT}/02_build_support_template.py" \
    --processed_root "${PROCESSED_ROOT}" \
    --checkpoint "${BASE_CKPT}" \
    --datasets "${DATASET}" \
    --fold "${FOLD}" \
    --method "${METHOD}" \
    --device cuda \
    --seed 2026 \
    --k_fg 3 \
    --k_bg 5 \
    --max_fg_per_image 512 \
    --max_bg_per_image 512 \
    --max_total_features 200000 \
    --overwrite | tee "${LOG_ROOT}/02_template.log"
fi

require_file "${META_ROOT}/support_template_${METHOD}.npz"

DATASET="${DATASET}" METHOD="${METHOD}" META_ROOT="${META_ROOT}" python - <<'PY'
import os
from pathlib import Path
import numpy as np

meta = Path(os.environ["META_ROOT"])
method = os.environ["METHOD"]
data = np.load(meta / f"support_template_{method}.npz")
class_ids = [int(v) for v in data["class_ids"].tolist()]
print("class_ids =", class_ids)
assert len(class_ids) >= 2, f"Expected multi-class template, got {class_ids}"
for cid in class_ids:
    assert f"proto_fg_c{cid}" in data.files
    assert f"proto_bg_c{cid}" in data.files
    assert f"shape_A_c{cid}" in data.files
print("[OK] multi-class support template verified")
PY

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
  if [[ -e "${TRAIN_DS_ROOT}" ]]; then
    echo "[ERROR] smoke training output exists: ${TRAIN_DS_ROOT}" >&2
    echo "        use RESET=1 for a fresh run, or START_STAGE=4 to reuse it." >&2
    exit 1
  fi

  echo "[3/6] train ${MAX_STEPS} steps"
  python -u "${SCRIPT_ROOT}/03_train_medsam_sac.py" \
    --processed_root "${PROCESSED_ROOT}" \
    --checkpoint "${BASE_CKPT}" \
    --datasets "${DATASET}" \
    --fold "${FOLD}" \
    --method "${METHOD}" \
    --out_root "${TRAIN_ROOT}" \
    --device cuda \
    --seed 2026 \
    --epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --ema_decay 0.99 \
    --lambda_out 1.0 \
    --lambda_seed 0.5 \
    --lambda_wac 0.5 \
    --lambda_proto 0.3 \
    --lambda_smooth 0.03 \
    --log_every 1 | tee "${LOG_ROOT}/03_train.log"

  TRAIN_DS_ROOT="${TRAIN_DS_ROOT}" METHOD="${METHOD}" MAX_STEPS="${MAX_STEPS}" python - <<'PY'
import os
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(os.environ["TRAIN_DS_ROOT"])
method = os.environ["METHOD"]
expected = int(os.environ["MAX_STEPS"])
df = pd.read_csv(root / f"medsam_ft_log_{method}.csv")
assert len(df) == expected, (len(df), expected)
assert {"full", "box"}.issubset(set(df["label_mode"]))
assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()
assert np.allclose(df.loc[df.label_mode == "full", "mean_qf"], 0.0)
print("label_ids seen =", sorted(df["label_id"].unique().tolist()))
print("[OK] clean 3D training smoke verified")
PY
fi

require_file "${SAC_CKPT}"

# Stages 4-6 use a temporary foreground-bearing, class-stratified subset.
# The original full split is restored automatically, including after an error.
if (( START_STAGE <= 6 && END_STAGE >= 4 )); then
  echo "[PREP] build foreground-bearing class-stratified smoke subset"

  cp -f "${FULL_SPLIT}" "${FULL_SPLIT_BACKUP}"
  SUBSET_ACTIVE=1

  PROCESSED_ROOT="${PROCESSED_ROOT}" \
  DATASET="${DATASET}" \
  FOLD="${FOLD}" \
  METHOD="${METHOD}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  FULL_SPLIT_BACKUP="${FULL_SPLIT_BACKUP}" \
  FULL_SPLIT="${FULL_SPLIT}" \
  SUBSET_SUMMARY="${SUBSET_SUMMARY}" \
  BASELINE_DIR="${BASELINE_DIR}" \
  python - <<'PY'
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

processed_root = Path(os.environ["PROCESSED_ROOT"])
dataset = os.environ["DATASET"]
fold = os.environ["FOLD"]
method = os.environ["METHOD"]
max_samples = int(os.environ["MAX_SAMPLES"])
source_split = Path(os.environ["FULL_SPLIT_BACKUP"])
target_split = Path(os.environ["FULL_SPLIT"])
summary_path = Path(os.environ["SUBSET_SUMMARY"])
baseline_dir = Path(os.environ["BASELINE_DIR"])

fold_root = processed_root / dataset / fold
meta_root = fold_root / "meta"

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

raw_manifest = load_json(meta_root / "manifest.json")
manifest = raw_manifest["samples"] if isinstance(raw_manifest, dict) and isinstance(raw_manifest.get("samples"), list) else raw_manifest
manifest_by_name = {
    str(item.get("slice_name") or item.get("filename")): item
    for item in manifest
    if item.get("slice_name") or item.get("filename")
}

prompts = load_json(fold_root / "prompts" / "prompts_train.json")
split_records = load_json(source_split)
template = np.load(meta_root / f"support_template_{method}.npz")
class_ids = [int(v) for v in template["class_ids"].tolist()]

def prompt_labels(meta):
    labels = set()
    if not isinstance(meta, dict):
        return labels
    for ins in meta.get("instances", []):
        if not isinstance(ins, dict):
            continue
        if ins.get("bbox_teacher", ins.get("bbox")) is None:
            continue
        cid = int(ins.get("label_id", 1))
        if cid > 0 and cid != 255:
            labels.add(cid)
    return labels

candidates = []
for rec in split_records:
    if str(rec.get("label_mode", "box")).lower() != "box":
        continue

    name = str(rec["slice_name"])
    labels = prompt_labels(prompts.get(name))
    if not labels:
        continue

    baseline_path = baseline_dir / name
    if not baseline_path.exists():
        continue
    baseline = np.load(baseline_path)
    baseline_fg = {
        int(v) for v in np.unique(baseline).tolist()
        if int(v) not in (0, 255)
    }
    if not baseline_fg:
        continue

    item = manifest_by_name.get(name, {})
    case_id = str(item.get("case_id") or "unknown_case")
    try:
        slice_idx = int(item.get("slice_idx", -1))
    except (TypeError, ValueError):
        slice_idx = -1

    candidates.append({
        "record": rec,
        "slice_name": name,
        "labels": labels,
        "baseline_fg": baseline_fg,
        "case_id": case_id,
        "slice_idx": slice_idx,
    })

assert candidates, "No foreground-bearing prompt candidates were found."

by_class = defaultdict(list)
for item in candidates:
    for cid in item["labels"]:
        by_class[cid].append(item)

selected = []
selected_names = set()
case_counts = Counter()
prompt_class_counts = Counter()
baseline_class_counts = Counter()

while len(selected) < max_samples:
    progress = False
    for cid in class_ids:
        pool = [
            item for item in by_class.get(cid, [])
            if item["slice_name"] not in selected_names
        ]
        if not pool:
            continue
        pool.sort(
            key=lambda x: (
                case_counts[x["case_id"]],
                -len(x["baseline_fg"]),
                x["case_id"],
                x["slice_idx"],
                x["slice_name"],
            )
        )
        item = pool[0]
        selected.append(item)
        selected_names.add(item["slice_name"])
        case_counts[item["case_id"]] += 1
        prompt_class_counts.update(item["labels"])
        baseline_class_counts.update(item["baseline_fg"])
        progress = True
        if len(selected) >= max_samples:
            break
    if not progress:
        break

if len(selected) < max_samples:
    remaining = [
        item for item in candidates
        if item["slice_name"] not in selected_names
    ]
    remaining.sort(
        key=lambda x: (
            case_counts[x["case_id"]],
            -len(x["labels"]),
            -len(x["baseline_fg"]),
            x["case_id"],
            x["slice_idx"],
            x["slice_name"],
        )
    )
    for item in remaining:
        selected.append(item)
        selected_names.add(item["slice_name"])
        case_counts[item["case_id"]] += 1
        prompt_class_counts.update(item["labels"])
        baseline_class_counts.update(item["baseline_fg"])
        if len(selected) >= max_samples:
            break

assert len(selected) == max_samples, (len(selected), max_samples)

target_split.write_text(
    json.dumps([item["record"] for item in selected], ensure_ascii=False, indent=2),
    encoding="utf-8",
)

summary = {
    "dataset": dataset,
    "fold": fold,
    "method": method,
    "num_samples": len(selected),
    "selection": "box + non-empty prompt + baseline foreground + class/case stratification",
    "template_class_ids": class_ids,
    "prompt_class_counts": {str(k): int(v) for k, v in sorted(prompt_class_counts.items())},
    "baseline_class_counts": {str(k): int(v) for k, v in sorted(baseline_class_counts.items())},
    "case_counts": dict(case_counts),
    "selected": [
        {
            "slice_name": item["slice_name"],
            "case_id": item["case_id"],
            "slice_idx": item["slice_idx"],
            "prompt_labels": sorted(item["labels"]),
            "baseline_foreground_labels": sorted(item["baseline_fg"]),
        }
        for item in selected
    ],
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[OK] temporary subset split: {target_split}")
PY
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
  echo "[4/6] generate ${MAX_SAMPLES} foreground-bearing pseudo labels"

  python -u "${SCRIPT_ROOT}/04_generate_pseudo_sac.py" \
    --processed_root "${PROCESSED_ROOT}" \
    --base_checkpoint "${BASE_CKPT}" \
    --sac_checkpoint "${SAC_CKPT}" \
    --datasets "${DATASET}" \
    --fold "${FOLD}" \
    --method "${METHOD}" \
    --device cuda \
    --max_samples 0 \
    --p_weight 0.65 \
    --qf_weight 0.35 \
    --p_threshold 0.50 \
    --q_threshold 0.58 \
    --fg_q_threshold 0.45 \
    --shape_threshold 0.10 \
    --bg_p_threshold 0.15 \
    --bg_qf_threshold 0.25 \
    --log_every 1 \
    --overwrite | tee "${LOG_ROOT}/04_generate.log"

  FOLD_ROOT="${FOLD_ROOT}" METHOD="${METHOD}" MAX_SAMPLES="${MAX_SAMPLES}" python - <<'PY'
import os
from pathlib import Path
import numpy as np

fold = Path(os.environ["FOLD_ROOT"])
method = os.environ["METHOD"]
expected = int(os.environ["MAX_SAMPLES"])

teacher = sorted((fold / "pseudo_teacher" / f"tri_train_{method}").glob("*.npy"))
student = sorted((fold / "pseudo_student" / f"tri_train_{method}").glob("*.npy"))

assert len(teacher) == expected, len(teacher)
assert len(student) == expected, len(student)
assert {p.name for p in teacher} == {p.name for p in student}

teacher_classes = set()
student_classes = set()

for p in teacher:
    teacher_classes.update(
        int(v) for v in np.unique(np.load(p)).tolist()
        if int(v) not in (0, 255)
    )
for p in student:
    student_classes.update(
        int(v) for v in np.unique(np.load(p)).tolist()
        if int(v) not in (0, 255)
    )

print("teacher foreground class IDs =", sorted(teacher_classes))
print("student foreground class IDs =", sorted(student_classes))

assert teacher_classes, "No foreground class was generated in teacher space."
assert student_classes, "No foreground class was generated in student space."

if len(student_classes) < 2:
    print(
        "[WARN] Only one foreground class was generated. "
        "The pipeline is functional, but 64-step model coverage is limited."
    )
else:
    print("[OK] multi-class pseudo generation verified")
PY
fi

if (( START_STAGE <= 5 && END_STAGE >= 5 )); then
  echo "[5/6] visualize selected smoke samples"
  rm -rf -- "${VIS_OUT}/${DATASET}"

  python -u "${SCRIPT_ROOT}/05_visualize_pseudo_sac.py" \
    --processed-root "${PROCESSED_ROOT}" \
    --output-root "${VIS_OUT}" \
    --datasets "${DATASET}" \
    --fold "${FOLD}" \
    --method "${METHOD}" \
    --baseline-pseudo-name tri_train \
    --sac-pseudo-name "tri_train_${METHOD}" \
    --prompt-space teacher \
    --display-mode native \
    --per-group 2 \
    --max-per-case 2 \
    --max-samples-scan "${MAX_SAMPLES}" \
    --summary-scope box \
    --overlay-alpha 0.58 \
    --error-alpha 0.62 \
    --dpi 180 \
    --sheet-dpi 160 \
    --render-individual \
    --render-contact-sheets \
    --allow-partial | tee "${LOG_ROOT}/05_visualize.log"
fi

if (( START_STAGE <= 6 && END_STAGE >= 6 )); then
  echo "[6/6] paired comparison"

  python -u "${SCRIPT_ROOT}/06_compare_pseudo_quality.py" \
    --fold_root "${FOLD_ROOT}" \
    --baseline_tri_dir "${BASELINE_DIR}" \
    --sac_tri_dir "${SAC_DIR}" \
    --method "${METHOD}" \
    --space student \
    --only_box \
    --out_csv "${COMPARE_OUT}" | tee "${LOG_ROOT}/06_compare.log"

  COMPARE_OUT="${COMPARE_OUT}" MAX_SAMPLES="${MAX_SAMPLES}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

out = Path(os.environ["COMPARE_OUT"])
expected = int(os.environ["MAX_SAMPLES"])
paired = pd.read_csv(out.with_name(out.stem + "_paired_delta.csv"))
assert set(paired["paired_samples"].astype(int).tolist()) == {expected}, paired
print(f"[OK] paired comparison verified: {expected} Box samples")
PY
fi

echo
echo "[OK] Synapse/BTCV clean smoke complete"
echo "training:       ${TRAIN_DS_ROOT}"
echo "pseudo labels:  ${SAC_DIR}"
echo "visualization:  ${VIS_OUT}/${DATASET}"
echo "comparison:     ${COMPARE_OUT}"
echo "subset summary: ${SUBSET_SUMMARY}"
