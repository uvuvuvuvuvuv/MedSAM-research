#!/usr/bin/env bash
set -euo pipefail

MEDSAM_REPO="${MEDSAM_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/MedSAM-main}"
SWIN_REPO="${SWIN_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/Swin-UMamba-main}"
IDEA_CODE="$MEDSAM_REPO/idea1_hard_full_medsam_ft"
AUDIT="${AUDIT:-/storage/baiyuting/data/out_data_idea1/baseline_reference/baseline_contract_audit.json}"

python - "$AUDIT" <<'PY'
import json, sys
p=sys.argv[1]
x=json.load(open(p, encoding="utf-8"))
assert x.get("passed") is True, p
print("[PASS] baseline audit")
PY

conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$MEDSAM_REPO:${PYTHONPATH:-}" \
  python -m py_compile \
    "$IDEA_CODE"/00_audit_baseline_contract.py \
    "$IDEA_CODE"/01_init_idea1_workspace.py \
    "$IDEA_CODE"/02_select_round0_random.py \
    "$IDEA_CODE"/03_build_full_finetune_pairs.py \
    "$IDEA_CODE"/04_finetune_medsam_mask_decoder.py \
    "$IDEA_CODE"/05_score_remaining_box_pool.py \
    "$IDEA_CODE"/06_select_next_hard_samples.py \
    "$IDEA_CODE"/07_generate_method_tri_pseudo.py \
    "$IDEA_CODE"/08_assemble_method_supervision.py \
    "$IDEA_CODE"/09_build_student_view.py \
    "$IDEA_CODE"/10_validate_method_data.py \
    "$IDEA_CODE"/11_finalize_method.py \
    "$IDEA_CODE"/run_dataset_full.py \
    "$IDEA_CODE"/ablation/build_hard20_fullonly.py \
    "$IDEA_CODE"/ablation/compare_tg3k_h50.py

for s in \
  "$IDEA_CODE/scripts/run_dataset_full.sh" \
  "$IDEA_CODE/scripts/run_tg3k_h50_ablation.sh" \
  "$IDEA_CODE/scripts/run_all_4gpu.sh" \
  "$IDEA_CODE/scripts/cleanup_finalize.sh" \
  "$IDEA_CODE/scripts/verify_final_repo.sh"
do
  bash -n "$s"
done

conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$MEDSAM_REPO:${PYTHONPATH:-}" \
  python - <<'PY'
import torch, segment_anything
print("[PASS] medsam env", torch.__version__, segment_anything.__file__)
PY

conda run -n swin_umamba --no-capture-output \
  env PYTHONPATH="$SWIN_REPO:$SWIN_REPO/swin_umamba:${PYTHONPATH:-}" \
  python - <<'PY'
import torch, einops
from nnunetv2.nets.SwinUMambaD import SwinUMambaD
print("[PASS] swin env", torch.__version__, einops.__version__)
PY


conda run -n medsam310 --no-capture-output \
  env PYTHONPATH="$IDEA_CODE:$MEDSAM_REPO:${PYTHONPATH:-}" \
  python -u "$IDEA_CODE/ablation/build_hard20_fullonly.py" \
  --help >/dev/null

echo "[PASS] ablation runtime import"

echo "[PASS] final repository preflight complete"
