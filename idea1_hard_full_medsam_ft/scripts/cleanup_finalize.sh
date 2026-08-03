#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
FORMAL_ROOT="${2:-/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft}"
MEDSAM_REPO="${MEDSAM_REPO:-/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft/MedSAM-main}"
IDEA_CODE="$MEDSAM_REPO/idea1_hard_full_medsam_ft"
STAMP="20260801"
CODE_ARCHIVE="$IDEA_CODE/_archive/rejected_screening_$STAMP"
RESULT_ARCHIVE="$FORMAL_ROOT/summaries/tg3k/rejected_screening_$STAMP"

mapfile -t CODE_FILES < <(
  find "$IDEA_CODE" -maxdepth 3 -type f \
    ! -path "$IDEA_CODE/_archive/*" \
    \( -name '12_*' -o -name '13_*' -o -name '14_*' -o -name '15_*' -o -name '16_*' -o -name '17_*' \
       -o -iname '*screen*' -o -iname '*dualseed*' -o -iname '*fullbox*' \
       -o -name '*.rej' -o -name '*.bak' \) \
    ! -path "$IDEA_CODE/ablation/*" \
    ! -path "$IDEA_CODE/scripts/run_all_4gpu.sh" \
    ! -path "$IDEA_CODE/scripts/run_dataset_full.sh" \
    ! -path "$IDEA_CODE/scripts/run_tg3k_h50_ablation.sh" \
    ! -path "$IDEA_CODE/scripts/cleanup_finalize.sh" \
    ! -path "$IDEA_CODE/scripts/verify_final_repo.sh" \
    | sort
)

RESULT_DIRS=(
  "$FORMAL_ROOT/screen_teacher_tg3k"
  "$FORMAL_ROOT/screen_teacher_tg3k_dualseed"
  "$FORMAL_ROOT/Swin-UMamba-main/work_dir/student_screen15_tg3k"
)

echo "Mode: $MODE"
echo "Code files to archive:"
printf '  %s\n' "${CODE_FILES[@]:-}"
echo "Screening result directories to compact:"
printf '  %s\n' "${RESULT_DIRS[@]}"

[[ "$MODE" == "--apply" ]] || { echo "[DRY RUN] rerun with --apply"; exit 0; }

mkdir -p "$CODE_ARCHIVE" "$RESULT_ARCHIVE"

for f in "${CODE_FILES[@]}"; do
  [[ -f "$f" ]] || continue
  rel="${f#$IDEA_CODE/}"
  mkdir -p "$CODE_ARCHIVE/$(dirname "$rel")"
  mv "$f" "$CODE_ARCHIVE/$rel"
done

for d in "${RESULT_DIRS[@]}"; do
  [[ -d "$d" ]] || continue
  name="$(basename "$d")"
  dst="$RESULT_ARCHIVE/$name"
  mkdir -p "$dst"
  rsync -a \
    --include='*/' \
    --include='*.json' \
    --include='*.csv' \
    --include='*.txt' \
    --include='*.log' \
    --exclude='*' \
    "$d/" "$dst/"
  rm -rf "$d"
done

find "$IDEA_CODE" -type d -name '__pycache__' -prune -exec rm -rf {} +
echo "[PASS] obsolete screening code archived; heavy screening outputs removed"
