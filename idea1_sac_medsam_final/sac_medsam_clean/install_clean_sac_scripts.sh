#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-/storage/baiyuting/data/MedSAM-main/idea1_sac_medsam_final}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="$TARGET_DIR/archive"
BACKUP_DIR="$ARCHIVE_DIR/backup/$STAMP"
EXPERIMENTAL_DIR="$ARCHIVE_DIR/experimental"
UTILITY_DIR="$ARCHIVE_DIR/utilities"
LEGACY_BACKUP_DIR="$ARCHIVE_DIR/legacy_backups"

mkdir -p "$TARGET_DIR" "$BACKUP_DIR" "$EXPERIMENTAL_DIR" "$UTILITY_DIR" "$LEGACY_BACKUP_DIR"

echo "[1/5] Backing up current Python/README files to: $BACKUP_DIR"
for file in "$TARGET_DIR"/*.py "$TARGET_DIR"/README.md; do
  [[ -e "$file" ]] || continue
  cp -a "$file" "$BACKUP_DIR/"
done

echo "[2/5] Archiving experimental and recovery scripts"
for name in \
  03_train_medsam_sac_v4_tawac.py \
  03_train_medsam_sac_fas.py \
  04a_calibrate_thresholds_sac.py \
  04b_calibrate_clean_sac.py; do
  if [[ -e "$TARGET_DIR/$name" ]]; then
    mv "$TARGET_DIR/$name" "$EXPERIMENTAL_DIR/${name%.py}_${STAMP}.py"
  fi
done

if [[ -e "$TARGET_DIR/07_remap_sac_teacher_to_student.py" ]]; then
  mv "$TARGET_DIR/07_remap_sac_teacher_to_student.py" \
    "$UTILITY_DIR/07_remap_sac_teacher_to_student_${STAMP}.py"
fi

for file in "$TARGET_DIR"/*.bak*; do
  [[ -e "$file" ]] || continue
  mv "$file" "$LEGACY_BACKUP_DIR/$(basename "$file").$STAMP"
done

echo "[3/5] Installing cleaned mainline scripts"
for name in \
  01_build_full_box_split.py \
  02_build_support_template.py \
  03_train_medsam_sac.py \
  04_generate_pseudo_sac.py \
  06_compare_pseudo_quality.py \
  README.md; do
  install -m 0644 "$SCRIPT_DIR/$name" "$TARGET_DIR/$name"
done

if [[ ! -e "$TARGET_DIR/05_visualize_pseudo_sac.py" ]]; then
  echo "[WARN] 05_visualize_pseudo_sac.py was not found; keep/install your validated visualization script separately."
fi

echo "[4/5] Syntax checking"
python -m py_compile \
  "$TARGET_DIR/01_build_full_box_split.py" \
  "$TARGET_DIR/02_build_support_template.py" \
  "$TARGET_DIR/03_train_medsam_sac.py" \
  "$TARGET_DIR/04_generate_pseudo_sac.py" \
  "$TARGET_DIR/06_compare_pseudo_quality.py"

echo "[5/5] Final mainline files"
find "$TARGET_DIR" -maxdepth 1 -type f -printf '%f\n' | sort

echo "[OK] Clean SAC-MedSAM mainline installed."
echo "     backup: $BACKUP_DIR"
echo "     target: $TARGET_DIR"
