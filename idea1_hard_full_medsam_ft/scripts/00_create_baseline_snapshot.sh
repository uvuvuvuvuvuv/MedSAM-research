#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <frozen_medsam_repo> <frozen_swin_repo> <idea_root>" >&2
  exit 2
fi

FROZEN_MEDSAM="$(cd "$1" && pwd)"
FROZEN_SWIN="$(cd "$2" && pwd)"
IDEA_ROOT="$3"
CODE_ROOT="$IDEA_ROOT/code"
SNAPSHOT="$CODE_ROOT/baseline_v1_snapshot"
WORKING="$CODE_ROOT/idea1_hard_full_medsam_ft"

if [[ -e "$SNAPSHOT" || -e "$WORKING" ]]; then
  echo "Refusing to overwrite existing snapshot/working copy:" >&2
  echo "  $SNAPSHOT" >&2
  echo "  $WORKING" >&2
  exit 3
fi

mkdir -p "$SNAPSHOT/MedSAM-main" "$SNAPSHOT/Swin-UMamba-main"

copy_medsam() {
  local src="$1" dst="$2"
  rsync -a \
    --exclude '.git/' \
    --exclude 'data/raw/' \
    --exclude 'data/organized/' \
    --exclude 'data/processed/' \
    --exclude 'data/progress/' \
    --exclude 'work_dir/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$src/" "$dst/"
}

copy_swin() {
  local src="$1" dst="$2"
  rsync -a \
    --exclude '.git/' \
    --exclude 'work_dir/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    "$src/" "$dst/"
}

copy_medsam "$FROZEN_MEDSAM" "$SNAPSHOT/MedSAM-main"
copy_swin "$FROZEN_SWIN" "$SNAPSHOT/Swin-UMamba-main"

MEDSAM_COMMIT="$(git -C "$FROZEN_MEDSAM" rev-parse HEAD 2>/dev/null || echo unknown)"
SWIN_COMMIT="$(git -C "$FROZEN_SWIN" rev-parse HEAD 2>/dev/null || echo unknown)"
MEDSAM_DIRTY="$(git -C "$FROZEN_MEDSAM" status --porcelain 2>/dev/null | wc -l || true)"
SWIN_DIRTY="$(git -C "$FROZEN_SWIN" status --porcelain 2>/dev/null | wc -l || true)"

cat > "$SNAPSHOT/baseline_commit.json" <<JSON
{
  "frozen_medsam_repo": "$FROZEN_MEDSAM",
  "frozen_swin_repo": "$FROZEN_SWIN",
  "medsam_commit": "$MEDSAM_COMMIT",
  "swin_commit": "$SWIN_COMMIT",
  "medsam_dirty_entries": $MEDSAM_DIRTY,
  "swin_dirty_entries": $SWIN_DIRTY
}
JSON

(
  cd "$SNAPSHOT"
  find MedSAM-main Swin-UMamba-main -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > baseline_files_sha256.txt
)

mkdir -p "$WORKING"
cp -a "$SNAPSHOT/MedSAM-main" "$WORKING/MedSAM-main"
cp -a "$SNAPSHOT/Swin-UMamba-main" "$WORKING/Swin-UMamba-main"
cp -a "$SNAPSHOT/baseline_commit.json" "$WORKING/parent_baseline_commit.json"
chmod -R u+w "$WORKING"
chmod -R a-w "$SNAPSHOT"

echo "[PASS] baseline snapshot: $SNAPSHOT"
echo "[PASS] Idea1 working copy: $WORKING"
echo "Install Idea1 only into: $WORKING/MedSAM-main"
