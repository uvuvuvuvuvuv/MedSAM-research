#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/MedSAM-research" >&2
  exit 2
fi

REPO_ROOT="$(cd "$1" && pwd)"
SRC_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="$REPO_ROOT/idea1_hard_full_medsam_ft"

if [[ -e "$TARGET" ]]; then
  echo "Refusing to overwrite existing directory: $TARGET" >&2
  exit 3
fi

cp -a "$SRC_ROOT" "$TARGET"
find "$TARGET" -type f -name '*.py' -print0 | xargs -0 python -m py_compile
chmod +x "$TARGET"/*.sh "$TARGET"/scripts/*.sh

echo "Installed: $TARGET"
echo "Frozen baseline root files were not modified."
