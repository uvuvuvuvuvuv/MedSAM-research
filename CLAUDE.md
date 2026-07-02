# MedSAM Research Repository Rules

## Repository Purpose
This repository contains medical segmentation research code based on MedSAM.
The stable baseline and experimental ideas must remain isolated.

## Git Branches
- `main` is the stable branch.
- Experimental work must use `exp/*` branches.
- Before editing, run:
  - `git branch --show-current`
  - `git status --short --branch`
- Do not edit experimental code while on `main`.
- Do not commit or push unless the user explicitly requests it.

## Scope Control
- Only modify files explicitly authorized by the user.
- Other files may be read for interface analysis but must not be edited.
- Do not rename scripts, checkpoints, directories, JSON keys, or CLI arguments without approval.
- Do not create fallback behavior for obsolete paths unless explicitly required.

## Frozen Baseline
- Frozen baseline data and outputs are read-only references.
- Experimental methods must never overwrite frozen baseline prompts, pseudo-labels, checkpoints, logs, or metrics.
- If a required experiment-specific file is missing, stop and report it.
- Never silently fall back to an old dataset, old prompt file, or root-level legacy path.

## Data Spaces
Keep the following spaces distinct:
- Native Space
- Teacher Space
- Student Space

All coordinate conversions must follow `geometry_meta.json`.
Do not infer resizing or padding rules from array shapes alone.

## Dataset Safety
- Training logic may only use the train split.
- Test data must not participate in training, template construction, threshold selection, hard-sample selection, or pseudo-label generation.
- 3D datasets must preserve case-level consistency.

## Output Isolation
Each experimental method must use a method-specific output root.
Generated data, weights, logs, pseudo-labels, and metrics must not be written into source directories.

## Validation
After Python edits:
1. Run `python -m py_compile <modified-file>`.
2. Run `git diff --check`.
3. Run `git diff --stat`.
4. Report every modified file.
5. Do not run formal training without approval.

## Destructive Actions
Never run:
- `git reset --hard`
- `git clean -fd`
- `git push --force`
- `rm -rf` on datasets, checkpoints, logs, or experiment outputs

## Local Machine Contract
- 如果仓库根目录存在 `CLAUDE.local.md`，在解析数据、prompt、checkpoint、日志或输出路径前必须先读取它。
- `CLAUDE.local.md` 是当前服务器的私有路径合同，不得提交其内容。
- 路径缺失时必须报告准确路径，不得自行回退到旧目录或无关数据集。
