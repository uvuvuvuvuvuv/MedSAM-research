# SAC-MedSAM Final — Clean Mainline

This directory contains the cleaned **Default SAC** mainline. Experimental branches that are not enabled in the reported default run are intentionally excluded from the core scripts.

## Main scripts

1. `01_build_full_box_split.py`
   - Default: `full_ratio=0.10`, `seed=2026`.
   - 2D datasets are split by image.
   - ACDC, Prostate158, BTCV and Synapse are split by case.

2. `02_build_support_template.py`
   - Reads **full samples only**.
   - Extracts normalized MedSAM image-encoder features.
   - Builds `k_fg=3` foreground prototypes and `k_bg=5` background prototypes per class.
   - Builds only `shape_A`; the redundant `shape_R` field has been removed.

3. `03_train_medsam_sac.py`
   - Freezes the image encoder and prompt encoder.
   - Trains the mask decoder only.
   - Full loss: `Dice + BCE`.
   - Box loss: `out + seed + WAC + prototype + TV smooth`.
   - Box samples do not load GT at code level.
   - Disabled FAS logic and all related arguments/log columns have been removed.

4. `04_generate_pseudo_sac.py`
   - Full samples load GT and retain exact supervision.
   - Box samples do not load `teacher_gt`, `native_gt` or `student_gt`.
   - Generates both teacher-space and student-space tri-state labels.
   - Uses `q_threshold` from the command line instead of a hard-coded `0.58`.
   - Geometry mapping supports offsets and uses nearest-neighbor interpolation for labels.
   - Generation statistics do not include GT-based evaluation metrics. The legacy filename `pseudo_quality_stats_<method>.csv` is retained so the existing visualization script remains compatible.

5. `05_visualize_pseudo_sac.py`
   - Keep the currently validated visualization script in the project directory.
   - It is not replaced by this package.

6. `06_compare_pseudo_quality.py`
   - Supports binary and multi-class datasets in one script.
   - Computes strict micro and macro metrics.
   - Uses the correct metric direction when counting SAC improvements.
   - Exports sample-level, paired-delta and per-class CSV files.

## Removed from the formal mainline

- FAS parameters, loss, gates and logs (`lambda_fas=0` in the previous default run).
- `shape_R`, because it is exactly recoverable from `shape_A` and was not consumed.
- GT loading for box training and box pseudo-label generation.
- Full-subset metrics inside the generation script.
- Duplicate teacher-to-student remap in the standard path.

`07_remap_sac_teacher_to_student.py` remains useful only as a recovery utility when teacher pseudo-labels already exist and student pseudo-labels need to be regenerated without running MedSAM again.

## Installation

```bash
bash install_clean_sac_scripts.sh \
  /storage/baiyuting/data/MedSAM-main/idea1_sac_medsam_final
```

The installer creates timestamped backups, archives experimental/backup scripts, installs the cleaned files and runs `py_compile`.

## Important reproducibility note

The cleaned scripts preserve the previous Default SAC numerical logic for the default configuration, except for code-audit corrections that were previously ineffective or redundant:

- `q_threshold` now actually controls the `Q` threshold; its default remains `0.58`.
- Box GT is no longer loaded, but it was not used in the previous box loss.
- FAS was removed because its default weight was `0.0`.
- `shape_R` was removed because it was unused.

Changing thresholds, loss weights, geometry offsets, split records or checkpoints will change generated results.
