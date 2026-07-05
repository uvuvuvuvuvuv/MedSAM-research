# Experiment Card: idea1_iterclean_bank_adaptshape

## Metadata

- **Method ID**: `idea1_iterclean_bank_adaptshape`
- **Created**: 2026-07-05
- **Status**: Development / Validation
- **Base Model**: MedSAM ViT-B (`medsam_vit_b.pth`)

## Research Question

Does iterative hard-sample selection with a Feature Bank and Adaptive Multi-Shape templates improve MedSAM fine-tuning for medical image segmentation compared to single-round training?

## Architecture

```
Round 0: Random Split → [Feature Bank + Shape Templates] → Train Decoder → Pseudo Labels → Hard Selection
Round 1: External Split from Selection → [Bank + Shapes]  → Train Decoder → Pseudo Labels → Hard Selection
...
```

### Key Components

1. **Feature Bank**: Coverage-based FG/BG feature tokens from frozen Image Encoder, L2-normalized, with ring-based background sampling and 3D per-case balancing.

2. **Adaptive Multi-Shape Templates**: Spherical k-means clustering of instance shape masks with automatic K selection. Used for shape affinity scoring in pseudo-label generation.

3. **EMA Training**: Mask-decoder-only fine-tuning with Exponential Moving Average for stable inference.

4. **Tri-Value Pseudo-Labels**: Score = alpha * P + beta * A + gamma * QF, where:
   - P: prediction probability from EMA model
   - A: shape affinity from bank template matching
   - QF: quality factor from top-K bank query with temperature scaling

5. **GT-IoU Hard Selection**: Selects hardest samples by IoU against ground truth. 2D: per-slice. 3D: per-case aggregation (worst20_mean).

## Parameters

### Feature Bank & Shape Templates (02)
| Parameter | Default | Description |
|-----------|---------|-------------|
| fg_coverage_threshold | 0.90 | Min coverage for FG token |
| max_fg_per_instance | 128 | Max FG tokens per instance |
| max_bg_per_instance | 128 | Max BG tokens per instance |
| max_fg_per_class | 4096 | Max FG tokens per class |
| max_bg_per_class | 4096 | Max BG tokens per class |
| bg_ring_width_tokens | 1 | Background ring width in feature-map tokens (0=disabled) |
| max_global_non_bg_coverage | 0.10 | Max allowed non-background area fraction in a reliable BG token |
| shape_size | 64 | Shape template resolution |
| kmax_shape | 5 | Max clusters per class |
| cluster_max_iter | 25 | Spherical k-means iterations |

### Training (03)
| Parameter | Default | Description |
|-----------|---------|-------------|
| epochs | 1 | Training epochs (full pass through training data) |
| max_steps | 0 | Max training steps (0 = unlimited, train full epoch). Smoke uses 5 (Round 0) or 2 (Round 1+) |
| lr | 1e-5 | Learning rate |
| weight_decay | 0.01 | Weight decay |
| ema_decay | 0.99 | EMA decay rate |
| max_grad_norm | 1.0 | Gradient clipping |

### Pseudo-Label Fusion (04)
| Parameter | Default | Description |
|-----------|---------|-------------|
| alpha | 0.5 | Prediction weight |
| beta | 0.3 | Shape affinity weight |
| gamma | 0.2 | Quality factor weight |
| tau_low | 0.1 | Background threshold |
| tau_high | 0.5 | Foreground threshold |
| temperature | 0.1 | Bank query temperature |
| top_k | 5 | Top-K bank neighbors |

## Schedule Presets

| Preset | 2D Rounds | 3D Rounds |
|--------|----------|----------|
| smoke_2d | [5, 7] Full images | — |
| formal_2d | [5, 10, 15, 20] Full images | — |
| formal_3d | — | [1, 2, 3, 4, 5] Cases |
| custom | User-specified | User-specified |

## Data Contract

### Inputs (per dataset, per fold)
- `meta/manifest.json` — dataset manifest
- `meta/geometry_meta.json` — coordinate transforms
- `prompts/prompts_train.json` — training prompts (IDEA1 writable root)
- `meta/label_meta.json` — class label mapping

### Outputs (per round)
- `meta/full_box_split_<run_id>.json` — Full/Box split
- `meta/support_bank_adaptshape_<run_id>.npz` — Feature bank + shape templates
- `meta/support_bank_adaptshape_stats_<run_id>.json` — Template stats
- `meta/hard_selection_<run_id>.json` — Hard selection payload
- `meta/hard_selection_summary_<run_id>.json` — Selection summary
- Pseudo-label directories: `pseudo_teacher_<run_id>/`, `pseudo_student_<run_id>/`

### Outputs (per training round)
- `work_dir/MedSAM_ft/<method>/<dataset>/<fold>/<round_tag>/` — Checkpoints, logs, summary

## Validation

### Stage Validators
Each stage output is validated by the runner:
- **Split**: Full/Box record counts, no test leakage, unit count matches schedule
- **Template**: NPZ keys (bank_fg_c*, bank_bg_c*, shape_templates_c*), no old keys, allow_pickle=False
- **Train**: Checkpoint non-empty, summary method/global_step/trainable correct
- **Pseudo**: Teacher/student counts match expected, audit output_complete=true
- **Select**: Summary counts consistent, test_samples_used=0

### Round Output Validator
```bash
python 08_validate_round_outputs.py --strict ...
```
Checks all stage outputs, NPZ contract, JSON validity, and stage_success markers.

### Test Suite
```bash
python -m unittest discover -s idea1_iterclean_bank_adaptshape/tests -p "test_*.py" -v
```
574 tests covering pipeline_common, instance_utils, runner utilities, and all pipeline stages.

## Safety Rules

- Frozen baseline (`/storage/baiyuting/data/MedSAM-main/data/processed/`) is read-only
- All outputs written to IDEA1 writable root (`/storage/baiyuting/data/out_data_idea1/`)
- Test data never participates in training, template construction, or hard sample selection
- 3D datasets preserve case-level consistency
- NPZ files must load with `allow_pickle=False`

## Known Limitations

- Split file must be generated by the runner before individual stage scripts can run
- Training requires GPU with sufficient VRAM for MedSAM ViT-B
- First round uses random sampling; selection quality improves with subsequent rounds
- 3D per-case balancing may discard tokens from very large cases
- ACDC中目标结构彼此邻近，且feature token具有有限空间分辨率，因此紧密框外一层ring中的部分token会同时覆盖确定背景与其他前景结构。全局非背景覆盖率超过0.10的token会被严格过滤，因而3D的bg_ring_valid_ratio低于2D。这是全局背景纯度约束的预期结果。
