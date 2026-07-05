# idea1_iterclean_bank_adaptshape

Iterative sampling pipeline for MedSAM fine-tuning with Feature Bank and Adaptive Multi-Shape templates.

## Pipeline Stages (per round)

| Stage | Script | Description |
|-------|--------|-------------|
| Split | `01_build_full_box_split.py` | Random (round 0) or external-selection (round 1+) Full/Box split |
| Template | `02_build_support_template.py` | Feature bank extraction + adaptive shape clustering |
| Train | `03_train_medsam_sac.py` | Mask-decoder-only fine-tuning with EMA |
| Pseudo | `04_generate_pseudo_sac.py` | Tri-value pseudo-label generation with bank quality scores |
| Select | `05_select_hard_by_gt_iou.py` | GT-IoU-based hard sample selection for next round |

## Key Design Decisions

### Feature Bank (02)
- Frozen Image Encoder from MedSAM ViT-B checkpoint
- Instance-level GT matching: component_id priority, bbox-IoU fallback
- FG token: per-instance area coverage >= 0.90 (area-averaged, not bilinear)
- BG token: ring region around original tight bbox AND global non-background
  coverage <= 0.10. The global non-background mask includes ALL foreground
  classes, ALL instances, and ignore (255) pixels. Only gt == 0 is
  considered reliable background.
- The original tight bbox is NEVER modified — the background ring is
  purely an exterior mask used only for Feature Bank sampling, never
  for MedSAM box prompts, shape crops, pseudo-label spatial constraints,
  or GT instance matching.
- Default bg_ring_width_tokens = 1 feature-map token (0 disables).
- Shape crop resize uses nearest-neighbor (preserves hard 0/1 boundaries).
- L2-normalized features; ring-based background sampling
- 3D per-case balanced sampling

### Adaptive Multi-Shape Clustering (02)
- Spherical k-means on flattened 64x64 shape masks
- Automatic K selection via support-unit count constraints (min_support=2)

### Training (03)
- Mask decoder only; Image/Prompt encoders frozen
- EMA for stable inference; Dice + BCEWithLogitsLoss
- Full-mode records only

### Pseudo-Label Fusion (04)
- Score = alpha * P + beta * A + gamma * QF
- Top-K bank query with temperature scaling for QF
- Tri-value output: FG (>= tau_high), BG (<= tau_low), unknown (255)

### Hard Sample Selection (05)
- GT-IoU scoring; 2D per-slice, 3D per-case aggregation (worst20_mean)
- Test data never participates

## Shared Modules

- `pipeline_common.py` — path construction, atomic JSON, Git checks, stage markers
- `instance_utils.py` — instance-level GT matching (connected-components, bbox IoU)

## NPZ Support File Contract

Global keys: `class_ids`, `feature_dim`, `shape_size`, `method`, `round_tag`, `run_id`

Per-class keys (for each class_id `c`): `bank_fg_c{c}`, `bank_bg_c{c}`, `shape_templates_c{c}`, `shape_semantic_centers_c{c}`, `shape_cluster_instance_counts_c{c}`, `shape_cluster_support_counts_c{c}`

All arrays loadable with `allow_pickle=False`. Forbidden old keys: `proto_fg_c*`, `proto_bg_c*`, `shape_A_c*`, `shape_R_c*`.

## Directory Contract

- **Source**: `idea1_iterclean_bank_adaptshape/`
- **Frozen baseline** (read-only): `/storage/baiyuting/data/MedSAM-main/data/processed/`
- **Writable root**: `/storage/baiyuting/data/out_data_idea1/`
- **Processed outputs**: `<writable_root>/MedSAM-main/data/processed/<dataset>/fold_0/`
- **Training outputs**: `<writable_root>/MedSAM-main/work_dir/MedSAM_ft/`

## Usage

### Runner (recommended)
```bash
python run_iterative_sampling.py \
    --preset smoke_2d \
    --datasets cvc_clinicdb \
    --method idea1_iterclean_bank_adaptshape \
    --run_name my_run
```

### Individual stage
```bash
# Build support template
python 02_build_support_template.py \
    --processed_root <path> --checkpoint <path> \
    --datasets cvc_clinicdb --method <method> --round_tag r00_full5

# Train mask decoder
python 03_train_medsam_sac.py \
    --processed_root <path> --checkpoint <path> \
    --datasets cvc_clinicdb --method <method> --out_root <path>

# Generate pseudo-labels
python 04_generate_pseudo_sac.py \
    --processed_root <path> --base_checkpoint <path> --ema_checkpoint <path> \
    --datasets cvc_clinicdb --method <method> --round_tag r00_full5 \
    --alpha 0.5 --beta 0.3 --gamma 0.2 --tau_low 0.1 --tau_high 0.5

# Select hard samples
python 05_select_hard_by_gt_iou.py \
    --processed_root <path> --datasets cvc_clinicdb \
    --method <method> --round_tag r00_full5 --round_id 0 --select_count 5
```

### Validate round outputs
```bash
python 08_validate_round_outputs.py \
    --processed_root <path> --dataset cvc_clinicdb \
    --method <method> --round_tag r00_full5 --strict
```

## Testing
```bash
PYTHONPATH=/storage/baiyuting/data/MedSAM-main \
    python -m unittest discover -s idea1_iterclean_bank_adaptshape/tests -p "test_*.py" -v
```

## Schedule Presets

| Preset | Round counts | Use case |
|--------|-------------|----------|
| `smoke_2d` | [5, 7] | 2D quick smoke test |
| `formal_2d` | [5, 10, 15, 20] | 2D full run |
| `formal_3d` | [1, 2, 3, 4, 5] | 3D per-case run |
| `custom` | user-specified | Custom schedule via --schedule |
