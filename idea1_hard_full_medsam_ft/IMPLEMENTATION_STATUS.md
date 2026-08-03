# Implementation status

## Implemented

- Frozen baseline contract audit.
- Independent Idea1 workspace creation.
- `tri_train_boxonly` read-only reference.
- 2D Round 0 random five-image selection.
- 3D Round 0 random one-case selection.
- 3D 5% annotation budget calculation.
- Full image/case to per-component MedSAM box-mask pairs.
- Mask-decoder-only fine-tuning with Dice + BCE.
- Cumulative round training from the previous checkpoint.
- Baseline-mask-based remaining Box scoring.
- 2D image macro IoU hard selection.
- 3D class-macro volume IoU, missed-class and worst-20% scoring.
- Probability NPZ and heatmap diagnostics.
- Isolated invocation of the frozen baseline pseudo generator.
- Independent `tri_box_final_<METHOD>` output.
- Independent `tri_train_<METHOD>` hybrid supervision.
- Student views that preserve the unchanged baseline loader interface.
- Data validation for counts, Full/Box partition, exact Full GT and Box `255` location.
- Student train/infer/eval shell templates.
- Code snapshot and writable working-copy script.
- Python syntax checks and synthetic tests.

## Validation completed in this environment

- `python -m py_compile` on all Python files.
- Six synthetic unit/integration tests:
  - 3D budget formula;
  - deterministic sampling;
  - Full/Box partition;
  - geometry mask round trip;
  - Round 0 selection and fine-tuning pair generation;
  - hard selection, hybrid supervision, student view and validation.

## Not executed here

The following require the user's server, datasets, MedSAM checkpoint and GPUs:

- Real MedSAM mask-decoder optimization.
- Real baseline pseudo generation for the 11 datasets.
- Full teacher iteration on real data.
- Swin-UMamba training/inference/evaluation.

Run the TG3K or CVC-ClinicDB smoke test first, then a 3D ACDC/Synapse smoke test, before the 11-dataset batch.
