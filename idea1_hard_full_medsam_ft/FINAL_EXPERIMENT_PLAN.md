# Idea1 Final Experiment Plan

## Active method

`idea1_hard_full_medsam_ft`

- 2D: random 5 Full at round 0, then add 5 hard images per round, maximum 20.
- 3D: random 1 Full case, then add 1 hard case per round, budget from the frozen contract.
- MedSAM mask decoder only, Dice+BCE, 300 steps/2D round, 1000 steps/3D round.
- Final student: exact Full GT + final fine-tuned MedSAM tri-state labels for remaining Box samples.
- Student: frozen baseline hyperparameters, 50 epochs, native-space evaluation.

## Final ablation

TG3K `Hard20-FullOnly`:

- Same 20 Full GT as Idea1.
- Remaining 3206 Box samples keep frozen baseline pseudo labels.
- Full 50-epoch student training.
- Compare with the completed TG3K Idea1 A50.

EMA, minimal Full/Box weak loss and Dual-Seed are rejected screening branches and are archived.

## Four-GPU queues

- GPU0: TN3K → BTCV
- GPU1: TG3K H50 ablation → OTU-2D → Synapse
- GPU2: Kvasir-SEG → PH2 → ACDC
- GPU3: CVC-ClinicDB → DDTI → Prostate158

The scheduler waits until each GPU has no compute process before starting its queue.

## Main commands

```bash
bash scripts/cleanup_finalize.sh --dry-run
bash scripts/cleanup_finalize.sh --apply
bash scripts/verify_final_repo.sh

nohup bash scripts/run_all_4gpu.sh \
  /storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft \
  > /storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft/logs/orchestrator/launcher.log \
  2>&1 &
```

## Monitoring

```bash
tail -f <FORMAL_ROOT>/logs/orchestrator/launcher.log
tail -f <FORMAL_ROOT>/logs/orchestrator/gpu0.log
tail -f <FORMAL_ROOT>/logs/orchestrator/gpu1.log
tail -f <FORMAL_ROOT>/logs/orchestrator/gpu2.log
tail -f <FORMAL_ROOT>/logs/orchestrator/gpu3.log
watch -n 5 nvidia-smi
```

## Resume rule

Rerun the same launcher after a failure. Dataset locks prevent duplicates, and completed
workspace/teacher/label/student/evaluation stages are skipped.
