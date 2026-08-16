# Reproducibility

This document records the reproducibility contract for the released MedSAM
research code.

The public repository is reorganized for readability, while the exact formal
experiment state remains preserved through Git provenance and frozen outputs.

---

## 1. Frozen MedSAM Teacher

The original MedSAM checkpoint is immutable.

The Baseline always uses the original frozen Teacher.

Idea1 initializes Teacher adaptation from the same checkpoint and may create
new adapted checkpoints, but must never overwrite the original checkpoint.

The original MedSAM installation used as the frozen reference must likewise
remain unchanged.

---

## 2. Source-code Provenance

The exact MedSAM working-tree state used for the formal experiments is
preserved separately from the cleaned public branch.

The formal snapshot tag is:

```text
formal-v2-20260816
```

The tag points to the formal experiment snapshot and should not be moved during
public-repository cleanup.

The public cleanup branch may contain structural refactors that improve
readability without redefining the completed experiment outputs.

---

## 3. Historical Artifact Compatibility

Existing formal artifacts retain the method identifier:

```text
idea1_hard_full_medsam_ft
```

The cleaned public source directory is:

```text
methods/idea1/
```

These must not be conflated.

Changing the source-code path does not authorize renaming existing selections,
pseudo-labels, manifests, checkpoints, or evaluation outputs.

---

## 4. Random Seeds

The final 2D Round-0 active-selection seed is:

```text
2026
```

The matched Student training seed is:

```text
3407
```

The same Student seed must be used across Baseline, Idea1, and Upper in the
matched comparison.

---

## 5. Final 2D Annotation Budget

For a 2D dataset with `N` training images:

```text
round_quota = min(max(1, ceil(0.01 * N)), 5)
```

and

```text
max_full = min(
    max(1, ceil(0.05 * N)),
    5 * round_quota
)
```

At most five acquisition rounds are allowed:

```text
Round 0
Round 1
Round 2
Round 3
Round 4
```

No Round 5 acquisition is permitted.

A remaining image is a hard candidate when:

```text
image_min_iou < 0.5
```

The active-learning controller terminates using one of:

```text
CONVERGED
CONTINUE
BUDGET_EXHAUSTED
```

---

## 6. 2D Dataset-level Maximum Budgets

The final 2D protocol implies the following maximum budgets:

| Dataset | Per-round quota | Maximum Full |
|---|---:|---:|
| TG3K | 5 | 25 |
| TN3K | 5 | 25 |
| OTU_2D | 5 | 25 |
| Kvasir-SEG | 5 | 25 |
| CVC-ClinicDB | 5 | 25 |
| DDTI | 5 | 25 |
| PH2 | 2 | 8 |

Actual Full usage may be smaller because early convergence is allowed.

---

## 7. Frozen 3D Protocol

The 3D experiments were completed before the final strict 2D budget patch.

Their annotation unit is a case.

The frozen 3D budget is:

```text
B_3D = max(
    1,
    min(
        5,
        ceil(0.05 * N_train_cases)
    )
)
```

Round 0 acquires `B_3D` Full cases.

Later acquisition adds one case per round while difficult cases remain under
the frozen 3D protocol.

A difficult case satisfies:

```text
S_case < 0.5
```

The reported 3D experiments must not be regenerated solely to force the later
2D budget onto them.

2D and 3D results should therefore be summarized separately.

---

## 8. Shared Pseudo-label Arbitration

Matched Baseline and Idea1 pseudo-label generation uses:

```text
methods/common/multiclass_resolver.py
```

The same probability-aware arbitration logic must be used by both arms.

Changing the resolver for only one arm invalidates a direct matched
comparison.

---

## 9. Student Protocol

The final matched 2D Student configuration is:

| Setting | Value |
|---|---|
| epochs | 50 |
| batch size | 8 |
| num workers | 4 |
| learning rate | `1e-4` |
| weight decay | `0.05` |
| encoder freeze | first 10 epochs |
| gradient clipping | 12 |
| seed | 3407 |
| AMP | enabled |
| deep supervision | enabled |
| deep-supervision outputs | 3 |
| crop policy | `full_image` |
| foreground sampling probability | `0.0` |
| minimum valid ratio | `0.0` |

The Student encoder is frozen during epochs 1-10 and is unfrozen from epoch 11.

The same dataset-specific MIM-adapted encoder must be used for Baseline,
Idea1, and Upper.

MIM initialization is not an experimental variable in the matched comparison.

---

## 10. Supervision Arms

The matched comparison is:

```text
Baseline
    original MedSAM
    + Box pseudo-label supervision

Idea1
    adapted MedSAM
    + GT for acquired Full samples
    + pseudo-labels for remaining Box samples

Upper
    GT for all training samples
```

The Student architecture and optimization must otherwise remain identical.

---

## 11. Evaluation

### 11.1 2D

Final 2D Student evaluation is performed in native image geometry.

Primary metrics are:

```text
Dice
IoU
```

MAE is additionally used where supported.

Final evaluation must not use debug resizing.

### 11.2 3D

The frozen 3D evaluation reports:

```text
DSC
HD95
ASSD
```

Predictions must be reconstructed in the correct native geometry before final
measurement.

### 11.3 Reporting

Do not combine all seven 2D datasets and four 3D datasets into one primary
11-dataset macro average.

Their annotation units and frozen active-selection protocols differ.

---

## 12. Pure Teacher Pseudo-label Audit

Pure Teacher pseudo-label quality must compare:

```text
Baseline Box pseudo-labels
vs
Idea1 final Box pseudo-labels
```

on the exact same final Box cohort.

Samples that became Full must be excluded from this comparison.

The assembled Idea1 training supervision is not a pure Teacher metric because
its Full subset contains GT.

---

## 13. Structural Regression Tests

From the repository root, run:

```bash
PYTHONPATH="$PWD/methods/idea1:$PWD" python -m unittest discover   -s methods/idea1/tests   -p 'test_*.py'   -v
```

The current regression suite covers:

```text
budget computation
deterministic sampling
geometry round-trip behavior
Round-0 selection
hard-sample acquisition
hybrid Full/Box supervision assembly
Student-view construction
method-data validation
```

A successful run should end with:

```text
Ran 6 tests
OK
```

---

## 14. Syntax and Repository Checks

After source-code refactors, run:

```bash
python -m py_compile   generate_pseudo_labels.py   methods/common/multiclass_resolver.py   methods/idea1/*.py
```

Then:

```bash
git diff --check
```

and:

```bash
git status --short
```

Before a release commit, the working tree should be clean and regression tests
must pass.

---

## 15. Frozen-output Rules

Do not overwrite:

```text
the original MedSAM checkpoint
formal Teacher checkpoints
formal selection metadata
formal pseudo-label outputs
formal Student outputs
final evaluation summaries
formal provenance bundles
```

Repository cleanup must operate on the clean development/public branch, not on
the frozen formal output directories.

---

## 16. Public-release Checklist

Before creating the public release tag:

1. run the full Idea1 regression test suite;
2. run Python syntax checks;
3. run `git diff --check`;
4. verify there are no stale imports from the old Idea1 package path;
5. verify public documentation contains no obsolete internal run commands;
6. verify Baseline and Idea1 remain clearly separated;
7. verify historical method IDs are preserved only where required;
8. verify the working tree is clean.

After these checks, create an immutable paper-release tag, for example:

```text
v1.0-paper
```

The formal experiment tag and the cleaned public release tag serve different
purposes and should both be retained.
