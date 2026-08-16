# Idea1

Idea1 is the Teacher-adaptation extension of the MedSAM Box-supervision
Baseline.

It uses a small Full-annotation budget to adapt the MedSAM mask decoder,
select difficult samples, regenerate pseudo-labels for the remaining
Box-supervised pool, and assemble the final Student supervision.

For the full experimental contract, see:

```text
docs/protocol.md
```

For repository layout and reproducibility rules, see:

```text
docs/repository_structure.md
docs/reproducibility.md
```

---

## 1. Method Overview

```text
Frozen MedSAM
     |
     v
Round-0 Full acquisition
     |
     v
Mask-decoder adaptation
     |
     v
Score remaining Box pool
     |
     v
Hard-sample selection
     |
     v
Iterative Teacher adaptation
     |
     v
Final Box pseudo-label generation
     |
     +----------------------+
     |                      |
     v                      v
  Full GT            Box pseudo-labels
     |                      |
     +----------+-----------+
                |
                v
        Student supervision
```

Idea1 does not replace the Baseline implementation.

It extends the same MedSAM pseudo-label pipeline with Full-sample acquisition
and Teacher adaptation.

---

## 2. Shared Logic

Matched Baseline and Idea1 experiments share the probability-aware multiclass
resolver:

```text
methods/common/multiclass_resolver.py
```

The canonical MedSAM pseudo-label generator remains at repository level:

```text
generate_pseudo_labels.py
```

---

## 3. Current Source Layout

Idea1 source code lives under:

```text
methods/idea1/
```

The directory contains:

```text
configuration
active-selection stages
Teacher-adaptation stages
pseudo-label assembly
validation
pipeline orchestration
regression tests
```

The historical experiment identifier is:

```text
idea1_hard_full_medsam_ft
```

This identifier is intentionally retained in existing artifact schemas.

It is not the current source-code directory name.

---

## 4. Final 2D Budget

For `N` training images:

```text
q = min(max(1, ceil(0.01 * N)), 5)
```

and:

```text
max_full = min(
    max(1, ceil(0.05 * N)),
    5 * q
)
```

At most five acquisition rounds are allowed:

```text
Round 0 ... Round 4
```

Round 0 uses:

```text
seed = 2026
```

For later rounds, an image is a hard candidate when:

```text
image_min_iou < 0.5
```

No Round 5 acquisition is permitted.

---

## 5. Final Supervision

For acquired Full samples:

```text
Student supervision = GT
```

For samples that remain Box-supervised:

```text
Student supervision = final adapted-Teacher pseudo-label
```

The final assembled supervision therefore contains:

```text
Full GT
+
Box pseudo-labels
```

Pure Teacher pseudo-label quality must be audited on the Box-only cohort
rather than on this assembled supervision.

---

## 6. 3D Compatibility

The reported 3D experiments use the earlier frozen case-level protocol.

They must not be silently converted to the later strict 2D budget.

See `docs/protocol.md` for the modality-specific protocol definitions.

---

## 7. Tests

From the repository root:

```bash
PYTHONPATH="$PWD/methods/idea1:$PWD" python -m unittest discover   -s methods/idea1/tests   -p 'test_*.py'   -v
```

Expected result:

```text
Ran 6 tests
OK
```

These tests cover the active-learning budget, deterministic sampling,
hard-sample acquisition, hybrid supervision assembly, Student-view
construction, and validation.

---

## 8. Development Rule

When refactoring this directory:

```text
do not overwrite frozen experiment outputs
do not rename historical artifact IDs globally
do not change the frozen 3D protocol
do not duplicate the Student training pipeline here
```

Student training, inference, and evaluation belong to the separate
Swin-UMamba research repository.
