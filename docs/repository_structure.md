# Repository Structure

This repository keeps the original MedSAM implementation separate from the
research methods built on top of it.

The public layout is organized around three method layers:

```text
methods/
├── baseline/
├── common/
└── idea1/
```

The separation is intentional:

- `baseline/` documents the frozen MedSAM-based Box-supervision baseline;
- `common/` contains logic shared by multiple experimental arms;
- `idea1/` contains the active-selection and Teacher-adaptation extension.

Historical experiment identifiers are preserved inside existing artifact
schemas even when the public source-code layout uses cleaner names.

---

## 1. Top-level Layout

The relevant public layout is:

```text
.
├── methods/
│   ├── baseline/
│   │   └── README.md
│   ├── common/
│   │   ├── __init__.py
│   │   └── multiclass_resolver_v2.py
│   └── idea1/
│       ├── configs/
│       ├── scripts/
│       ├── tests/
│       ├── README.md
│       └── implementation files
│
├── docs/
│   ├── protocol.md
│   ├── reproducibility.md
│   └── repository_structure.md
│
├── segment_anything/
├── generate_pseudo_labels.py
└── ...
```

The upstream MedSAM implementation remains at repository level.

Research-specific code is placed under `methods/` whenever possible.

---

## 2. Baseline Layer

The Baseline uses the original frozen MedSAM Teacher with Box prompts.

The canonical pseudo-label generator remains:

```text
generate_pseudo_labels.py
```

The Baseline does not:

```text
acquire Full annotations
fine-tune MedSAM
perform active sample selection
```

The directory:

```text
methods/baseline/
```

therefore remains intentionally thin.

Its purpose is to document the Baseline contract rather than duplicate the
shared MedSAM inference implementation.

Duplicating the complete pseudo-label generator inside `methods/baseline/`
would make matched Baseline/Idea1 maintenance harder and could introduce
unintended implementation differences.

---

## 3. Shared Method Logic

Code used by more than one experimental arm belongs under:

```text
methods/common/
```

The current shared probability-aware multiclass resolver is:

```text
methods/common/multiclass_resolver_v2.py
```

Matched Baseline and Idea1 experiments must use the same arbitration logic.

The resolver location is a source-code path and is independent of historical
pseudo-label or experiment identifiers.

---

## 4. Idea1 Layer

Idea1 is implemented under:

```text
methods/idea1/
```

Its responsibilities include:

```text
workspace initialization
Full-annotation budget computation
Round-0 acquisition
fine-tuning-pair construction
MedSAM mask-decoder adaptation
remaining-pool scoring
hard-sample selection
iterative Teacher adaptation
final Box pseudo-label generation
Full/Box supervision assembly
Student-view construction
data validation
pipeline orchestration
```

The implementation is an extension of the Baseline, not a second copy of the
entire MedSAM repository.

---

## 5. Historical Method Identifier

Existing formal artifacts use:

```text
idea1_hard_full_medsam_ft
```

as the historical method identifier.

The public source-code directory is:

```text
methods/idea1/
```

These names serve different purposes:

```text
methods/idea1/
    source-code organization

idea1_hard_full_medsam_ft
    experiment/artifact identifier
```

The historical identifier may appear in:

```text
selection metadata
round directories
pseudo-label directory names
validation reports
run manifests
formal result paths
```

It must not be globally replaced merely because the source-code directory was
renamed.

---

## 6. Dataset Workspace

A prepared dataset workspace is organized around a fold:

```text
<workspace>/<dataset>/fold_0/
├── native_npy/
├── teacher_npy/
├── student_npy/
├── prompts/
├── pseudo_teacher/
├── pseudo_student/
├── rounds/
└── meta/
```

### 6.1 `native_npy/`

Stores data in the native/original geometry used as the reference for final
evaluation and geometry reconstruction.

### 6.2 `teacher_npy/`

Stores Teacher-space images and masks used by MedSAM.

### 6.3 `student_npy/`

Stores Student-space images and masks used by the downstream segmentation
network.

### 6.4 `prompts/`

Stores prompt metadata used for Teacher inference.

### 6.5 `pseudo_teacher/`

Stores Teacher-space pseudo-label products when required.

### 6.6 `pseudo_student/`

Stores Student-space pseudo-labels and the final assembled Student
supervision.

### 6.7 `rounds/`

Stores active-learning state and per-round products.

A historical Idea1 run may use:

```text
rounds/
└── idea1_hard_full_medsam_ft/
    ├── round_00/
    ├── round_01/
    └── ...
```

### 6.8 `meta/`

Stores manifests, geometry metadata, budget metadata, validation reports, and
final method state.

---

## 7. Per-round Structure

A typical Idea1 round contains logically separate products such as:

```text
round_XX/
├── selection/
├── finetune_pairs/
├── checkpoints/
├── diagnosis/
└── metadata / logs
```

Not every round must contain every product.

For example, a terminal round may stop after diagnosis and selection state
without creating a later acquisition round.

The authoritative round state is the saved metadata rather than the mere
existence of a directory.

---

## 8. Pseudo-label Layout

The final Box-only pseudo-label set for Idea1 follows the historical naming
contract:

```text
pseudo_student/
└── tri_box_final_idea1_hard_full_medsam_ft/
```

The final assembled Student supervision follows:

```text
pseudo_student/
└── tri_train_idea1_hard_full_medsam_ft/
```

The assembled supervision contains:

```text
Full samples -> GT
Box samples  -> final adapted-Teacher pseudo-labels
```

These artifact names are retained for reproducibility.

---

## 9. Student-view Boundary

The MedSAM repository is responsible for:

```text
Teacher inference
Teacher adaptation
active sample selection
pseudo-label generation
supervision assembly
Student-view preparation
```

Student model training, inference, and evaluation belong to the separate
Swin-UMamba research repository.

The Student repository is responsible for:

```text
MIM adaptation
Student training
Student inference
2D evaluation
3D evaluation
```

Duplicated Student training scripts should not be maintained inside this
MedSAM repository.

---

## 10. Configuration

Idea1 configuration files live under:

```text
methods/idea1/configs/
```

Historical configuration values that define artifact identifiers must remain
compatible with existing formal outputs.

In particular, changing the public directory layout does not imply changing
the stored `method_id`.

---

## 11. Tests

Idea1 regression tests live under:

```text
methods/idea1/tests/
```

They cover the current active-learning contract, including:

```text
budget computation
deterministic sampling
geometry round trips
Round-0 acquisition
hard-sample selection
Full/Box supervision assembly
Student-view construction
validation
```

Tests should be run after structural refactors.

---

## 12. Documentation

Public documentation is intentionally consolidated into:

```text
docs/protocol.md
docs/reproducibility.md
docs/repository_structure.md
```

Method-specific entry information is kept in:

```text
methods/baseline/README.md
methods/idea1/README.md
```

Internal development notes, temporary status reports, old directory-design
documents, and frozen experiment-operation notes should remain recoverable
through Git history rather than being maintained as parallel public
documentation.

---

## 13. Naming Rules

Use clean names for public source-code organization.

Do not embed development-state terms such as:

```text
FINAL
LOCKED
new
old
temporary
```

in the stable public API unless they are part of a historical artifact that
must remain unchanged.

Protocol-version strings such as `v2` may remain where they are part of an
existing implementation or artifact contract, but they should not be used as
a substitute for clear source-code organization.

---

## 14. Extension Rules

When adding a new method:

1. reuse shared logic from `methods/common/` where appropriate;
2. avoid copying the entire Baseline implementation;
3. isolate method-specific code under its own method directory;
4. preserve frozen historical outputs;
5. use a new output namespace rather than overwriting an existing method;
6. add regression tests for new protocol logic;
7. document the experimental variable explicitly.

The intended design is:

```text
MedSAM core
    |
    +--> shared pseudo-label logic
    |       |
    |       +--> Baseline
    |       |
    |       +--> Idea1
    |
    +--> future methods
```

---

## 15. Repository-cleanup Principle

Public-code cleanup may change:

```text
source directories
documentation names
internal module organization
developer-facing entry points
```

but must not silently change:

```text
frozen checkpoints
formal experiment outputs
dataset splits
stored selection decisions
historical method IDs
evaluation results
```

This separation allows the repository to become easier to read without
invalidating the provenance of the completed experiments.
