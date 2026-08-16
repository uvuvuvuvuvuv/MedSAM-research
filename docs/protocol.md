# Experimental Protocol

This document defines the experimental protocol for the MedSAM-based
**Baseline**, **Idea1**, and **Upper** systems used in this project.

The central goal is to ensure that the compared systems differ primarily in
their supervision strategy, while the Student architecture, initialization,
optimization, data split, and evaluation protocol remain matched whenever a
direct comparison is made.

---

## 1. Experimental Systems

Three systems are considered.

### 1.1 Baseline

The Baseline uses the original frozen MedSAM model as the Teacher.

For each Box-supervised training sample:

1. the image and Box prompt are provided to MedSAM;
2. MedSAM produces candidate masks and their associated probabilities;
3. the shared probability-aware multiclass resolver converts the predictions
   into a tri-state pseudo-label;
4. the generated pseudo-label is used as supervision for the downstream
   Student.

The Baseline does **not**:

- acquire Full annotations;
- perform active learning;
- fine-tune the MedSAM Teacher.

Conceptually:

```text
Image + Box Prompt
        |
        v
  Frozen MedSAM
        |
        v
Mask Candidates
+ Probabilities
        |
        v
Shared Multiclass
Resolver
        |
        v
Tri-state Pseudo-label
        |
        v
      Student
```

### 1.2 Idea1

Idea1 extends the Baseline with a small Full-annotation budget.

The additional operations are:

1. acquire a small subset of Full annotations;
2. construct fine-tuning pairs from the selected Full samples;
3. fine-tune the MedSAM mask decoder;
4. evaluate the remaining Box-supervised pool;
5. identify difficult samples;
6. acquire additional Full samples when annotation budget remains;
7. iteratively adapt the Teacher;
8. regenerate pseudo-labels for the remaining Box samples;
9. combine Full GT and regenerated Box pseudo-labels into the final Student
   supervision.

Conceptually:

```text
Frozen MedSAM
     |
     v
Round-0 Full Acquisition
     |
     v
Mask Decoder Adaptation
     |
     v
Score Remaining Box Pool
     |
     v
Hard-Sample Selection
     |
     v
Additional Full Acquisition
     |
     v
Iterative Teacher Adaptation
     |
     v
Final Box Pseudo-label Generation
     |
     +----------------------+
     |                      |
     v                      v
  Full GT            Box Pseudo-labels
     |                      |
     +----------+-----------+
                |
                v
        Student Supervision
```

All pseudo-label generation and arbitration logic that is applicable to both
systems must remain shared between Baseline and Idea1.

### 1.3 Upper

Upper is the fully supervised reference system.

All training samples use Full ground-truth masks.

Upper does not depend on MedSAM pseudo-labels or Teacher adaptation.

Conceptually:

```text
Full GT
   |
   v
Student
```

Upper is used as a fully supervised reference ceiling rather than as a weakly
supervised method.

---

## 2. Tri-state Supervision

Baseline and Idea1 use tri-state supervision.

The label definition is:

```text
0       background
1...K   foreground class labels
255     ignore / uncertain
```

Pixels labeled `255` are treated as uncertain and are ignored by the
corresponding Student supervision loss.

Baseline and Idea1 use the same probability-aware multiclass arbitration
implementation:

```text
methods/common/multiclass_resolver.py
```

This shared resolver is part of the fairness contract.

Using different conflict-resolution logic for Baseline and Idea1 would
introduce an additional experimental variable and is therefore not allowed in
the matched comparison.

The selected candidate mask and its associated probability representation must
remain aligned throughout pseudo-label generation.

---

## 3. Spatial Coordinate Systems

The pipeline distinguishes three spatial coordinate systems.

### 3.1 Native Space

Native Space corresponds to the original dataset geometry.

It is the reference geometry for final evaluation.

Typical contents include:

```text
native_npy/
```

### 3.2 Teacher Space

Teacher Space is the geometry used by MedSAM.

Teacher-side images, prompts, masks, and pseudo-labels must remain consistent
within this space.

Typical contents include:

```text
teacher_npy/
pseudo_teacher/
```

### 3.3 Student Space

Student Space is the geometry used by the downstream Student network.

Student-side training images and supervision are represented in this space.

Typical contents include:

```text
student_npy/
pseudo_student/
```

### 3.4 Geometry Contract

All geometry transformations must be explicitly recorded.

Teacher-space predictions must not be directly compared with Native-space GT
without applying the corresponding geometric transformation.

Final Student evaluation must use the intended native/test geometry.

Debug-only resizing must not be used for final reported results.

---

## 4. Data-Split Contract

Training, validation, and test partitions are fixed before running the method.

The test split is immutable.

Test images and test GT must not participate in:

```text
Teacher fine-tuning
active sample acquisition
training pseudo-label generation
Student training
MIM adaptation
hyperparameter selection
```

Training GT may be used to simulate Full annotation acquisition and to measure
Teacher difficulty during the active-learning experiment.

This use of training GT is part of the controlled annotation-budget simulation
and must not be confused with test-time supervision.

---

## 5. Final 2D Active-Learning Protocol

The final 2D experiments use **images** as annotation units.

Let:

```text
N = number of training images
```

### 5.1 Per-round Full Annotation Quota

The Full acquisition quota per round is:

```text
q = min(max(1, ceil(0.01 * N)), 5)
```

This means:

- at least one Full image may be acquired per round;
- the nominal acquisition rate is approximately 1% of the training set;
- no more than five new Full images may be acquired in one round.

### 5.2 Maximum Cumulative Full Budget

The maximum cumulative Full budget is:

```text
max_full = min(
    max(1, ceil(0.05 * N)),
    5 * q
)
```

This simultaneously constrains the total Full annotation to approximately 5%
of the training set while also respecting the maximum of five acquisition
rounds.

### 5.3 Maximum Number of Acquisition Rounds

Only five acquisition rounds are permitted:

```text
Round 0
Round 1
Round 2
Round 3
Round 4
```

There is no Round 5 acquisition.

If difficult samples still remain after the available Round-4 budget is used,
the active-learning procedure terminates as `BUDGET_EXHAUSTED`.

### 5.4 Dataset-level Maximum 2D Budgets

The protocol-level maximum budgets for the seven 2D datasets are:

| Dataset | Per-round quota `q` | Maximum Full |
|---|---:|---:|
| TG3K | 5 | 25 |
| TN3K | 5 | 25 |
| OTU_2D | 5 | 25 |
| Kvasir-SEG | 5 | 25 |
| CVC-ClinicDB | 5 | 25 |
| DDTI | 5 | 25 |
| PH2 | 2 | 8 |

These are **maximum budgets**, not necessarily the number of Full annotations
actually used.

A dataset may terminate earlier if the remaining Box-supervised pool already
satisfies the convergence criterion.

---

## 6. Round 0

Round 0 does not use Teacher difficulty ranking.

The initial Full subset is selected using deterministic random sampling.

The fixed random seed is:

```text
seed = 2026
```

The number of Round-0 Full samples is:

```text
min(q, remaining annotation capacity)
```

The selected sample IDs must be recorded in the Round-0 selection metadata.

All remaining training samples stay in the Box-supervised pool.

---

## 7. Teacher Adaptation

Idea1 adapts MedSAM using the currently accumulated Full subset.

The adaptation target is the MedSAM **mask decoder**.

The original MedSAM checkpoint is treated as the immutable initialization
source and must never be overwritten by an adapted Teacher checkpoint.

For every selected Full sample, the fine-tuning pair contains:

```text
Teacher-space image
Box prompt
Full GT mask
```

The same geometry and prompt conventions used by the Baseline must be retained.

Teacher adaptation is therefore an extension of the Baseline rather than a
separate segmentation pipeline.

---

## 8. Remaining Box-pool Scoring

After Teacher adaptation, every remaining Box-supervised image is evaluated
with the current Teacher.

For each target instance, IoU is computed.

The final 2D protocol defines image difficulty using:

```text
image_min_iou
```

That is, the image-level score is determined by the worst target instance in
the image.

An image becomes a hard candidate when:

```text
image_min_iou < 0.5
```

Hard candidates are ranked in ascending order:

```text
lower IoU
   =
harder sample
```

This prioritizes images on which the current Teacher performs worst.

---

## 9. Round 1 and Later

For every later acquisition round:

1. evaluate all currently remaining Box samples;
2. identify samples satisfying `image_min_iou < 0.5`;
3. sort the hard candidates from lower IoU to higher IoU;
4. acquire at most `q` new Full samples;
5. respect the remaining cumulative Full budget;
6. add the newly acquired samples to the cumulative Full subset;
7. adapt the Teacher again;
8. repeat until a stopping condition is reached.

Selection is cumulative.

A sample that has already become Full must never return to the Box pool.

---

## 10. Active-Learning States

The controller uses three states.

### 10.1 `CONVERGED`

The remaining pool satisfies:

```text
all remaining image_min_iou >= 0.5
```

No additional Full annotation is required.

The active-learning procedure terminates early.

### 10.2 `CONTINUE`

At least one hard sample remains and additional Full annotation capacity is
available.

The controller proceeds to the next acquisition round.

### 10.3 `BUDGET_EXHAUSTED`

Hard samples remain, but additional acquisition is no longer permitted because:

```text
the cumulative Full budget has been reached
```

or

```text
the maximum acquisition-round budget has been reached
```

The current Teacher becomes the final Teacher.

No Round 5 is created.

---

## 11. Final Box Pseudo-label Generation

After Teacher adaptation terminates, the final Teacher regenerates
pseudo-labels for all samples that remain Box-supervised.

The final Box supervision follows the same shared pseudo-label generation
contract used for the matched Baseline comparison.

The Student-space final Box supervision follows the historical artifact schema:

```text
pseudo_student/
└── tri_box_final_<method_id>/
```

Teacher-space pseudo-labels are stored separately when required.

Only samples that remain Box-supervised belong to this final Box subset.

---

## 12. Full-GT and Box-Pseudo Assembly

The final Idea1 Student supervision contains two mutually exclusive groups.

### 12.1 Full Subset

For acquired Full samples:

```text
Student supervision = Full GT
```

The Teacher pseudo-label is not used as the final Student supervision for these
samples.

### 12.2 Box Subset

For samples that were never acquired as Full:

```text
Student supervision = final adapted-Teacher pseudo-label
```

### 12.3 Unified Student Supervision

The two subsets are assembled into:

```text
pseudo_student/
└── tri_train_<method_id>/
```

For the historical formal experiment artifacts:

```text
method_id = idea1_hard_full_medsam_ft
```

The released public source-code directory is:

```text
methods/idea1/
```

These two names represent different concepts:

```text
methods/idea1/
    = source-code organization

idea1_hard_full_medsam_ft
    = historical experiment/artifact identifier
```

The historical method identifier must not be globally replaced inside existing
artifact schemas.

---

## 13. Baseline / Idea1 / Upper Fairness Contract

The direct comparison between Baseline, Idea1, and Upper must keep the Student
pipeline matched.

The intended experimental variable is the supervision source.

### Baseline

```text
Original frozen MedSAM
+
Box pseudo-label supervision
```

### Idea1

```text
Adapted MedSAM
+
Full GT for acquired samples
+
Box pseudo-labels for remaining samples
```

### Upper

```text
Full GT for all training samples
```

Student architecture, optimization, initialization, evaluation geometry, and
evaluation implementation must remain matched unless explicitly studied as
another experiment.

---

## 14. Matched 2D Student Protocol

The final matched 2D comparison uses the following Student settings:

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

The Student encoder is frozen during epochs 1-10 and becomes trainable from
epoch 11.

---

## 15. Shared MIM Initialization

For each dataset, Baseline, Idea1, and Upper must use the same corresponding
MIM-adapted Student encoder.

Therefore:

```text
Baseline
Idea1
Upper
```

share the same dataset-specific MIM initialization.

MIM adaptation is not an experimental variable in the matched three-arm
comparison.

---

## 16. 2D Dataset Group

The final 2D protocol is applied to:

```text
tg3k
tn3k
otu_2d
kvasirseg
cvc_clinicdb
ddti
ph2
```

These seven datasets form the final 2D protocol group.

2D macro statistics may be reported across this group.

---

## 17. Frozen 3D Dataset Group

The 3D experiments are:

```text
btcv
synapse
acdc
prostate158
```

For 3D experiments, the annotation unit is a **case**, rather than an
individual 2D slice.

The reported 3D experiments were completed under the previously frozen
case-level active-selection protocol.

The later strict 2D budget update was **not** retroactively applied to these
completed 3D experiments.

Therefore:

```text
final strict 2D protocol
!=
historical frozen 3D protocol
```

This distinction is intentional and must be preserved for reproducibility.

The frozen 3D experiments must not be rerun solely to force the later 2D budget
definition onto them.

---

## 18. Frozen 3D Active-selection Principle

For the frozen 3D experiments, Teacher predictions are reconstructed at
case level.

Case difficulty is measured from case-level segmentation performance.

A case is treated as difficult when:

```text
S_case < 0.5
```

Difficult cases are prioritized from worst to best.

The frozen 3D case-level annotation budget is:

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

Each later acquisition round adds one new Full case while difficult cases
remain and the frozen 3D protocol allows continued acquisition.

The exact historical 3D run must be reproduced from its frozen configuration
and provenance rather than by substituting the later 2D budget rules.

---

## 19. Evaluation Protocol

### 19.1 2D

Final Student predictions are evaluated in native image geometry.

Primary segmentation metrics are:

```text
Dice
IoU
```

MAE is additionally reported where supported by the evaluation pipeline.

Debug resizing must not be used for final reported results.

### 19.2 3D

The frozen 3D evaluation reports:

```text
DSC
HD95
ASSD
```

Metrics are computed after reconstructing prediction volumes in the correct
native geometry.

### 19.3 Modality-stratified Reporting

Because 2D and 3D use different annotation units and frozen acquisition
protocols, their macro statistics must be reported separately.

A single indiscriminate 11-dataset macro average should not be used as the main
experimental summary.

---

## 20. Pure Teacher Pseudo-label Quality

Teacher pseudo-label quality must be evaluated separately from final assembled
training-supervision quality.

For pure Teacher evaluation:

```text
Baseline Box pseudo
vs
Idea1 final Box pseudo
```

must be compared on the exact same final Box sample cohort.

The cohort must exclude samples that became Full.

This comparison answers:

```text
Did Teacher adaptation improve pseudo-label quality
for samples that remained Box-supervised?
```

---

## 21. Final Training-Supervision Quality

Idea1 final supervision contains:

```text
Full GT
+
Box pseudo-labels
```

Therefore its quality may be evaluated as final training-supervision quality.

However, this metric must not be called pure Teacher pseudo-label quality,
because Full samples use exact GT.

The distinction is:

```text
Pure Box Teacher quality
    -> Box samples only

Final supervision quality
    -> Full GT + Box pseudo-labels
```

---

## 22. Student Transfer Analysis

Improved supervision does not automatically imply proportional Student
generalization.

When investigating this effect, distinguish:

```text
supervision improvement
Student training-set fitting improvement
Student test-set improvement
fully supervised headroom
```

For a higher-is-better metric, the Student test gain is:

```text
G_test = M_Idea1,test - M_Baseline,test
```

The training-fit gain may be defined as:

```text
G_fit = M_Idea1,trainGT - M_Baseline,trainGT
```

The transfer differential is:

```text
G_test - G_fit
```

Upper should be used to quantify the remaining fully supervised headroom.

---

## 23. Gap-closure Analysis

For higher-is-better metrics such as Dice and IoU:

```text
gap_closure =
(Idea1 - Baseline) /
(Upper - Baseline)
```

For lower-is-better metrics:

```text
gap_closure =
(Baseline - Idea1) /
(Baseline - Upper)
```

Gap closure must be interpreted carefully when the Baseline-to-Upper
denominator is small.

Absolute metric gains should always be reported together with gap closure.

---

## 24. Required Validation

Before a released supervision set is accepted, validation must confirm:

```text
all training samples are accounted for
Full and Box subsets are disjoint
Full + Box = complete training set
Full samples contain GT supervision
Box samples contain valid tri-state supervision
Box pseudo-labels preserve ignore pixels where expected
no test sample appears in training supervision
geometry metadata is consistent
```

The final validation report must contain no unresolved errors.

---

## 25. Reproducibility Rules

The following rules are mandatory.

Do not overwrite:

```text
the frozen original MedSAM checkpoint
frozen baseline resources
completed formal experiment outputs
historical run manifests
```

Do not introduce differences between Baseline and Idea1 in:

```text
Student optimizer
Student training length
MIM initialization
Student architecture
evaluation geometry
evaluation implementation
```

unless such a difference is explicitly defined as a separate experiment.

Do not reuse an incompatible old Baseline pseudo-label set when the shared
pseudo-label arbitration protocol has changed.

Do not modify the frozen 3D protocol while cleaning or reorganizing the final
2D public code.

---

## 26. Historical Artifact Compatibility

Existing formal outputs may contain names such as:

```text
idea1_hard_full_medsam_ft
tri_box_final_idea1_hard_full_medsam_ft
tri_train_idea1_hard_full_medsam_ft
teacher_iteration_final_idea1_hard_full_medsam_ft.json
```

These are historical artifact identifiers.

They are intentionally preserved to maintain reproducibility.

Public source-code organization may use cleaner names such as:

```text
methods/baseline/
methods/common/
methods/idea1/
```

Source-code cleanup must not silently rewrite existing experiment schemas.

---

## 27. Protocol Summary

The experimental comparison is:

```text
Baseline
    Original MedSAM
    + Box pseudo-label supervision

Idea1
    Small Full budget
    + hard-sample acquisition
    + MedSAM mask-decoder adaptation
    + regenerated Box pseudo-labels
    + Full GT on acquired samples

Upper
    Full GT supervision for all training samples
```

For the final 2D experiments:

```text
per-round quota
= min(max(1, ceil(0.01*N)), 5)

maximum Full budget
= min(max(1, ceil(0.05*N)), 5*round_quota)

maximum acquisition rounds
= 5

hard threshold
= image_min_iou < 0.5

Round-0 seed
= 2026
```

The core scientific comparison must preserve all non-supervision variables so
that observed Student differences can be attributed to the supervision
strategy as cleanly as possible.
