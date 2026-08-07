# Idea1 V2 目录结构与扩展指南

> 目的：让后续只看目录、配置和核心代码即可恢复 V2 实验，不依赖聊天记录。  
> 原则：V1 冻结；V2 独立输出；Teacher / pseudo / MIM / Student / evaluation 分层存档。

---

## 1. 两个代码仓库

### MedSAM 研究仓库

```text
/storage/baiyuting/data/git_repos/MedSAM-research
```

当前 V2 分支：

```text
idea1-student-v2-mim-protocol
```

V2 核心代码锚点：

```text
40f9c31
```

负责：

- Baseline V2 pseudo；
- Idea1 V2 active learning；
- MedSAM decoder FT；
- final pseudo；
- probability-aware multiclass conflict resolver；
- Student supervision view 生成。

### Swin-UMamba 研究仓库

```text
/storage/baiyuting/data/git_repos/Swin-UMamba-research
```

建议使用同名 V2 分支：

```text
idea1-student-v2-mim-protocol
```

负责：

- target-domain MIM；
- Student dataset adapter；
- Baseline / Idea1 / Upper Student training；
- test evaluation。

---

## 2. 冻结资源

### MedSAM 主仓

```text
/storage/baiyuting/data/MedSAM-main
```

用途：

- 原始 MedSAM；
- 原始 checkpoint；
- Baseline V2 Teacher source。

不得把 V2 代码开发直接写回该目录。

### Swin-UMamba 主仓

```text
/storage/baiyuting/data/Swin-UMamba-main
```

VMamba-Tiny checkpoint：

```text
/storage/baiyuting/data/Swin-UMamba-main/
└── data/pretrained/vmamba/vmamba_tiny_e292.pth
```

该文件作为 V2 MIM 的共同 ImageNet 初始化。

---

## 3. V1 冻结结果

V1 formal root 与已有结果保持只读语义。

示意：

```text
/storage/baiyuting/data/out_data_idea1/
└── formal_runs/
    └── idea1_hard_full_medsam_ft/
```

V2 不得覆盖这里的任何：

```text
rounds/
teacher/
diagnosis/
selection/
pseudo/
student/
evaluation/
```

---

## 4. 推荐 V2 总根目录

```text
/storage/baiyuting/data/out_data_idea1/
└── formal_runs/
    └── idea1_student_v2_mim_protocol/
```

建议分为：

```text
idea1_student_v2_mim_protocol/
├── medsam_workspace/
├── student_views/
├── student_v2_mim_adaptation/
├── baseline_v2_mim_protocol/
├── idea1_v2_mim_protocol/
├── upper_v2_mim_protocol/
├── evaluation/
├── audits/
├── logs/
└── run_manifests/
```

---

## 5. MedSAM V2 workspace

建议：

```text
<V2_ROOT>/medsam_workspace/
└── <dataset>/
    └── fold_0/
        ├── meta/
        ├── rounds/
        ├── final/
        └── ...
```

其中 `fold_root`：

```text
<V2_ROOT>/medsam_workspace/<dataset>/fold_0
```

### 5.1 meta

```text
meta/
├── annotation_budget_<method>.json
├── method_lineage_<method>.json
├── teacher_iteration_final_<method>.json
└── final_method_<method>.json
```

必须保存：

- budget version；
- annotation unit；
- per-round quota；
- cumulative 5% cap；
- IoU threshold；
- final round；
- terminal status；
- final checkpoint；
- Git commit / config lineage（若当前脚本已支持则直接写；否则在 run manifest 中补充）。

### 5.2 rounds

```text
rounds/
└── <method>/
    ├── round_00/
    │   ├── selection/
    │   ├── finetune_pairs/
    │   ├── teacher/
    │   └── diagnosis/
    ├── round_01/
    └── ...
```

每轮至少保留：

```text
selection/selection.json
finetune_pairs/pairs.json
teacher/medsam_ft.pth
diagnosis/per_image_metrics.csv      # 2D
diagnosis/per_case_metrics.csv       # 3D
diagnosis/selection_result.json
```

### 5.3 Round0

Round0 目录必须能审计：

- seed；
- random selection；
- round0 quota；
- actual selected count；
- cumulative Full；
- remaining Box。

### 5.4 Round1+

每轮必须能审计：

- `image_min_iou` 或 `case_min_slice_class_iou`；
- hard candidate；
- selected_new_ids；
- current Full count；
- remaining budget；
- `CONTINUE / CONVERGED / BUDGET_EXHAUSTED`。

---

## 6. Student supervision view

建议：

```text
<V2_ROOT>/student_views/
├── baseline_v2/
│   └── <dataset>/fold_0/
├── idea1_v2/
│   └── <dataset>/fold_0/
└── upper_v2/
    └── <dataset>/fold_0/
```

逻辑：

### Baseline V2

```text
all train
→ Box
→ Original MedSAM
→ V2 resolver
→ tri pseudo
```

### Idea1 V2

```text
Full subset
→ GT

remaining Box
→ final FT MedSAM
→ V2 resolver
→ tri pseudo
```

### Upper V2

```text
all train
→ GT
```

如果现有 Student loader 需要固定 pseudo name，可在对应 view 内使用稳定的逻辑名，不把绝对路径硬编码进训练代码。

---

## 7. 推荐 pseudo 名称

为防止和 V1 混淆，V2 名称必须带版本。

推荐：

```text
tri_train_baseline_v2_probability_arbitration
tri_train_idea1_v2_probability_arbitration
```

Upper 直接使用 GT，不需要伪标签名。

禁止复用：

```text
tri_train_boxonly
```

作为 Baseline V2 正式输入。

---

## 8. Shared MIM

推荐：

```text
<V2_ROOT>/student_v2_mim_adaptation/
└── <dataset>/
    └── fold_0/
        ├── adapted_encoder_last.pth
        ├── adapted_encoder_for_train_last.pth
        ├── train_log.*
        └── config.*
```

原则：

```text
一个 dataset/fold
只运行一次 Shared MIM
```

然后：

```text
Baseline V2
Idea1 V2
Upper V2
```

三组都引用同一个：

```text
adapted_encoder_for_train_last.pth
```

不得三组分别训练 MIM。

---

## 9. Student 输出

### Baseline

```text
<V2_ROOT>/baseline_v2_mim_protocol/
└── <dataset>/fold_0/
```

### Idea1

```text
<V2_ROOT>/idea1_v2_mim_protocol/
└── <dataset>/fold_0/
```

### Upper

```text
<V2_ROOT>/upper_v2_mim_protocol/
└── <dataset>/fold_0/
```

每个目录建议至少保留：

```text
config/
checkpoint/
train_log/
prediction/
metrics/
```

三组目录结构必须同构，便于后续自动汇总。

---

## 10. Evaluation

推荐：

```text
<V2_ROOT>/evaluation/
└── <dataset>/
    └── fold_0/
        ├── baseline_v2/
        ├── idea1_v2/
        ├── upper_v2/
        └── summary/
```

`summary/` 建议保存：

```text
main_metrics.csv
comparison.json
run_manifest.json
```

2D：

- DSC；
- IoU；
- 以及当前正式协议中的其他指标。

3D：

- DSC；
- HD95；
- ASSD；
- 以 native reconstruction + real spacing + no resize 为正式协议。

---

## 11. Audits

推荐统一：

```text
<V2_ROOT>/audits/
├── pseudo_quality/
├── pseudo_statistics/
├── active_learning/
├── student_supervision/
└── evaluation_protocol/
```

重点审计：

### pseudo

- foreground ratio；
- unknown ratio；
- coverage；
- pseudo-vs-GT IoU；
- Baseline V2 vs Idea1 V2 changed pixels；
- Full 样本是否正确替换为 GT；
- multiclass tie / conflict count。

### active learning

- quota；
- cumulative 5%；
- minimum IoU；
- selected hard samples；
- terminal state。

### Student

- shared MIM checkpoint hash/path；
- encoder freeze；
- crop policy；
- DS outputs；
- optimizer；
- LR；
- epoch；
- seed。

---

## 12. Logs 与 run manifest

每次正式运行建议写：

```text
<V2_ROOT>/run_manifests/
└── <dataset>_<fold>_<timestamp>.json
```

至少记录：

```text
dataset
fold
medsam_git_branch
medsam_git_commit
swin_git_branch
swin_git_commit
pseudo_protocol
active_learning_protocol
student_protocol
mim_checkpoint
output_paths
start_time
end_time
terminal_state
```

这样后面扩展到新数据集时，不需要依赖聊天记录判断“当时到底跑了哪一版代码”。

---

## 13. 核心文件职责

MedSAM V2：

```text
generate_pseudo_labels.py
    通用 MedSAM proposal / pseudo 生成

idea1_hard_full_medsam_ft/
├── 01_init_idea1_workspace.py
│   初始化隔离 workspace + Active Learning V2 budget
│
├── 02_select_round0_random.py
│   Round0 随机 Full
│
├── 03_build_full_finetune_pairs.py
│   构造 Full 微调 pair
│
├── 04_finetune_medsam_mask_decoder.py
│   微调 MedSAM mask decoder
│
├── 05_score_remaining_box_pool.py
│   remaining Box pseudo-vs-GT 诊断
│   2D: image_min_iou
│   3D: case_min_slice_class_iou
│
├── 06_select_next_hard_samples.py
│   CONTINUE / CONVERGED / BUDGET_EXHAUSTED
│
├── 07_generate_method_tri_pseudo.py
│   生成 final method tri pseudo
│
├── multiclass_resolver_v2.py
│   probability-aware multiclass resolver
│
├── run_iterative_teacher.py
│   唯一 Active Learning V2 controller
│
├── run_dataset_full.py
│   dataset 级 wrapper；调用 run_iterative_teacher
│
├── run_final_pipeline.py
│   final pseudo / hybrid / student view
│
└── configs/default.json
    V2 protocol config
```

Swin V2：

```text
pipeline/
├── adapt_student_mim.py
├── student_patch_dataset.py
└── train_student.py
```

正式 runner 建议统一支持：

```text
Shared MIM
Baseline V2
Idea1 V2
Upper V2
```

---

## 14. 后续扩展规则

### 新数据集

只允许增加：

- dataset manifest / split；
- dataset-specific path/config；
- evaluation adapter（确有必要时）。

不应复制出另一套 Active Learning 代码。

### 新主动学习策略

如果未来形成 V3：

- 新建 protocol version；
- 新 output root；
- 新 branch；
- 不覆盖 `active_learning_v2`；
- 旧 V2 结果保持可复现。

### 新 resolver

如果更改：

- same-class fusion；
- cross-class conflict；
- tie policy；
- box union policy；

必须提升 pseudo protocol 名称，不能仍叫：

```text
student_v2_probability_arbitration
```

---

## 15. 推荐正式目录实例

TG3K：

```text
/storage/baiyuting/data/out_data_idea1/formal_runs/
└── idea1_student_v2_mim_protocol/
    ├── medsam_workspace/
    │   └── tg3k/
    │       └── fold_0/
    │
    ├── student_views/
    │   ├── baseline_v2/
    │   │   └── tg3k/fold_0/
    │   ├── idea1_v2/
    │   │   └── tg3k/fold_0/
    │   └── upper_v2/
    │       └── tg3k/fold_0/
    │
    ├── student_v2_mim_adaptation/
    │   └── tg3k/fold_0/
    │
    ├── baseline_v2_mim_protocol/
    │   └── tg3k/fold_0/
    │
    ├── idea1_v2_mim_protocol/
    │   └── tg3k/fold_0/
    │
    ├── upper_v2_mim_protocol/
    │   └── tg3k/fold_0/
    │
    ├── evaluation/
    │   └── tg3k/fold_0/
    │
    ├── audits/
    ├── logs/
    └── run_manifests/
```

这是后续所有 2D/3D V2 实验建议遵守的统一布局。
