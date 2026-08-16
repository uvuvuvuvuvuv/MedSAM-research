# 1. 全局根目录

固定使用下面几个根路径：

```
IDEA_ROOT=/storage/baiyuting/data/out_data_idea1
BASE_DIR=$IDEA_ROOT/MedSAM-main/data
PROCESSED_ROOT=$BASE_DIR/processed
```

展开后：

```
/storage/baiyuting/data/out_data_idea1/
├── MedSAM-main/data/processed/
│   ├── kvasirseg/fold_0/
│   ├── cvc_clinicdb/fold_0/
│   ├── tn3k/fold_0/
│   ├── tg3k/fold_0/
│   ├── ddti/fold_0/
│   ├── otu_2d/fold_0/
│   ├── ph2/fold_0/
│   ├── btcv/fold_0/
│   ├── synapse/fold_0/
│   ├── acdc/fold_0/
│   └── prostate158/fold_0/
├── logs/
└── analysis_teacher_diagnosis_<METHOD>/
```

------

# 2. 方法名命名规则

以后每一个实验先定义一个唯一的方法名：

```
METHOD=idea1_<variant>
```

# 3. 每个数据集的基础目录

每个数据集都在：

```
$PROCESSED_ROOT/<dataset>/fold_0/
```

例如 TG3K：

```
$PROCESSED_ROOT/tg3k/fold_0/
```

内部基础结构：

```
fold_0/
├── native_npy/
│   ├── imgs/
│   └── gts/
├── teacher_npy/
│   ├── imgs/
│   └── gts/
├── student_npy/
│   ├── imgs/
│   └── gts/
├── prompts/
├── meta/
├── pseudo_student/
└── pseudo_teacher/
```

含义：

```
native_npy      原始图像空间
teacher_npy     MedSAM 输入空间，通常是 1024×1024
student_npy     后续训练网络使用的空间
prompts         输入 MedSAM 的 prompt JSON
meta            manifest、geometry、prompt 审计、统计信息
pseudo_student  student 空间伪标签，主要用于训练和评估
pseudo_teacher  teacher 空间伪标签，主要用于检查和可视化
```

------

# 4. Prompt 文件设计

Prompt 文件放在：

```
$PROCESSED_ROOT/<dataset>/fold_0/prompts/
```

命名规则：

```
prompts_train_${METHOD}.json
```

# 5. Prompt 审计和统计文件

Prompt 的审计与统计文件放在：

```
$PROCESSED_ROOT/<dataset>/fold_0/meta/
```

命名规则：

```
${METHOD}_prompt_audit_train.csv
${METHOD}_prompt_summary_train.json
prompt_build_stats_${METHOD}.json
```

# 6. 伪标签目录设计

伪标签分为两个空间：

```
pseudo_student/
pseudo_teacher/
```

主要使用：

```
pseudo_student/
```

因为后续训练和指标评估一般基于 student 空间。

正式伪标签目录命名规则：

```
tri_train_${METHOD}/
```

其中 `tri` 表示三值伪标签：

```
0    background
1    foreground
255  unknown / ignore
```

注意：

```
pseudo_student/tri_train/
pseudo_teacher/tri_train/
```

通常是 `generate_pseudo_labels.py` 的临时输出目录。

正式保存实验结果时，必须复制到：

```
pseudo_student/tri_train_${METHOD}/
pseudo_teacher/tri_train_${METHOD}/
```

否则下一次生成别的方法时，`tri_train/` 会被覆盖。

# 7. Box-only baseline 目录

无论新 Idea1 怎么改，都建议保留 box-only baseline：

```
pseudo_student/tri_train_boxonly/
pseudo_teacher/tri_train_boxonly/
```

新方法和 box-only 的关系应该是：

```
boxonly:
tri_train_boxonly/

new idea1:
tri_train_${METHOD}/
```

不要覆盖 `tri_train_boxonly`，它是对比基线。

# 8. 分析结果目录设计

新实验分析目录应为：

```
$IDEA_ROOT/analysis_teacher_diagnosis_${METHOD}/
```

每个数据集内部：

```
analysis_teacher_diagnosis_${METHOD}/<dataset>/
├── stats_and_vis/
├── categorized_vis/
├── category_summary.csv
└── category_counts.json
```

# 9. 指标输出目录

指标和普通可视化放在：

```
$IDEA_ROOT/analysis_teacher_diagnosis_${METHOD}/<dataset>/stats_and_vis/
```

常见文件：

```
per_sample_metrics.csv
summary_metrics.csv
vis/
```

含义：

```
per_sample_metrics.csv
每张图、每个方法的 IoU / Dice / Precision / Recall / fg_ratio / unknown_ratio。

summary_metrics.csv
每个方法在整个数据集上的平均指标。

vis/
普通对比可视化图。
```

# 10. 日志目录设计

日志统一放在：

```
$IDEA_ROOT/logs/
```

按阶段和方法名划分：

```
logs/
├── prompts/${METHOD}/
├── pseudo/${METHOD}/
├── analysis/${METHOD}/
└── train/${METHOD}/
```

常见日志文件：

```
logs/prompts/${METHOD}/run_prompt_2d.log
logs/prompts/${METHOD}/run_prompt_3d.log
logs/pseudo/${METHOD}/nohup_pseudo.log
logs/pseudo/${METHOD}/<dataset>.log
logs/analysis/${METHOD}/compare.log
```
