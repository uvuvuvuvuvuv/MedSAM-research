# Idea1 V2 实验协议与复现说明

> 适用分支：`idea1-student-v2-mim-protocol`  
> V2 核心代码锚点：`40f9c31`  
> 目标：在不覆盖 V1 冻结结果的前提下，统一 Baseline V2 / Idea1 V2 / Upper V2 的 Teacher、伪标签、MIM、Student 与评估协议。

---

## 1. V2 的核心目标

V2 不改变研究问题本身，而是把实验协议收紧为三个可公平比较的组别：

| 组别 | Teacher | Student 监督 |
|---|---|---|
| **Baseline V2** | 原始 MedSAM | 全部训练样本使用 Box 三值伪标签 |
| **Idea1 V2** | Active Learning V2 选择困难 Full 后微调的 MedSAM | Full 样本使用 GT；剩余 Box 样本使用三值伪标签 |
| **Upper V2** | 不需要 Teacher | 全部训练样本使用 GT |

三组 Student 必须共享同一套训练协议。Upper 只改变监督来源，不改变 Student 架构、MIM、优化器、训练轮数或评估方式。

---

## 2. V1 / V2 隔离原则

### 2.1 V1 必须冻结

V1 已完成的以下内容不得覆盖：

- Frozen Baseline；
- Idea1 V1；
- Frozen Upper；
- 原有正式评估结果；
- 原有可视化、诊断 CSV、selection JSON、teacher checkpoint；
- 原有 `baseline_boxonly` 控制实验。

### 2.2 V2 必须使用新的输出树

V2 所有新生成的：

- active-learning selection；
- MedSAM FT checkpoint；
- Baseline V2 pseudo；
- Idea1 V2 pseudo；
- Student view；
- MIM checkpoint；
- Baseline / Idea1 / Upper Student checkpoint；
- test prediction；
- metrics；

都必须写入新的 V2 目录，不能写回 V1 formal run。

推荐统一根目录：

```text
/storage/baiyuting/data/out_data_idea1/formal_runs/
└── idea1_student_v2_mim_protocol/
```

---

## 3. Active Learning V2

### 3.1 统一原则

2D 和 3D 都采用：

- 每轮理论新增 Full 标注：
  `max(ceil(0.01 × N), 5)`
- 累计 Full 标注预算：
  `max(1, ceil(0.05 × N))`
- IoU 阈值：
  `0.5`

区别只在标注单位：

- 2D：`image`
- 3D：`case`

---

### 3.2 2D

设训练图像数为 `N_train_images`。

每轮理论新增：

```text
B_round_2D = max(ceil(0.01 × N_train_images), 5)
```

累计 Full 上限：

```text
B_max_2D = max(1, ceil(0.05 × N_train_images))
```

Round0：

- 随机选择；
- 实际数量为 `min(B_round_2D, B_max_2D)`。

Round1+：

- 在 remaining Box pool 中计算每个实例的 pseudo-vs-GT IoU；
- 对每张图像取：

```text
image_min_iou = min(all instance IoUs in the image)
```

- `image_min_iou < 0.5` 即为困难图像；
- 按 `image_min_iou` 从小到大选择；
- 本轮实际新增：

```text
min(
    B_round_2D,
    B_max_2D - current_full_count,
    num_current_hard_images
)
```

---

### 3.3 3D

设训练 case 数为 `N_train_cases`。

每轮理论新增：

```text
B_round_3D = max(ceil(0.01 × N_train_cases), 5)
```

累计 Full 上限：

```text
B_max_3D = max(1, ceil(0.05 × N_train_cases))
```

Round0：

- 随机选择 case；
- 实际数量为 `min(B_round_3D, B_max_3D)`。

Round1+：

- annotation unit 固定为 case；
- 在每个 remaining case 中计算有效 `slice × class` IoU；
- case 困难度：

```text
case_min_slice_class_iou
=
min(all valid slice-class IoUs in the case)
```

- `case_min_slice_class_iou < 0.5` 即为困难 case；
- 按该值从小到大选择；
- 本轮实际新增：

```text
min(
    B_round_3D,
    B_max_3D - current_full_case_count,
    num_current_hard_cases
)
```

---

### 3.4 终止状态

V2 只允许三种状态：

#### `CONVERGED`

```text
所有 remaining pseudo target 与 GT 的 IoU >= 0.5
```

#### `CONTINUE`

```text
仍存在 IoU < 0.5 的困难 target
且累计 Full 预算尚未耗尽
```

#### `BUDGET_EXHAUSTED`

```text
仍存在 IoU < 0.5 的困难 target
但累计 Full 已达到 ceil(5%) 上限
```

`max_rounds=8` 不再是实验终止条件。

---

## 4. V2 多类伪标签合同

### 4.1 目标

Baseline V2 与 Idea1 V2 必须使用同一个 resolver，避免因为冲突处理逻辑不同导致不公平比较。

### 4.2 统一 resolver

核心文件：

```text
idea1_hard_full_medsam_ft/multiclass_resolver_v2.py
```

核心规则：

1. 同类 overlap：
   - support 使用 OR；
   - probability 使用 MAX；
   - 不产生冲突。

2. 不同类 overlap：
   - 只在已选择 refined-mask support 内进行；
   - 使用与 `best_mask` 同一个 MedSAM multimask candidate 的概率；
   - 按概率做 pixel-level argmax。

3. 数值 tie：
   - epsilon = `1e-6`；
   - 无法可靠区分时保持 `255`。

4. canonical box union 外：
   - `0`。

5. box union 内但没有被任何 selected refined mask 确认：
   - `255`。

6. proposal 失败：
   - valid prompt box 仍先进入 canonical box union；
   - 未确认区域保持 `255`，不能错误写成背景。

### 4.3 概率来源

概率图必须与最终选中的 mask 来自同一个 MedSAM multimask candidate。

禁止：

```text
先用一个 candidate 选 mask
再独立跑一次 predictor 取另一张 probability map
```

V2 中 `best_mask`、`best_probability` 和对应 score 必须保持 candidate 对齐。

---

## 5. Baseline V2 / Idea1 V2 / Upper V2

### 5.1 Baseline V2

Teacher：

```text
Original MedSAM
```

监督生成：

```text
全部训练样本
    ↓
Box prompts
    ↓
Original MedSAM
    ↓
V2 probability-aware resolver
    ↓
全部 Box 三值伪标签
```

Baseline V2 不能直接复用 V1 的 `tri_train_boxonly`，因为 V2 resolver 已改变。

---

### 5.2 Idea1 V2

Teacher：

```text
Original MedSAM
    ↓
Round0 random Full
    ↓
MedSAM mask decoder FT
    ↓
remaining Box pseudo-vs-GT diagnosis
    ↓
minimum-IoU hard selection
    ↓
继续 FT / 停止
```

最终 Student 监督：

```text
Full samples
→ GT

Remaining Box samples
→ final FT MedSAM
→ V2 probability-aware resolver
→ tri pseudo
```

---

### 5.3 Upper V2

Upper 不依赖 MedSAM Teacher。

监督：

```text
全部训练样本 → GT
```

但 Student 侧必须与另外两组保持完全一致。

---

## 6. Student V2 协议

三组 Student 必须共享以下协议。

### 6.1 Backbone / 初始化

- Swin-UMamba；
- 使用同一个 VMamba-Tiny ImageNet checkpoint；
- 先进行一次共享的 target-domain MIM；
- Baseline / Idea1 / Upper 都从同一个 adapted encoder 初始化。

### 6.2 MIM

当前项目级 V2 设置：

```text
image_size      = 192
mask_ratio      = 0.6
mask_patch_size = 16
optimizer       = AdamW
lr              = 2e-4
weight_decay    = 0.05
freeze_encoder  = first 10 MIM epochs
epochs          = 50   # 项目级预算，不声称是论文针对 TG3K 的规定
```

MIM 输出以最后一轮为正式训练初始化：

```text
adapted_encoder_last.pth
adapted_encoder_for_train_last.pth
```

训练 loss 最优 checkpoint 只作为审计信息，不作为三组不同的 early-stopping 来源。

### 6.3 Student segmentation

统一设置：

```text
epochs          = 50
optimizer       = AdamW
lr              = 1e-4
weight_decay    = 0.05
freeze_encoder  = first 10 epochs
seed            = 3407
crop_policy     = full_image
deep_supervision_outputs = 3
```

Deep supervision 采用前三个最高空间分辨率输出，权重：

```text
4/7, 2/7, 1/7
```

三组必须完全一致。

---

## 7. 结果比较

### 7.1 主表

建议主表仍报告：

```text
Frozen Baseline / Baseline V2
Idea1 V2
Upper V2
```

正式 V2 对比必须保证 Student protocol 一致。

### 7.2 内部诊断

至少保存：

- annotation budget；
- 每轮 selection；
- per-image / per-case IoU；
- minimum IoU；
- hard candidate 数量；
- terminal state；
- pseudo label statistics；
- 0 / FG / 255 比例；
- Baseline V2 与 Idea1 V2 pseudo 差异；
- Student train log；
- test metrics。

---

## 8. V2 复现顺序

```text
1. 冻结代码版本
2. 创建新的 V2 output root
3. 生成 Baseline V2 pseudo
4. 运行 Idea1 V2 Active Learning
5. 生成 Idea1 V2 final pseudo
6. 做 Baseline/Idea1 pseudo audit
7. 运行一次 Shared MIM
8. Baseline V2 Student
9. Idea1 V2 Student
10. Upper V2 Student
11. 使用同一 test protocol 评估三组
12. 汇总正式结果与诊断
```

---

## 9. 禁止事项

- 不覆盖 V1 formal results；
- 不复用 V1 pseudo 作为 Baseline V2；
- 不让 Baseline / Idea1 使用不同 resolver；
- 不让三组 Student 使用不同 MIM 初始化；
- 不以 `max_rounds=8` 作为主动学习停止条件；
- 不以 macro IoU 代替 minimum-IoU 困难判定；
- 不把 3D slice 当作 Full 标注单位；
- 不把 `ceil(5%)` 表述为“严格不超过 5%”；论文中应写为“5% cumulative budget, rounded up to the nearest whole annotation unit”。

---

## 10. 版本锁

MedSAM V2 核心代码锚点：

```text
branch: idea1-student-v2-mim-protocol
commit: 40f9c31
```

后续文档提交可以位于该 commit 之后，但任何正式实验结果都应同时记录：

- Git branch；
- Git commit；
- config；
- dataset；
- fold；
- output root；
- terminal status。
