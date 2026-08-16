# Idea1：少量困难 Full 标注驱动的 MedSAM Mask Decoder 迭代微调方案
## V2：Baseline 代码继承、方法目录隔离与可复现实现版

> **方法定位**：先完整整理并冻结 `frozen_baseline_v1` 的代码、数据合同和结果，再从这份 baseline 代码快照复制出 Idea1 工作副本，仅增加“少量 Full 抽样、困难样本筛选、MedSAM mask decoder 微调和方法专属监督集组装”。
> **核心原则**：原始 baseline 代码、baseline 三值伪标签、baseline student checkpoint 和 baseline 评估结果始终只读；任何 Full GT 替换操作只发生在 Idea1 自己的 `tri_train_${METHOD}` 目录内。
> **当前正式比较**：`Baseline / Idea1 / Upper`。
> **当前不包含**：双阈值、FG/BG bank、prototype、shape template、half-shell、形状损失、置信度融合及其他软约束。

---

# 0. 本版相对上一版的关键修正

上一版虽然说明了“不覆盖 frozen baseline”，但以下表述仍容易造成误解：

```text
在原仓库 extensions/ 中直接增加 Idea1
Full GT 覆盖到 tri_train
构造 final_fold
```

本版统一修正为：

1. **先整理 baseline 代码快照，再从快照复制 Idea1 工作代码**；
2. **原始 baseline 仓库和 frozen 结果不作为 Idea1 的写入目录**；
3. **baseline Box-only 伪标签独立保存为 `tri_train_boxonly`**；
4. **Idea1 最终混合监督独立保存为 `tri_train_${METHOD}`**；
5. “Full GT 覆盖”改称为**组装 Idea1 方法专属监督集**；
6. Full GT 只替换 `tri_train_${METHOD}` 中相应样本，不修改：
   - 原始 baseline 的 `pseudo_student/tri_train`；
   - Idea1 根目录中的 `tri_train_boxonly`；
   - baseline student checkpoint；
   - baseline 评估结果；
7. student 使用方法专属 `student_view`，在不改 baseline loader 的情况下读取不同监督目录；
8. 因此可以随时独立比较：
   - frozen baseline；
   - Idea1；
   - upper；
   - box-only pseudo 与 Idea1 hybrid label。

一句话概括：

> **先复制 baseline 代码形成可追溯的 Idea1 工作副本，再在 Idea1 独立数据根目录中生成新监督和新结果；绝不在 frozen baseline 原目录上做原地修改。**

---

# 1. 文档目标

本文档用于指导一个不了解前序工作的实现者，完成以下工作：

```text
整理 frozen baseline 代码
→ 固定代码和数据血缘
→ 创建 Idea1 工作副本
→ Round 0 随机 Full
→ Round 1+ 困难 Full
→ 微调 MedSAM mask decoder
→ 为剩余 Box 生成 baseline-compatible 三值伪标签
→ 组装 Idea1 专属 hybrid supervision
→ 使用原 Swin-UMamba 训练
→ 与 Baseline / Upper 比较
```

阅读本文件和 frozen baseline 文档后，实现者应明确：

- 哪些代码来自 baseline；
- 哪些文件必须原样保留；
- Idea1 新增了哪些脚本；
- 数据从哪里读取；
- 新文件写到哪里；
- Round 0 和后续轮次如何选样；
- 2D 与 3D 的标注单位；
- MedSAM 微调哪些参数；
- 三值伪标签如何生成；
- Full GT 如何进入 student；
- baseline 与 Idea1 如何同时保留并公平比较；
- 如何检查泄漏、空间、标签、spacing 和结果完整性。

---

# 2. 三组实验的最终定义

| 组别 | Teacher | Student 监督 | 输出是否独立 |
|---|---|---|---:|
| Baseline | 原始 MedSAM | 全部训练样本使用 Box 三值伪标签 | 是，frozen 只读 |
| Idea1 | 少量 Full GT 迭代微调后的 MedSAM | Full GT + 剩余 Box 三值伪标签 | 是，方法专属目录 |
| Upper | 不依赖伪标签 | 全部训练样本使用 GT | 是，frozen 只读 |

## 2.1 Baseline

```text
train GT
→ native-space tight boxes
→ original MedSAM
→ threshold = 0.5
→ Box-only tri pseudo
→ original Swin-UMamba
```

Baseline student label：

```text
0      background
1..K   class label
255    unknown / ignore
```

## 2.2 Idea1

```text
Round 0 random Full
→ Full GT + baseline boxes
→ fine-tune MedSAM mask decoder
→ score remaining Box pool
→ Round 1+ add hardest Full
→ repeat until budget/stop
→ final fine-tuned MedSAM
→ tri pseudo for remaining Box
→ Full samples use GT
→ Idea1-specific hybrid supervision
→ original Swin-UMamba
```

## 2.3 Upper

```text
all train GT
→ original Swin-UMamba
```

---

# 3. Idea1 相对 Baseline 的唯一变化

## 3.1 新增内容

```text
1. Round 0 固定随机 Full；
2. Round 1 开始基于训练集隐藏 GT-IoU 选择困难 Full；
3. 累计 Full GT 用于微调 MedSAM mask decoder；
4. 最终微调 teacher 为剩余 Box 生成三值伪标签；
5. Full 样本在 Idea1 专属监督目录中使用 GT。
```

## 3.2 必须继承且不得改变

```text
数据集和 split
native / teacher / student 三空间
teacher 输入 1024×1024×3
native-space tight box
每个连通目标独立 box
prompt 文件内容
threshold = 0.5
三值标签定义
多类别冲突规则
3D 空切片规则
student 架构
student crop
student loss
student 超参数
test split
推理脚本
2D/3D 评估协议
```

## 3.3 明确删除或禁止

```text
双阈值
low/high threshold
FG bank
BG bank
prototype bank
shape clustering
shape template
adaptive shape
half-shell
shape loss
TV loss
box weak loss
bank fusion
额外 student 模块
```

---

# 4. Baseline-first 代码组织原则

## 4.1 为什么必须先整理 baseline 代码

Idea1 不是独立重写一套 MedSAM→Swin-UMamba，而是：

```text
frozen_baseline_v1
└── idea1_hard_full_medsam_ft
```

因此必须先获得一份可追溯、不可变的 baseline 代码快照。否则后续出现指标变化时，无法判断变化来自：

- 新的 Idea1；
- baseline 代码被无意修改；
- student loader 改变；
- pseudo 规则改变；
- eval 脚本改变；
- spacing 或 geometry 改变。

## 4.2 原始仓库的地位

以下目录是当前实际 baseline 来源：

```text
/storage/baiyuting/data/MedSAM-main
/storage/baiyuting/data/Swin-UMamba-main
```

在建立 Idea1 后，它们分为两类用途：

### 只读参考

```text
原始 baseline 代码
原始 processed 数据
原始 baseline pseudo
原始 baseline student 结果
原始 upper student 结果
```

### 禁止 Idea1 写入

```text
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/pseudo_student/tri_train
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/pseudo_teacher/tri_train
/storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline
/storage/baiyuting/data/Swin-UMamba-main/work_dir/upper
```

## 4.3 两份代码

Idea1 根目录必须保留：

```text
1. baseline_v1_snapshot：不可写的代码快照；
2. idea1_hard_full_medsam_ft：从快照复制出来的可开发工作副本。
```

关系：

```text
原始仓库当前 frozen commit
        ↓ snapshot
baseline_v1_snapshot（只读）
        ↓ copy
idea1_hard_full_medsam_ft（可写）
        ↓ add only Idea1
```

---

# 5. 全局根目录

沿用此前 IdeaX 设计：

```bash
IDEA_ROOT=/storage/baiyuting/data/out_data_idea1
METHOD=idea1_hard_full_medsam_ft

CODE_ROOT=$IDEA_ROOT/code
BASE_DIR=$IDEA_ROOT/MedSAM-main/data
PROCESSED_ROOT=$BASE_DIR/processed
VIEW_ROOT=$IDEA_ROOT/student_views
WORK_ROOT=$IDEA_ROOT/Swin-UMamba-main/work_dir
LOG_ROOT=$IDEA_ROOT/logs
SUMMARY_ROOT=$IDEA_ROOT/summaries
```

整体结构：

```text
/storage/baiyuting/data/out_data_idea1/
├── code/
├── baseline_reference/
├── MedSAM-main/
│   └── data/
│       └── processed/
├── student_views/
├── Swin-UMamba-main/
│   └── work_dir/
├── analysis_teacher_diagnosis_<METHOD>/
├── logs/
└── summaries/
```

---

# 6. 代码目录设计

## 6.1 完整结构

```text
$IDEA_ROOT/code/
├── baseline_v1_snapshot/
│   ├── MedSAM-main/
│   ├── Swin-UMamba-main/
│   ├── baseline_commit.json
│   ├── baseline_files_sha256.txt
│   └── README_BASELINE_SNAPSHOT.md
│
└── idea1_hard_full_medsam_ft/
    ├── MedSAM-main/
    │   ├── generate_prompts.py
    │   ├── generate_pseudo_labels.py
    │   ├── utils/
    │   └── idea1/
    │       ├── configs/
    │       │   ├── default.yaml
    │       │   └── datasets.yaml
    │       ├── 00_audit_baseline_contract.py
    │       ├── 01_init_idea1_workspace.py
    │       ├── 02_select_round0_random.py
    │       ├── 03_build_full_finetune_pairs.py
    │       ├── 04_finetune_medsam_mask_decoder.py
    │       ├── 05_score_remaining_box_pool.py
    │       ├── 06_select_next_hard_samples.py
    │       ├── 07_generate_method_tri_pseudo.py
    │       ├── 08_assemble_method_supervision.py
    │       ├── 09_build_student_view.py
    │       ├── 10_validate_method_data.py
    │       ├── 11_finalize_method.py
    │       ├── run_iterative_teacher.py
    │       └── README.md
    │
    ├── Swin-UMamba-main/
    │   ├── pipeline/
    │   │   ├── train_student.py
    │   │   ├── infer_student.py
    │   │   ├── eval_2d.py
    │   │   ├── eval_3d.py
    │   │   └── test_student_patch_dataset.py
    │   └── scripts/
    │       ├── run_train_idea1.sh
    │       ├── run_infer_idea1.sh
    │       └── run_eval_idea1.sh
    │
    ├── parent_baseline_commit.json
    ├── idea1_diff_from_baseline.patch
    ├── idea1_files_sha256.txt
    └── README_CODE_LINEAGE.md
```

## 6.2 Baseline snapshot 包含什么

推荐复制整个代码仓库，但排除大数据和运行结果：

```text
保留：
源代码
配置
requirements / environment 文件
必要 shell 脚本
git commit 信息

排除：
data/raw
data/organized
data/processed
work_dir
缓存
__pycache__
大型 checkpoint
已有预测和评估结果
```

checkpoint 不复制进代码目录，而在 lineage 中记录绝对路径和 SHA256。

## 6.3 Idea1 工作副本如何创建

推荐顺序：

```text
Step 1：记录原始 MedSAM/Swin repo commit 和 git status
Step 2：复制代码形成 baseline_v1_snapshot
Step 3：baseline_v1_snapshot 设置为只读
Step 4：复制 baseline_v1_snapshot 形成 Idea1 工作副本
Step 5：只在 Idea1 工作副本中新增 idea1/ 和运行脚本
Step 6：生成 baseline→Idea1 diff patch
```

示意命令：

```bash
mkdir -p "$CODE_ROOT/baseline_v1_snapshot"
mkdir -p "$CODE_ROOT/idea1_hard_full_medsam_ft"

rsync -a \
  --exclude '.git' \
  --exclude 'data/raw' \
  --exclude 'data/organized' \
  --exclude 'data/processed' \
  --exclude 'work_dir' \
  --exclude '__pycache__' \
  /storage/baiyuting/data/MedSAM-main/ \
  "$CODE_ROOT/baseline_v1_snapshot/MedSAM-main/"

rsync -a \
  --exclude '.git' \
  --exclude 'data' \
  --exclude 'work_dir' \
  --exclude '__pycache__' \
  /storage/baiyuting/data/Swin-UMamba-main/ \
  "$CODE_ROOT/baseline_v1_snapshot/Swin-UMamba-main/"

cp -a \
  "$CODE_ROOT/baseline_v1_snapshot/MedSAM-main" \
  "$CODE_ROOT/idea1_hard_full_medsam_ft/"

cp -a \
  "$CODE_ROOT/baseline_v1_snapshot/Swin-UMamba-main" \
  "$CODE_ROOT/idea1_hard_full_medsam_ft/"
```

正式实现时必须根据实际目录层级检查 `cp` 结果，避免多套一层目录。

## 6.4 哪些 baseline 文件保持原样

Idea1 工作副本中的以下代码原则上保持与 baseline snapshot 字节一致：

```text
utils/processed.py
generate_prompts.py
generate_pseudo_labels.py 的 baseline 核心规则
train_student.py
infer_student.py
eval_2d.py
eval_3d.py
test_student_patch_dataset.py
```

Idea1 优先通过“调用、包装、传参”复用 baseline，而不是复制其算法后重新实现。

## 6.5 允许的最小适配

若 baseline 脚本只支持固定输出路径，可在 Idea1 工作副本中增加：

```text
--output_dir
--pseudo_subdir
--checkpoint
--progress_root
```

但必须满足：

- 默认值仍保持 baseline 行为；
- 不改变 threshold；
- 不改变标签定义；
- 不改变空间映射；
- 不改变 loss；
- diff 中能够清楚看到只增加路径参数；
- 对相同输入与 checkpoint，baseline snapshot 和 Idea1 wrapper 输出逐文件一致。

---

# 7. Baseline 与 Idea1 数据目录隔离

## 7.1 原始 frozen baseline

原始 baseline 保持：

```text
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/
└── pseudo_student/tri_train/

/storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/
```

Idea1 不修改这些目录。

## 7.2 Idea1 独立 processed 根目录

```text
$PROCESSED_ROOT/<dataset>/fold_0/
├── native_npy/
├── teacher_npy/
├── student_npy/
├── prompts/
├── meta/
├── rounds/
├── pseudo_teacher/
└── pseudo_student/
```

## 7.3 共享输入数据

```text
native_npy
teacher_npy
student_npy
prompts
meta
```

必须与 baseline 对齐。

推荐两种模式：

### 正式最安全模式

完整复制到 Idea1 根目录：

```text
优点：完全独立，不可能写穿到 baseline
缺点：占用更多磁盘
```

### 存储节省模式

只读软链接：

```text
优点：节省空间
缺点：实现必须保证任何脚本都不向链接目录写文件
```

建议：

```text
large immutable arrays：可使用只读软链接
small metadata/prompts：复制并保存 SHA256
method outputs：绝不使用软链接，必须为 Idea1 实体目录
```

---

# 8. 方法伪标签目录

沿用此前 IdeaX 命名规则。

## 8.1 Box-only baseline reference

```text
$PROCESSED_ROOT/<dataset>/fold_0/
├── pseudo_student/
│   └── tri_train_boxonly/
└── pseudo_teacher/
    └── tri_train_boxonly/
```

来源：

```text
原始 frozen baseline 的 tri_train
```

建议复制或只读链接，并保存 hash。

其作用：

- 伪标签层面对照；
- 必要时在 Idea1 根目录重跑 student box-only；
- 检查 Idea1 是否真正只改变指定样本和 teacher。

## 8.2 Idea1 方法监督

```text
$PROCESSED_ROOT/<dataset>/fold_0/
├── pseudo_student/
│   └── tri_train_${METHOD}/
└── pseudo_teacher/
    └── tri_train_${METHOD}/
```

其中：

```text
tri_train_boxonly：
  全部 train 样本都是原始 MedSAM Box 三值伪标签

tri_train_${METHOD}：
  Full 样本是 GT
  剩余 Box 样本是最终微调 MedSAM 三值伪标签
```

## 8.3 明确禁止的操作

禁止：

```text
将 Full GT 写入 tri_train_boxonly
将 Full GT 写入原始 baseline tri_train
将 Idea1 pseudo 写回 baseline processed root
删除 baseline pseudo 后重命名 Idea1
用同一目录轮流生成 boxonly 和 Idea1
```

---

# 9. “Full GT 覆盖”的准确含义

不再使用容易误解的说法：

```text
覆盖 baseline tri_train
```

正式表述改为：

> **组装 Idea1 方法专属 hybrid supervision。**

## 9.1 组装过程

先创建一个全新的空目录：

```text
pseudo_student/tri_train_${METHOD}/
```

然后：

### 对 Box 样本

```text
复制最终微调 MedSAM 生成的 student-space tri pseudo
```

### 对 Full 样本

```text
复制 student_npy/gts 中对应的 GT
```

最终：

```text
Full → {0, 1..K}
Box  → {0, 1..K, 255}
```

## 9.2 为什么仍称为一个统一监督目录

frozen baseline student loader 默认读取：

```text
pseudo_student/tri_train
```

为了不修改 loader 的标签逻辑，Idea1 使用 `student_view` 将：

```text
pseudo_student/tri_train
```

软链接到：

```text
pseudo_student/tri_train_${METHOD}
```

注意：这个链接存在于 Idea1 的方法 view 中，不存在于 frozen baseline 原目录。

---

# 10. Student view 设计

## 10.1 目的

不同方法都向现有 student 脚本暴露同样的 fold 接口：

```text
fold_root/
├── native_npy/
├── teacher_npy/
├── student_npy/
├── prompts/
├── meta/
└── pseudo_student/
    └── tri_train/
```

但 `tri_train` 实际指向不同方法目录。

## 10.2 Box-only view

```text
$VIEW_ROOT/boxonly/<dataset>/fold_0/
├── native_npy   -> $PROCESSED_ROOT/<dataset>/fold_0/native_npy
├── teacher_npy  -> ...
├── student_npy  -> ...
├── prompts      -> ...
├── meta         -> ...
└── pseudo_student/
    └── tri_train
        -> $PROCESSED_ROOT/<dataset>/fold_0/pseudo_student/tri_train_boxonly
```

## 10.3 Idea1 view

```text
$VIEW_ROOT/${METHOD}/<dataset>/fold_0/
├── native_npy   -> $PROCESSED_ROOT/<dataset>/fold_0/native_npy
├── teacher_npy  -> ...
├── student_npy  -> ...
├── prompts      -> ...
├── meta         -> ...
└── pseudo_student/
    └── tri_train
        -> $PROCESSED_ROOT/<dataset>/fold_0/pseudo_student/tri_train_${METHOD}
```

## 10.4 Upper

Upper 可以继续使用 frozen upper 结果作为参考。

如果需要在 Idea1 根目录自包含重跑：

```text
$VIEW_ROOT/upper/<dataset>/fold_0/
```

其中 student 读取 GT，不使用 pseudo。

## 10.5 直接比较

student 训练只需替换 `--fold_root`：

```text
Box-only rerun:
  --fold_root $VIEW_ROOT/boxonly/<dataset>/fold_0

Idea1:
  --fold_root $VIEW_ROOT/${METHOD}/<dataset>/fold_0
```

student 代码、mode、loss 和超参数保持一致。

---

# 11. Baseline Contract Audit

在编写和运行 Idea1 前必须完成。

## 11.1 代码检查

确认 baseline snapshot 中：

```text
generate_prompts.py
generate_pseudo_labels.py
train_student.py
infer_student.py
eval_2d.py
eval_3d.py
```

与冻结版本一致。

## 11.2 三值伪标签检查

必须从实际代码确认：

```text
threshold = 0.5
box 外 = 0
box 内 p >= 0.5 = class_id
box 内 p < 0.5 = 255
多类冲突 = 255
3D 空切片 = 全 0
```

## 11.3 数据检查

```text
manifest train/test 无交集
prompt 只来自 train
无 test prompt
无 test pseudo
teacher/student shape 正确
3D case_id 和 slice_idx 正确
3D spacing_zyx 正确
geometry_meta 可用
```

## 11.4 输出

```text
$IDEA_ROOT/baseline_reference/
├── baseline_lineage.json
├── baseline_contract_audit.json
├── baseline_code_sha256.txt
├── baseline_data_sha256.txt
├── baseline_metrics/
└── upper_metrics/
```

所有硬断言通过后才能进入 Round 0。

---

# 12. 总体迭代流程

```text
Baseline code snapshot
        ↓
Idea1 working copy
        ↓
Baseline contract audit
        ↓
Round 0 random Full
        ↓
Build Full box-mask pairs
        ↓
Fine-tune mask decoder
        ↓
Score remaining Box pool
        ↓
Select hard Full for next round
        ↓
Repeat until budget or stop
        ↓
Final MedSAM
        ↓
Generate final Box tri pseudo
        ↓
Assemble tri_train_${METHOD}
        ↓
Build student view
        ↓
Train / infer / evaluate student
        ↓
Compare Baseline / Idea1 / Upper
```

---

# 13. Round 0

## 13.1 2D

从 train split 的有前景图像中随机选择：

```text
5 images
seed = 2026
```

一张图被选为 Full 后，该图所有病灶均为 Full。

## 13.2 3D

从 train cases 中随机选择：

```text
1 case
seed = 2026
```

一个 case 被选为 Full 后，该病例全部 slice、全部类别均为 Full。

## 13.3 必须保存

```text
random_seed.json
selected_new_ids.json
cumulative_full_ids.json
remaining_box_ids.json
selection_audit.json
```

---

# 14. 2D 困难样本筛选

## 14.1 评分对象

```text
train split
且
不在累计 Full 集中
```

## 14.2 每个实例

当前 MedSAM：

```text
image + baseline tight box
→ sigmoid probability
→ threshold 0.5
→ binary prediction
```

与对应 GT component 计算 IoU。

预测为空、GT 存在：

```text
IoU = 0
```

## 14.3 图像困难度

一张图有 \(M_i\) 个实例：

\[
S_i^{2D}
=
\frac{1}{M_i}
\sum_{j=1}^{M_i}IoU_{ij}
\]

## 14.4 候选与排序

候选：

```text
image_macro_iou < 0.5
```

排序：

```python
(
    image_macro_iou,
    -empty_instance_count,
    image_id,
)
```

## 14.5 每轮数量

```text
add_per_round = 5
max_full_images = 20
```

实际新增：

\[
k_r=\min(5,\ 20-|F_r|,\ |H_r|)
\]

停止：

```text
Full 达到20
无 IoU<0.5 候选
Box pool 为空
```

---

# 15. 3D 困难病例筛选

## 15.1 标注预算

使用 train case 数：

\[
B_{3D}
=
\max
\left(
1,\,
\min
\left(
5,\,
\lceil0.05N_{\text{train cases}}\rceil
\right)
\right)
\]

## 15.2 重建病例预测

对每个剩余 Box case：

```text
按 slice_idx 排序
→ 当前 MedSAM 对每个 box 推理
→ threshold 0.5
→ 映射回 native slice
→ 按 class_id 合并
→ 重建 3D volume
```

## 15.3 分器官 3D IoU

对 GT 中存在的类别 \(k\)：

\[
IoU_{c,k}^{3D}
=
\frac{|P_{c,k}\cap G_{c,k}|}
{|P_{c,k}\cup G_{c,k}|}
\]

```text
GT 有、prediction 空：0
GT 无：不参与 macro
双方空：不参与
```

## 15.4 病例主困难度

\[
S_c^{macro}
=
\frac{1}{|\mathcal K_c|}
\sum_{k\in\mathcal K_c}
IoU_{c,k}^{3D}
\]

## 15.5 辅助指标

完全漏检类别数：

\[
E_c
=
\sum_{k\in\mathcal K_c}
\mathbf 1[|P_{c,k}|=0]
\]

最差 20% slice-class IoU：

\[
T_c=\operatorname{mean}(\text{lowest 20\% slice-class IoUs})
\]

## 15.6 候选与排序

候选：

```text
case_macro_3d_iou < 0.5
或
empty_prediction_classes > 0
```

排序：

```python
(
    case_macro_3d_iou,
    -empty_prediction_classes,
    worst20_slice_class_iou,
    case_id,
)
```

每轮：

```text
add 1 case
```

停止：

```text
Full case 达到 B3D
无困难候选
Box case pool 为空
```

---

# 16. Full GT 到 MedSAM 微调对

## 16.1 训练对

每个训练对：

```text
teacher image
+ one tight box
+ one binary component mask
```

## 16.2 2D

```text
每图
→ 每个类别/连通病灶
→ 一个 box-mask pair
```

## 16.3 3D

```text
每个 Full case
→ 每个有前景 slice
→ 每个 class
→ 每个 connected component
→ 一个 box-mask pair
```

空 slice 不进入 MedSAM 微调。

## 16.4 Prompt 来源

优先复用 baseline：

```text
prompts/prompts_train.json
```

不能重新引入 margin、jitter 或 box expansion。

## 16.5 空间

```text
native GT component
→ geometry_meta
→ nearest-neighbor
→ teacher binary mask
```

---

# 17. MedSAM 微调

## 17.1 参数冻结

```text
image_encoder: frozen
prompt_encoder: frozen
mask_decoder: trainable
```

## 17.2 初始化

```text
Round 0：
  original medsam_vit_b.pth

Round r>0：
  previous round medsam_ft.pth
```

每轮训练累计 Full 集。

## 17.3 Loss

\[
\mathcal L
=
\mathcal L_{\mathrm{Dice}}
+
\mathcal L_{\mathrm{BCEWithLogits}}
\]

权重：

```text
dice = 1.0
bce = 1.0
```

## 17.4 第一版锁定超参数

| 参数 | 2D | 3D |
|---|---:|---:|
| optimizer | AdamW | AdamW |
| lr | 1e-5 | 1e-5 |
| weight_decay | 0.01 | 0.01 |
| batch_size | 2 | 2 |
| num_workers | 4 | 4 |
| incremental steps/round | 300 | 1000 |
| AMP | on | on |
| grad clip | 1.0 | 1.0 |
| scheduler | none | none |
| augmentation | none | none |
| seed | 2026 | 2026 |
| pseudo threshold | 0.5 | 0.5 |

这些值在正式运行前仅做 smoke test：

```text
显存
梯度
loss 是否有限
输出 shape
checkpoint 可读
```

不使用 test 指标调参。

## 17.5 3D 采样

```text
先均匀采 case
再在 case 内均匀采 pair
```

避免 slice 多的病例支配训练。

---

# 18. 每轮诊断与概率图

## 18.1 所有 Box 样本保存数值

```text
per_instance_metrics.csv
per_image_metrics.csv
per_case_metrics.csv
diagnosis_summary.json
```

## 18.2 概率图

保存 sigmoid probability，不是二值结果。

2D 困难候选保存：

```text
raw npz
heatmap
overlay
```

3D：

```text
最终选中的困难 case：所有前景 slice
其他候选：最差5张 slice
```

## 18.3 分析量

欠激活：

\[
R_{\text{under}}
=
\frac{|\{x\in GT:0.1\le p(x)<0.5\}|}{|GT|}
\]

完全未识别：

\[
R_{\text{miss}}
=
\frac{|\{x\in GT:p(x)<0.1\}|}{|GT|}
\]

过激活：

\[
R_{\text{over}}
=
\frac{|\{x\in B\setminus GT:p(x)\ge0.5\}|}
{|B\setminus GT|}
\]

仅用于分析，不改变阈值。

---

# 19. 最终三值伪标签

## 19.1 Box 样本

沿用 baseline：

\[
Y^{tri}(x)=
\begin{cases}
0, & x\notin B\\
c, & x\in B,\ p(x)\ge0.5\\
255, & x\in B,\ p(x)<0.5
\end{cases}
\]

多类冲突：

```text
255
```

3D 空切片：

```text
全 0
```

## 19.2 Full 样本

不使用 MedSAM 伪标签：

```text
直接复制 student-space GT
```

## 19.3 生成顺序

```text
07_generate_method_tri_pseudo.py
  只为 Box pool 生成 pseudo

08_assemble_method_supervision.py
  创建新的 tri_train_${METHOD}
  写入 Box pseudo
  写入 Full GT
```

注意：脚本名称不再使用 `overlay baseline`，避免误解。

---

# 20. 每轮输出目录

```text
$PROCESSED_ROOT/<dataset>/fold_0/
└── rounds/
    └── ${METHOD}/
        ├── round_00/
        │   ├── selection/
        │   ├── finetune_pairs/
        │   ├── teacher/
        │   ├── diagnosis/
        │   └── round_state.json
        ├── round_01/
        └── ...
```

每轮至少保存：

```text
input checkpoint
output checkpoint
新增 Full IDs
累计 Full IDs
剩余 Box IDs
训练 steps
训练日志
困难评分
停止判断
stage_time
```

---

# 21. Student 输出目录

```text
$WORK_ROOT/
├── boxonly_rerun/
│   └── <dataset>/fold_0/
├── ${METHOD}/
│   └── <dataset>/fold_0/
└── upper_rerun/
    └── <dataset>/fold_0/
```

其中正式 Idea1：

```text
$WORK_ROOT/${METHOD}/<dataset>/fold_0/
├── last.pth
├── train_log.csv
├── pred_test/
├── eval_2d/ 或 eval_3d/
├── run_train_idea1.log
├── run_infer_idea1.log
├── run_eval_idea1.log
├── stage_time_train_idea1.json
├── stage_time_infer_idea1.json
└── stage_time_eval_idea1.json
```

原始 frozen baseline 和 upper 结果仍保留在原仓库。

---

# 22. Student 训练超参数

完全继承 baseline：

```text
epochs = 50
lr = 1e-4
weight_decay = 0.05
freeze_encoder_epochs = 10
amp = on
deep_supervision = on
pretrained = vmamba_tiny_e292.pth
```

| Dataset | Batch size | Workers |
|---|---:|---:|
| btcv | 1 | 0 |
| synapse | 1 | 0 |
| acdc | 2 | 2 |
| prostate158 | 2 | 2 |
| kvasirseg | 8 | 4 |
| cvc_clinicdb | 8 | 4 |
| tn3k | 8 | 4 |
| tg3k | 8 | 4 |
| ddti | 8 | 4 |
| otu_2d | 8 | 4 |
| ph2 | 8 | 4 |

Idea1 student 必须从 baseline 相同 pretrained 初始化，不从 baseline `last.pth` 继续。

---

# 23. 推理与评估

## 23.1 2D

```text
Dice
IoU
MAE（kvasirseg、cvc_clinicdb）
```

## 23.2 3D

```text
DSC
HD95(mm)
ASSD(mm)
```

必须：

```text
恢复 native slice
按 slice_idx 重建
使用真实 spacing_zyx
不允许 fallback (1,1,1)
逐器官统计
```

一方为空：

```text
Dice = 0
HD95/ASSD = NaN
单独统计 empty-prediction failure
```

---

# 24. 公平比较方式

## 24.1 Student 最终比较

```text
原始 frozen Baseline
Idea1 final
原始 frozen Upper
```

三者：

```text
相同 split
相同 student
相同 pretrained
相同 crop
相同 epochs
相同 eval
```

## 24.2 伪标签比较

必须分成两种口径。

### 相同 Box 子集

仅在 Idea1 最终仍属于 Box 的样本上比较：

```text
tri_train_boxonly
vs
final teacher Box pseudo
```

这样衡量 teacher 微调是否改善剩余 Box。

### 全训练监督

比较：

```text
Baseline：全部 Box pseudo
Idea1：Full GT + Box pseudo
```

这个口径衡量最终 student 监督质量，但其中包含 Full GT 的直接收益。

不得把二者混成同一个“teacher pseudo 提升”。

---

# 25. 配置文件

```yaml
method:
  method_id: idea1_hard_full_medsam_ft
  parent_method: frozen_baseline_v1

paths:
  frozen_medsam_repo: /storage/baiyuting/data/MedSAM-main
  frozen_swin_repo: /storage/baiyuting/data/Swin-UMamba-main
  idea_root: /storage/baiyuting/data/out_data_idea1
  processed_root: /storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed
  view_root: /storage/baiyuting/data/out_data_idea1/student_views
  work_root: /storage/baiyuting/data/out_data_idea1/Swin-UMamba-main/work_dir
  base_medsam_checkpoint: /storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth

selection:
  seed: 2026
  difficulty_iou_threshold: 0.5

  two_d:
    round0_random_images: 5
    add_per_round: 5
    max_full_images: 20

  three_d:
    round0_random_cases: 1
    add_per_round: 1
    max_full_cases_absolute: 5
    max_full_cases_ratio: 0.05
    ratio_rounding: ceil
    worst_slice_fraction: 0.20

medsam_finetune:
  trainable: mask_decoder
  optimizer: adamw
  lr: 1.0e-5
  weight_decay: 0.01
  batch_size: 2
  num_workers: 4
  amp: true
  grad_clip_norm: 1.0
  dice_weight: 1.0
  bce_weight: 1.0
  scheduler: none
  augmentation: none
  steps_2d_per_round: 300
  steps_3d_per_round: 1000
  resume_previous_round: true

pseudo:
  contract: baseline_single_threshold
  threshold: 0.5
  outside_box_value: 0
  unknown_value: 255
  conflict_value: 255
  empty_3d_slice_value: 0

student:
  inherit_frozen_baseline_hyperparameters: true
```

正式运行保存解析后的：

```text
config_resolved.json
```

---

# 26. 脚本职责

## 26.1 `00_audit_baseline_contract.py`

```text
代码 hash
threshold
tri contract
prompt
split
geometry
spacing
baseline结果位置
```

## 26.2 `01_init_idea1_workspace.py`

```text
创建 Idea1 processed
复制/链接共享输入
复制 boxonly pseudo reference
计算3D budget
写 lineage
```

## 26.3 `02_select_round0_random.py`

```text
2D随机5张
3D随机1 case
```

## 26.4 `03_build_full_finetune_pairs.py`

```text
Full GT
→ per-component box-mask pairs
```

## 26.5 `04_finetune_medsam_mask_decoder.py`

```text
只训练 mask decoder
Dice+BCE
```

## 26.6 `05_score_remaining_box_pool.py`

```text
current MedSAM
→ probability
→ IoU
→ 2D image score
→ 3D case score
```

## 26.7 `06_select_next_hard_samples.py`

```text
更新 Full/Box 集
判断停止
```

## 26.8 `07_generate_method_tri_pseudo.py`

```text
只对最终 Box pool
使用 baseline 单阈值逻辑
输出方法临时 Box pseudo
```

## 26.9 `08_assemble_method_supervision.py`

```text
创建 tri_train_${METHOD}
Box写 pseudo
Full写 GT
不读取或修改 baseline 输出目录
```

## 26.10 `09_build_student_view.py`

```text
构建 boxonly view
构建 Idea1 view
校验链接目标
```

## 26.11 `10_validate_method_data.py`

```text
数量
值域
255
Full/Box
split
空间
3D case一致性
```

---

# 27. 必须通过的数据断言

```text
[ ] baseline snapshot 与 frozen commit 对齐
[ ] Idea1 工作代码来自 baseline snapshot
[ ] baseline→Idea1 diff 只含允许改动
[ ] original baseline 目录没有任何修改
[ ] tri_train_boxonly 保持不变
[ ] tri_train_${METHOD} 为独立实体目录
[ ] Round 0 seed=2026
[ ] 2D Full 单位为 image
[ ] 3D Full 单位为 case
[ ] Full∩Box=空
[ ] Full∪Box=train
[ ] test 不参与筛选、微调和 pseudo
[ ] threshold=0.5
[ ] MedSAM 只训练 mask decoder
[ ] loss 只有 Dice+BCE
[ ] Full label 无255
[ ] Box 255 只在 box union 内
[ ] student view 指向正确方法目录
[ ] baseline student结果未覆盖
[ ] upper student结果未覆盖
[ ] 3D spacing真实有效
```

---

# 28. 执行顺序

## Phase 0：整理代码

```text
记录 frozen commits
→ 建 baseline snapshot
→ hash
→ 设置只读
→ 复制 Idea1 工作副本
```

## Phase 1：Baseline audit

```text
伪标签规则
prompt
split
geometry
spacing
student/eval
```

## Phase 2：创建 Idea1 数据工作区

```text
复制/链接共享输入
复制 boxonly pseudo
创建 method rounds
创建 lineage/config
```

## Phase 3：2D smoke test

```text
Round0随机5张
→ build pairs
→ finetune
→ score
→ select
→ assemble method labels
→ validate
```

## Phase 4：3D smoke test

```text
Round0随机1 case
→ case budget
→ pair build
→ finetune
→ volume score
→ hard selection
→ spacing audit
```

## Phase 5：完整 teacher 迭代

```text
11 datasets
Round0
→ finetune
→ score
→ select
→ stop
```

## Phase 6：最终监督

```text
final Box pseudo
→ tri_train_${METHOD}
→ Full GT
→ method validation
→ student view
```

## Phase 7：Student

```text
train
→ infer
→ eval
```

## Phase 8：汇总

```text
Baseline
Idea1
Upper
annotation budget
teacher Box-subset gain
student gain
time
```

---

# 29. 结果目录

```text
$SUMMARY_ROOT/
├── baseline_reference_metrics/
├── upper_reference_metrics/
├── idea1_summary_metrics_2d.csv
├── idea1_summary_metrics_3d.csv
├── idea1_summary_metrics_3d_per_organ.csv
├── idea1_annotation_budget.csv
├── idea1_round_diagnosis.csv
├── idea1_box_subset_pseudo_comparison.csv
├── idea1_time_summary.csv
└── method_lineage_summary.json
```

---

# 30. 常见疑问

## 30.1 Full GT 是否会覆盖 baseline 伪标签？

不会。

Full GT 只写入：

```text
tri_train_${METHOD}
```

以下始终保持：

```text
原始 baseline tri_train
Idea1 tri_train_boxonly
baseline student结果
upper student结果
```

## 30.2 如何与 baseline 对比？

Student 结果：

```text
frozen baseline eval
vs
Idea1 eval
```

伪标签结果：

```text
tri_train_boxonly
vs
tri_train_${METHOD}
```

其中 teacher 改善应在相同 Box 子集上比较。

## 30.3 为什么不直接修改 baseline repo？

因为这会破坏方法归因和复现。Idea1 必须能通过代码 diff 清楚说明新增了什么。

## 30.4 是否需要重新预处理？

通常不需要。Idea1 继承 baseline 的：

```text
native_npy
teacher_npy
student_npy
prompts
meta
```

只有 baseline audit 失败时才重新处理。

## 30.5 Student loader 是否必须修改？

不需要。通过 `student_view` 将方法目录映射成 loader 期望的：

```text
pseudo_student/tri_train
```

---

# 31. 最终流程图

```text
Original frozen baseline repositories/results
                    ↓
        Create baseline_v1_snapshot
                    ↓
        Copy to Idea1 working code
                    ↓
        Baseline Contract Audit
                    ↓
       Independent Idea1 processed root
                    ↓
Round 0:
  2D random 5 images
  3D random 1 case
                    ↓
Full GT + baseline tight boxes
                    ↓
Fine-tune MedSAM mask decoder
Dice + BCE
                    ↓
Score remaining train Box pool
                    ↓
2D:
  image macro IoU < 0.5
  add up to 5 / round
  max 20

3D:
  class-macro 3D IoU < 0.5
  or missed class
  add 1 case / round
  max min(5, ceil(5% train cases))
                    ↓
Final fine-tuned MedSAM
                    ↓
Generate Box pseudo with baseline threshold 0.5
                    ↓
Assemble NEW tri_train_${METHOD}
  Full → GT
  Box  → tri pseudo
                    ↓
Build Idea1 student_view
                    ↓
Original Swin-UMamba settings
                    ↓
Native-space test evaluation
                    ↓
Baseline / Idea1 / Upper comparison
```

---

# 32. 最终结论

本方案的代码和数据关系必须理解为：

```text
baseline snapshot
    └── Idea1 code copy + new scripts

baseline data contract
    └── Idea1 independent data workspace

baseline tri pseudo
    ├── original frozen tri_train
    └── Idea1 reference tri_train_boxonly

Idea1 final supervision
    └── tri_train_${METHOD}
        ├── Full GT
        └── remaining Box tri pseudo
```

因此：

> **Idea1 是在 baseline 代码副本上增加功能，而不是在 baseline 正式目录上原地改写；Full GT 只参与组装 Idea1 方法专属监督目录，baseline 始终完整保留，可随时独立复现和比较。**
