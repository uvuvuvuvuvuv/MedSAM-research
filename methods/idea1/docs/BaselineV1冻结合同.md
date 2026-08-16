# MedSAM→Swin-UMamba 正式实验流程（冻结版）

> 适用范围：当前正式 baseline 主线，仅保留
> 3D：`btcv, synapse, acdc, prostate158`
> 2D：`kvasirseg, cvc_clinicdb, tn3k, tg3k, ddti, otu_2d, ph2`
> 当前正式学生端只保留两组：`baseline` 与 `upper`。
> 当前 baseline 目录结构已经固定：**只允许覆盖重写，不允许自动新建新分支目录**。
> 若为 debug，允许手动新建目录，但必须**手动指定**，不能让批处理脚本自动生成。

---

## 0. 实验环境与可复现性记录

当前正式硬件环境：
- GPU：4 × NVIDIA RTX 4090
- CUDA：12.8
- MedSAM 环境：medsam310
- Swin-UMamba 环境：swin_umamba

正式实验要求：
- 所有训练、推理、评估必须记录 conda 环境名；
- 所有训练日志必须保存命令行参数；
- 正式结果必须能追溯到对应 processed / prompt / pseudo / checkpoint / eval 脚本版本。

当前环境和目录：

#### MedSAM 侧

- 环境：`medsam310`
- 操作目录：`/storage/baiyuting/data/MedSAM-main`

#### Swin-UMamba 侧

- 环境：`swin_umamba`
- 操作目录：`/storage/baiyuting/data/Swin-UMamba-main`

正式执行命令必须显式激活对应环境后再运行。

## 1. 本版总定义

### 1.1 当前正式主线只保留两种模式
- `baseline`：弱监督主线，训练标签为 `pseudo_student/tri_train`
- `upper`：全监督上界，训练标签为 `student_gt`

### 1.3 当前这版真正回答的问题
1. 在严格无泄漏、三空间显式建模、native-space tight box、train-only tri pseudo 的设定下，`tri pseudo` 能否训练出可用 student。
2. 在同一 student、同一 crop、同一评估口径下，`upper` 相对 `baseline` 的差距有多大。

---

## 2. 最重要的硬规则

### 2.1 目录规则

当前 baseline 目录结构已经固定，所有正式运行只能覆盖原目录。

正式目录结构以 **第 7 节“全局目录结构合同”** 为唯一准则。

允许执行：
- 删除旧结果
- 在固定路径下重新生成同名文件和同名目录

禁止执行：
- 自动新建 `fold_0_gpu2`
- 自动新建 `fold_0_debug`
- 自动新建 `baseline_v2`
- 自动新建时间戳目录
- 自动新建任何“为了区分实验”的额外层级

### 2.2 debug 例外
如果只是 debug：
- 可以新建目录
- 但必须由你**手动输入目录名**
- 不能写进正式批处理脚本里自动生成
- debug 结果不得混入正式 baseline 统计

### 2.3 正式批处理的工作方式
正式训练、推理、评估脚本都遵循：
- 先 `rm -rf` 原目标目录
- 再 `mkdir -p` 同一个固定目录
- 最后把新结果写回原位置

也就是说：**是覆盖重写，不是并行保留多个版本**。

---

## 3. 三套空间必须严格区分

### 3.1 Native Space
原始图像 / 原始 GT 空间。
这是最终正式评估的唯一真空间。

### 3.2 Teacher Space
MedSAM 输入空间，固定为：

- `1024 × 1024 × 3`

### 3.3 Student Space
Swin-UMamba 输入空间，按数据集固定尺寸。

### 3.4 三空间规则
任何 bbox、伪标签、预测结果的跨空间变换都必须依赖：

- `geometry_meta.json`

禁止：
- 手写 resize 推导
- 在下游脚本里凭经验反推坐标
- 直接把 student-space 预测拿去做 3D 正式指标

---

## 4. 当前正式数据集范围

### 4.1 3D 数据集
- `btcv`
- `synapse`
- `acdc`
- `prostate158`

### 4.2 2D 数据集
- `kvasirseg`
- `cvc_clinicdb`
- `tn3k`
- `tg3k`
- `ddti`
- `otu_2d`
- `ph2`

你这次已经明确收缩，当前正式批处理就只保留上面这 11 个数据集。

---

## 5. 当前固定的 student crop

### 5.1 3D
- `btcv`: `512 × 512`
- `synapse`: `512 × 512`
- `acdc`: `320 × 320`
- `prostate158`: `320 × 320`

### 5.2 2D
- `kvasirseg`: `352 × 352`
- `cvc_clinicdb`: `352 × 352`
- `tn3k`: `256 × 256`
- `tg3k`: `256 × 256`
- `ddti`: `256 × 256`
- `otu_2d`: `256 × 256`
- `ph2`: `256 × 256`

### 5.3 crop 锁定规则
一旦某个数据集 crop 已用于当前 baseline 周期：
- 不允许中途修改
- 如果以后要改，必须作为**新 crop 分支实验**
- 并重跑所有 student-side 相关链路

但当前你要求的是：
- baseline 文件结构固定
- 正式脚本只覆盖重写

所以这版默认：**不改 crop，不开新 crop 分支**。

---

## 6. 数据划分协议

### 6.1 3D 数据
#### `btcv`
- patient / case / volume-level split
- 禁止 slice-level random split

#### `synapse`
- benchmark split
- 同一 case 的所有 slice 必须在同一 split

#### `acdc`
- official challenge split

#### `prostate158`
- official fixed patient list

### 6.2 2D 数据
#### `kvasirseg`
- fixed widely-used split

#### `cvc_clinicdb`
- fixed widely-used split

#### `tn3k`
- official split

#### `tg3k`
- official `train/val` 映射为当前 `train/test`

#### `ddti`
- stratified split，按良恶性分层

#### `otu_2d`
- internal pre-defined split

#### `ph2`
- stratified split，按病理类别分层

### 6.3 无泄漏原则
全流程严格禁止：
- 生成并使用 `test pseudo`
- 用 test split 调阈值
- 用 test split 选 checkpoint
- 用 test prompt 反哺 student 训练
- 3D 数据做 slice-level 洗牌

---

### 6.4 Test Split Immutable Contract

test split 是只读评估资源。

严格禁止：
- 修改 test GT
- 对 test GT 做 tiny component filter
- 对 test GT 做 morphology
- 生成 test prompt
- 生成 test pseudo
- 用 test split 调阈值、选 checkpoint 或选择超参数

允许：
- 对 test image 做与 train 一致的确定性输入预处理
- 对 test GT 做 nearest-neighbor 几何映射，仅用于 evaluation alignment
- 在 evaluation 阶段读取 test GT 计算指标

一句话原则：
train 可以构造监督，test 只能用于最终评价。

## 7. 全局目录结构合同

本节是当前正式 baseline 的唯一目录结构定义。
后续所有 Stage 0 / Stage 1 / Stage 1.5 / Stage 2 / student training / inference / evaluation 均必须与本节对齐。

当前正式 baseline 只覆盖 11 个数据集：

```text
3D:
btcv, synapse, acdc, prostate158

2D:
kvasirseg, cvc_clinicdb, tn3k, tg3k, ddti, otu_2d, ph2
```

### 7.1 全局根目录

#### MedSAM 侧

```
/storage/baiyuting/data/MedSAM-main/
├── data/
│   ├── raw/
│   ├── organized/
│   ├── processed/
│   └── progress/
├── work_dir/
│   └── MedSAM/
├── utils/
│   └── processed.py
├── generate_prompts.py
└── generate_pseudo_labels.py
```

含义：

```
data/raw/          原始数据来源
data/organized/    统一命名、固定 split、raw mask 审计后的数据
data/processed/    三空间预处理、prompt、pseudo 的正式数据根目录
data/progress/     数据准备阶段进度记录
work_dir/MedSAM/   MedSAM checkpoint 目录
```

#### Swin-UMamba 侧

```
/storage/baiyuting/data/Swin-UMamba-main/
├── pipeline/
│   ├── train_student.py
│   ├── infer_student.py
│   ├── eval_2d.py
│   ├── eval_3d.py
│   ├── test_student_patch_dataset.py
│   └── screen_cases.py
├── data/
│   └── pretrained/
└── work_dir/
	├── baseline/
	└── upper/
```

含义：

```
pipeline/          student 训练、推理、评估、检查脚本
data/pretrained/   Swin-UMamba / VMamba 预训练权重
work_dir/baseline/ baseline 训练、推理、评估输出
work_dir/upper/    upper 训练、推理、评估输出
```

### 7.2 raw 数据目录

```
MedSAM-main/data/raw/<dataset>/
```

用途：

```
保存官方或当前正式使用的原始图像与原始标签。
```

约束：

- raw 数据只读；
- 不在 raw 层直接改标签；
- 所有标签修复必须通过 organized / processed 的显式脚本完成；
- raw 层不参与训练。

------

### 7.3 organized 数据目录

```
MedSAM-main/data/organized/<dataset>/
├── meta/
│   ├── raw_manifest.json
│   ├── split_meta.json
│   ├── sanity_check.csv
│   ├── label_source_audit.json
│   ├── label_source_audit.csv
│   └── label_source_audit_summary.json
├── train/
│   └── cases/<case_id>/
└── test/
    └── cases/<case_id>/
```

用途：

```
organized 层负责固定 split、统一命名、2D raw mask 审计与二值化规则落地。
```

正式规则：

- `tn3k / tg3k / kvasirseg` 的 JPEG / 灰度 mask 使用 `>=128` 二值化；
- `cvc_clinicdb / ddti / otu_2d / ph2` 保持纯二值原样；
- organized 层不做 test GT 的语义清洗；
- test GT 只作为 evaluation asset 保留。

------

### 7.4 processed 数据目录

```
MedSAM-main/data/processed/<dataset>/fold_0/
├── native_npy/
│   └── gts/
├── teacher_npy/
│   ├── imgs/
│   └── gts/
├── student_npy/
│   ├── imgs/
│   └── gts/
├── prompts/
│   └── prompts_train.json
├── pseudo_teacher/
│   └── tri_train/
├── pseudo_student/
│   └── tri_train/
└── meta/
    ├── manifest.json
    ├── split_meta.json
    ├── geometry_meta.json
    ├── label_meta.json
    ├── leakage_audit.json
    ├── preprocess_stats.json
    ├── tiny_gt_filter_records.json
    ├── stage_time_preprocess.json
    ├── stage_time_prompts.json
    └── stage_time_pseudo_train.json
```

#### 7.4.1 native_npy

```
native_npy/gts/
```

用途：

- 保存 native-space GT；
- prompt 的 tight bbox 只能从 native GT 生成；
- 禁止从 teacher/student resized GT 反推 bbox。

说明：

- 当前正式目录至少需要 `native_npy/gts/`；
- 如果后续脚本保存 native image，可增加 `native_npy/imgs/`，但当前 baseline 不强制要求。

#### 7.4.2 teacher_npy

```
teacher_npy/imgs/
teacher_npy/gts/
```

用途：

- `teacher_npy/imgs/`：MedSAM 输入图像，固定为 `1024×1024×3`；
- `teacher_npy/gts/`：teacher-space 标签，仅用于检查和空间对齐，不用于 student 训练。

#### 7.4.3 student_npy

```
student_npy/imgs/
student_npy/gts/
```

用途：

- `student_npy/imgs/`：Swin-UMamba 输入图像；
- `student_npy/gts/`：upper 使用的全监督标签，以及 test evaluation 使用的 GT 映射版本。

注意：

- baseline 训练不使用 `student_npy/gts/train`；
- baseline 训练只使用 `pseudo_student/tri_train`；
- test GT 只用于 evaluation，不用于 prompt / pseudo / training。

#### 7.4.4 prompts

```
prompts/prompts_train.json
```

正式规则：

- 只生成 train prompt；
- 不生成、不保存 `prompts_test.json`；
- 不保存 root-level `fold_0/prompts_train.json`；
- 不保存 root-level `fold_0/prompts_test.json`。

禁止：

```
prompts/prompts_test.json
fold_0/prompts_train.json
fold_0/prompts_test.json
```

#### 7.4.5 pseudo_teacher

```
pseudo_teacher/tri_train/
```

用途：

- 用于 teacher-space tri pseudo 保存、空间对齐检查和伪标签统计分析；
- 不直接作为 student 训练标签。

#### 7.4.6 pseudo_student

```
pseudo_student/tri_train/
```

用途：

- 保存 student-space tri pseudo；
- baseline 训练唯一合法标签来源。

标签定义：

```
0      background
1..K   foreground / class label
255    unknown / ignore
```

正式规则：

- 只生成 `tri_train/`；
- 不生成 `tri_test/`；
- 不保留 hard pseudo；
- 不保留 weight map；
- 当前正式 baseline 不生成 vis_train。
  如后续 debug 需要可视化，必须手动开启，并且不得进入 frozen baseline 统计。

#### 7.4.7 meta

```
meta/
├── manifest.json
├── split_meta.json
├── geometry_meta.json
├── label_meta.json
├── leakage_audit.json
├── preprocess_stats.json
├── tiny_gt_filter_records.json
├── stage_time_preprocess.json
├── stage_time_prompts.json
└── stage_time_pseudo_train.json
```

核心文件含义：

```
manifest.json                 所有样本路径、split、case_id、slice_idx 的索引
split_meta.json               数据划分、类别数、teacher/student target size 等
geometry_meta.json            native / teacher / student 空间映射合同
label_meta.json               标签类别、前景定义、二值/多类信息
leakage_audit.json            split 与泄漏检查记录
preprocess_stats.json         预处理统计
tiny_gt_filter_records.json   tiny component filter 记录，只允许出现 train 或空
stage_time_preprocess.json    Stage 1 计时
stage_time_prompts.json       Stage 1.5 计时
stage_time_pseudo_train.json  Stage 2 计时
```

强制要求：

- 所有路径查找必须依赖 `manifest.json`；
- 所有空间映射必须依赖 `geometry_meta.json`；
- 所有 split 判断必须依赖 `split_meta.json` 或 `manifest.json`；
- 禁止通过文件名猜 split；
- 禁止手写 resize 反推坐标。

------

### 7.5 进度、日志与计时目录

当前 baseline 将“进度记录”“日志记录”“正式计时”严格区分。

全链路统一原则：

```text
progress 只记录阶段状态；
run_*.log 只保存命令输出和错误信息；
stage_time_*.json 是唯一正式计时来源；
summary_time_11datasets_hdotm.csv 只从 stage_time_*.json 汇总。
```

#### 7.5.1 数据准备阶段进度目录

```
MedSAM-main/data/progress/
├── summary.csv
└── events.csv
```

用途：

- 记录 Stage 1 / Stage 1.5 / Stage 2 的运行进度；
- 即 preprocessing / prompt generation / pseudo-label generation；
- 只作为运行状态记录；
- 不参与训练、推理、评估；
- 不作为最终时间统计来源。

文件含义：

```
summary.csv   每个 dataset / stage 的完成状态
events.csv    每个 dataset / stage 的事件记录
```

#### 7.5.2 数据准备阶段正式计时目录

数据准备阶段的正式计时文件只写入：

```
MedSAM-main/data/processed/<dataset>/fold_0/meta/
├── stage_time_preprocess.json
├── stage_time_prompts.json
└── stage_time_pseudo_train.json
```

对应关系：

| Stage                           | 文件                           | 汇总列              |
| ------------------------------- | ------------------------------ | ------------------- |
| Stage 1 preprocessing           | `stage_time_preprocess.json`   | `preprocess_h.mm`   |
| Stage 1.5 prompt generation     | `stage_time_prompts.json`      | `prompts_h.mm`      |
| Stage 2 pseudo-label generation | `stage_time_pseudo_train.json` | `pseudo_train_h.mm` |

注意：

- `data/progress/summary.csv` 不作为正式计时来源；
- `data/progress/events.csv` 不作为正式计时来源；
- 正式时间统计只读取 `stage_time_*.json`。

#### 7.5.3 当前不固定任何可视化目录

当前 frozen baseline 不固定任何可视化输出目录。

以下目录均不作为当前正式合同：

```text
MedSAM-main/data/vis/
MedSAM-main/data/vis/pseudo/<dataset>/
MedSAM-main/data/vis/teacher/<dataset>/
MedSAM-main/data/vis/student/<dataset>/
Swin-UMamba-main/work_dir/vis/baseline/<dataset>/
Swin-UMamba-main/work_dir/vis/upper/<dataset>/
```

因此，当前 frozen baseline 不要求生成任何 image panel、preview、activation map、screen case 或 test prediction visualization。

#### 7.5.4 Student 训练、推理、评估日志与计时目录

student 侧不写入 `MedSAM-main/data/progress/`，也不写入 `MedSAM-main/data/vis/`。

baseline 固定写入：

```
Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/
├── run_train_baseline.log
├── run_infer_baseline.log
├── run_eval_baseline.log
├── stage_time_train_baseline.json
├── stage_time_infer_baseline.json
└── stage_time_eval_baseline.json
```

upper 固定写入：

```
Swin-UMamba-main/work_dir/upper/<dataset>/fold_0/
├── run_train_upper.log
├── run_infer_upper.log
├── run_eval_upper.log
├── stage_time_train_upper.json
├── stage_time_infer_upper.json
└── stage_time_eval_upper.json
```

日志文件用途：

```
run_*.log  保存命令行、stdout/stderr 和错误信息
```

计时文件用途：

```
stage_time_*.json  作为正式时间统计来源
```

注意：

- `run_*.log` 不作为正式时间统计来源；
- `train_log.csv` 不作为正式时间统计来源；
- file mtime 不作为正式时间统计来源。



### 7.6 Swin-UMamba 工作目录

```
Swin-UMamba-main/work_dir/
├── baseline/
│   └── <dataset>/fold_0/
└── upper/
    └── <dataset>/fold_0/
```

#### 7.6.1 baseline

```
Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/
├── last.pth
├── train_log.csv
├── pred_test/
├── eval_2d/ 或 eval_3d/
├── stage_time_train_baseline.json
├── stage_time_infer_baseline.json
├── stage_time_eval_baseline.json
├── run_train_baseline.log
├── run_infer_baseline.log
└── run_eval_baseline.log
```

用途：

- baseline 训练、推理、评估结果；
- baseline 训练标签为 `pseudo_student/tri_train`；
- `255` 作为 ignore。

#### 7.6.2 upper

```
Swin-UMamba-main/work_dir/upper/<dataset>/fold_0/
├── last.pth
├── train_log.csv
├── pred_test/
├── eval_2d/ 或 eval_3d/
├── stage_time_train_upper.json
├── stage_time_infer_upper.json
├── stage_time_eval_upper.json
├── run_train_upper.log
├── run_infer_upper.log
└── run_eval_upper.log
```

用途：

- upper 训练、推理、评估结果；
- upper 训练标签为 `student_npy/gts/train`；
- upper 用于衡量同一 student 架构下全监督上界。

#### 7.6.3 pred_test

```
pred_test/
```

用途：

- 保存 test split 的 student prediction；
- 由 `infer_student.py` 生成；
- 进入 `eval_2d.py` 或 `eval_3d.py`；
- 不作为训练输入。

#### 7.6.4 eval_2d / eval_3d

```
eval_2d/
eval_3d/
```

用途：

- 保存正式数值评估结果；
- 2D 指标：Dice / IoU / MAE；
- 3D 指标：DSC / HD95(mm) / ASSD(mm)。

规则：

- 2D evaluation 使用 test GT；
- 3D evaluation 必须恢复 native slice 尺寸，并按 `slice_idx` 重建 volume；
- 3D evaluation 必须使用真实 spacing；
- evaluation 不依赖任何 prompt 或 pseudo。

------

### 7.7 汇总目录

当前汇总文件写入：

```
Swin-UMamba-main/work_dir/
├── summary_metrics_2d.csv
├── summary_metrics_3d.csv
├── summary_metrics_3d_per_organ.csv
└── summary_time_11datasets_hdotm.csv
```

说明：

- 当前正式 baseline 为 11 个数据集；
- 新生成的正式时间表应使用 `summary_time_11datasets_hdotm.csv`；
- 旧版 `summary_time_12datasets_hdotm.csv` 只能作为历史记录，不再代表当前正式 baseline。

------

### 7.8 日志目录

正式日志与计时目录遵循 §7.5 和 §7.6。

当前 frozen baseline 不强制建立全局 `logs/` 目录。

------

### 7.9 禁止出现的正式目录

当前 baseline 正式目录中禁止自动生成：

```
fold_0_debug/
fold_0_gpu2/
fold_0_tmp/
baseline_v2/
upper_v2/
timestamp 目录
pseudo_student/tri_test/
pseudo_teacher/tri_test/
prompts/prompts_test.json
fold_0/prompts_train.json
fold_0/prompts_test.json
```

如果 debug 需要新目录，必须手动指定，且不得混入正式统计。

------

### 7.10 IdeaX 目录隔离原则

baseline 是 frozen reference。当前正式 baseline 根目录只保存 box-only baseline 结果，不直接保存 IdeaX 产物。

所有 IdeaX 实验统一使用 §17 中定义的独立根目录：

```text
/storage/baiyuting/data/out_data_idea1/
```

禁止覆盖：

```
prompts/prompts_train.json
pseudo_student/tri_train/
pseudo_teacher/tri_train/
Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/
```

除非明确是在重跑 frozen baseline。

---

## 8. Stage 1：预处理

### 8.1 目的
建立三空间合同：
- `native -> teacher`
- `teacher -> native`
- `native -> student`
- `student -> native`

并导出：

- native_npy/gts
- teacher_npy/imgs
- teacher_npy/gts
- student_npy/imgs
- student_npy/gts
- manifest.json
- split_meta.json
- geometry_meta.json
- preprocess_stats.json
- tiny_gt_filter_records.json
- stage_time_preprocess.json

### 8.2 正式命令
```bash
cd /storage/baiyuting/data/MedSAM-main

python utils/processed.py \
  --organized_root /storage/baiyuting/data/MedSAM-main/data/organized \
  --processed_root /storage/baiyuting/data/MedSAM-main/data/processed \
  --datasets btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,ph2 \
  --student_size_json /storage/baiyuting/data/MedSAM-main/utils/data/student_size_policy_literature.json \
  --progress_root /storage/baiyuting/data/MedSAM-main/data/progress \
  --progress_every 1
```

### 8.3 运行后必须检查
- `meta/manifest.json`
- `meta/split_meta.json`
- `meta/geometry_meta.json`

尤其确认：
- 3D 数据有 `slice_idx`
- 3D 数据有 `spacing`
- `teacher_target_size` 正确
- `student_target_h / student_target_w` 正确

还必须确认：

- `native_npy/gts` 是否存在；
- `preprocess_stats.json` 是否存在；
- `tiny_gt_filter_records.json` 是否存在；
- `tiny_gt_filter_records.json` 中 `split` 只能是 `train` 或空；
- `stage_time_preprocess.json` 是否存在；
- `test GT` 未被 tiny filter 修改。

尤其是：

```
tiny_gt_filter_records.json 中 split 只能是 train 或空
```

这是你现在最高规范的关键证据。

---

### 8.4 Test GT Immutable Rule

Stage 1 会对 train/test 图像都执行确定性图像预处理，包括格式转换、归一化、resize/pad 和三空间映射。

但是，test GT 只允许进行几何映射，用于最终 evaluation alignment。

严格禁止：
- 对 test GT 执行 tiny component filter
- 对 test GT 执行 morphology
- 对 test GT 执行语义清洗
- 用 test GT 生成 prompt
- 用 test GT 生成 pseudo

tiny GT filter 只允许作用于 train split，并且仅用于训练监督构造。

## 9. Stage 1.5：strict prompt 生成

### 9.1 正式 prompt 只允许生成 train split：

- 允许：prompts/prompts_train.json
- 禁止：prompts/prompts_test.json
- 禁止：fold_0/prompts_test.json
- 禁止：test GT → prompt

### 9.2 固定规则

bbox 必须满足：
- 只由 GT 自动生成
- 只在 native space 定义
- 只允许 tight bbox
- 目前不允许 expansion / margin / jitter / random perturbation

### 9.3 三种规则
- `instance`：单类多目标
- `union`：弥散结构整体取一个框
- `per_class_component`：多类任务每类按离散连通域分别出框

### 9.4 当前 11 个数据集的建议口径
- `btcv / synapse / acdc / prostate158`：`per_class_component`
- `kvasirseg / cvc_clinicdb / tn3k / tg3k / ddti / otu_2d / ph2`：默认单前景任务，按 `instance` 或单目标 tight box 主线处理

### 9.4.1 当前正式新增规则：连通域独立 bbox
当前正式版补充写死：

- 2D：一真实病灶一框
- 3D 多类别：同类多个离散区域也必须分别出框
- 严禁再用一个 union box 套住同类多个离散连通域

也就是说：
- 2D 多病灶样本不能把两个病灶合成一个大框
- `btcv / synapse / acdc / prostate158` 中，同一器官若在某个 slice 出现多个离散连通域，也必须分别给 box

### 9.5 正式命令
```bash
cd /storage/baiyuting/data/MedSAM-main

python generate_prompts.py \
  --processed_root /storage/baiyuting/data/MedSAM-main/data/processed \
  --datasets btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,ph2 \
  --fold all \
  --split train \
  --overwrite \
  --progress_root /storage/baiyuting/data/MedSAM-main/data/progress \
  --progress_every 25
```

### 9.6 prompt 对齐检查
正式要求：
- `missing_in_manifest = 0`

建议检查：
```bash
python - <<'PY'
import os, json

fold_root = "/storage/baiyuting/data/MedSAM-main/data/processed/kvasirseg/fold_0"

with open(os.path.join(fold_root, "meta", "manifest.json"), "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(os.path.join(fold_root, "prompts", "prompts_train.json"), "r", encoding="utf-8") as f:
    prompts = json.load(f)

manifest_train = {x["slice_name"] for x in manifest if x.get("split") == "train"}
prompt_keys = set(prompts.keys())

print("manifest_train =", len(manifest_train))
print("prompt_keys =", len(prompt_keys))
print("missing_in_manifest =", len(prompt_keys - manifest_train))
print("missing_in_prompt =", len(manifest_train - prompt_keys))
print("first_20_prompt_only =", list(sorted(prompt_keys - manifest_train))[:20])
print("first_20_manifest_only =", list(sorted(manifest_train - prompt_keys))[:20])
PY
```

### 9.7 当前最容易出错的点
最常见问题：
- 旧 `prompts_train.json` 没删
- 新旧 prompt 混读
- `find_prompt_json(...)` 优先读到了旧 prompt

所以正式重跑前建议先删：
- 根目录旧 prompt
- `prompts/` 子目录旧 prompt

---

## 10. Stage 2：train tri pseudo 生成

### 10.1 当前主线唯一合法标签定义
对每个像素：

- 框外：`0`
- 框内且被 MedSAM 判定为前景：`1..K`
- 框内但未被确认前景：`255`

### 10.2 补充规则
- 二分类任务前景 = `1`
- 多分类任务保留原始类别 `1..K`
- 多类冲突区统一写 `255`
- 3D 空切片直接全 `0`，不送入 MedSAM

### 10.3 当前正式主线只保留
- `pseudo_teacher/tri_train`
- `pseudo_student/tri_train`

### 10.4 当前明确不保留

- `vis_train`
- `hard pseudo`
- `weight_map`
- `test pseudo`

### 10.5 正式命令
```bash
cd /storage/baiyuting/data/MedSAM-main

python generate_pseudo_labels.py \
  --base_dir /storage/baiyuting/data/MedSAM-main/data \
  --checkpoint /storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth \
  --datasets btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,ph2 \
  --fold all \
  --split train \
  --overwrite \
  --vis_limit 0 \
  --progress_root /storage/baiyuting/data/MedSAM-main/data/progress \
  --progress_every 20
```

### 10.6 这里必须写死的规则
- 正式批量只跑 `--split train`
- 正式批量统一 `--vis_limit 0`
- 正式批量允许 `--overwrite`
- 不允许 `--split all`
- 不允许生成 `test pseudo`

### 10.7 小样本 sanity check
第一次建议先跑一个代表性数据集：
- `btcv`

比如：
```bash
python generate_pseudo_labels.py   --base_dir /storage/baiyuting/data/MedSAM-main/data   --checkpoint /storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth   --datasets btcv   --fold fold_0   --split train   --max_samples 120   --overwrite   --vis_limit 0
```

### 10.8 伪标签阶段必须检查
- `pseudo_student/tri_train` 是否存在
- `pseudo_box255_stats_train.json` 是否正常
- `stage_time_pseudo_train.json` 是否生成
- `255` 是否只在 bbox 内
- 空切片是否全 0

伪标签完整性要求：

- `len(pseudo_student/tri_train/*.npy) == len(prompts/prompts_train.json)`
- `len(pseudo_teacher/tri_train/*.npy) == len(prompts/prompts_train.json)`
- `pseudo_box255_stats_train.json` 中 `num_errors = 0`
- `pseudo_box255_stats_train.json` 中 `num_processed = num_prompt_items`
- `pseudo_box255_stats_train.json` 中 `num_skipped_existing = 0`
- `prompt_json_path` 必须指向 `prompts/prompts_train.json`
- 标签取值只能属于 `{0, 1..K, 255}`
- `255` 只允许出现在 box union 内
- 禁止存在 `pseudo_student/tri_test`
- 禁止存在 `pseudo_teacher/tri_test`

---

## 11. Stage 2.5：学生端数据检查

正式训练前必须做一次数据集读取检查：

```bash
python test_student_patch_dataset.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --split train
```

必须确认：
- baseline 能正确读 `pseudo_student/tri_train`
- upper 能正确读 `student_gt`
- image / mask shape 对齐
- `255` 只在 baseline 中出现

---

## 12. Stage 3：学生端训练

### 12.1 当前正式两组
#### baseline
- 训练标签：`pseudo_student/tri_train`
- `255` 作为 ignore

#### upper
- 训练标签：`student_gt`

### 12.2 当前固定超参数主线
- `epochs = 50`
- `lr = 1e-4`
- `weight_decay = 0.05`
- `freeze_encoder_epochs = 10`
- `amp = on`
- `deep_supervision = on`

### 12.3 当前 batch size / num_workers
- `btcv`: `batch_size=1`, `num_workers=0`
- `synapse`: `batch_size=1`, `num_workers=0`
- `acdc`: `batch_size=2`, `num_workers=2`
- `prostate158`: `batch_size=2`, `num_workers=2`
- `kvasirseg`: `batch_size=8`, `num_workers=4`
- `cvc_clinicdb`: `batch_size=8`, `num_workers=4`
- `tn3k`: `batch_size=8`, `num_workers=4`
- `tg3k`: `batch_size=8`, `num_workers=4`
- `ddti`: `batch_size=8`, `num_workers=4`
- `otu_2d`: `batch_size=8`, `num_workers=4`
- `ph2`: `batch_size=8`, `num_workers=4`

### 12.4 正式训练命令模板
#### baseline
```bash
CUDA_VISIBLE_DEVICES=2 python -u /storage/baiyuting/data/Swin-UMamba-main/pipeline/train_student.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --mode baseline   --epochs 50   --batch_size <bs>   --num_workers <nw>   --lr 1e-4   --weight_decay 0.05   --freeze_encoder_epochs 10   --amp   --deep_supervision   --pretrained_ckpt /storage/baiyuting/data/Swin-UMamba-main/data/pretrained/vmamba/vmamba_tiny_e292.pth   --out_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0
```

#### upper
```bash
CUDA_VISIBLE_DEVICES=2 python -u /storage/baiyuting/data/Swin-UMamba-main/pipeline/train_student.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --mode upper   --epochs 50   --batch_size <bs>   --num_workers <nw>   --lr 1e-4   --weight_decay 0.05   --freeze_encoder_epochs 10   --amp   --deep_supervision   --pretrained_ckpt /storage/baiyuting/data/Swin-UMamba-main/data/pretrained/vmamba/vmamba_tiny_e292.pth   --out_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/upper/<dataset>/fold_0
```

### 12.5 当前正式批处理脚本的行为
你的 `run_train_all_x2.sh` 已经写死：
- 只处理这 11 个数据集
- `btcv / synapse` 固定 `bs=1, nw=0`
- 其余 3D 用 `bs=2, nw=2`
- 其余 2D 用 `bs=8, nw=4`
- 输出目录固定为 `work_dir/<mode>/<dataset>/fold_0`
- 运行前先删原目录再写回同一目录
- x2 只改变并行调度方式，不改变单个任务的训练超参数、输出目录和评估口径

---

## 13. Stage 4：推理

### 13.1 输出目录
固定写入：
- `work_dir/baseline/<dataset>/fold_0/pred_test`
- `work_dir/upper/<dataset>/fold_0/pred_test`

### 13.2 正式命令模板
```bash
python /storage/baiyuting/data/Swin-UMamba-main/pipeline/infer_student.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --mode baseline   --deep_supervision   --ckpt /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/last.pth   --out_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/pred_test   --amp
```

upper 只要把 `mode` 和 `ckpt/out_dir` 换成 upper。

### 13.3 当前批处理脚本行为
`run_infer_eval_all.sh` 的规则也是：
- 先删 `pred_test`
- 再在原固定路径重建
- 不新建其他版本目录

---

## 14. Stage 5：正式评估

### 14.1 2D 数据
当前正式指标：
- `Dice`
- `IoU`
- `MAE`（只对 `kvasirseg`、`cvc_clinicdb` 加 `--report_mae`）

正式命令模板：
```bash
python /storage/baiyuting/data/Swin-UMamba-main/pipeline/eval_2d.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --pred_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/pred_test   --save_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/eval_2d   --split test   --require_native_gt
```

### 14.2 3D 数据
当前正式指标：
- `DSC`
- `HD95(mm)`
- `ASSD(mm)`

正式原则：
- 必须先恢复到 native slice 尺寸
- 必须按 `slice_idx` 排序重建 volume
- 必须使用真实 `spacing`
- 多类任务必须逐器官统计

正式命令模板：
```bash
python /storage/baiyuting/data/Swin-UMamba-main/pipeline/eval_3d.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --pred_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/pred_test   --save_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/eval_3d   --split test   --require_native_gt
```

### 14.3 当前 3D class name map 必须补齐
正式映射至少要保证：
- `btcv`
- `synapse`
- `acdc`
- `prostate158`

都能输出器官名，而不只是 `class_1/class_2/...`。

---

## 15. Stage 6：计时与汇总

### 15.1 正式时间来源

当前 baseline 的正式时间统计只读取各阶段 `stage_time_*.json`。

禁止使用以下内容作为正式时间来源：

- `data/progress/summary.csv`
- `data/progress/events.csv`
- `run_*.log`
- `train_log.csv`
- file mtime estimate

这些文件只能用于运行监控、训练曲线分析或 debug，不进入正式时间表。

### 15.2 每阶段应有的 stage_time
#### MedSAM 侧
- `stage_time_preprocess.json`
- `stage_time_prompts.json`
- `stage_time_pseudo_train.json`

#### Swin-UMamba 侧
- `stage_time_train_baseline.json`
- `stage_time_infer_baseline.json`
- `stage_time_eval_baseline.json`
- `stage_time_train_upper.json`
- `stage_time_infer_upper.json`
- `stage_time_eval_upper.json`

### 15.3 汇总前必须确认
如果 timer 还显示很长的旧时间：
- 先确认对应阶段是否真的生成了新的 `stage_time_*.json`
- 如果没有，就必须先重跑该阶段
- 再重跑 timer

---

## 16. 当前最稳执行顺序

### 16.1 单数据集先跑通
建议先拿 `btcv` 跑通：

```text
processed
→ generate_prompts
→ 对齐检查（missing_in_manifest = 0）
→ generate_pseudo_labels (train, vis_limit=0)
→ 检查 pseudo stats + stage_time
→ test_student_patch_dataset
→ train_student baseline
→ train_student upper
→ infer baseline
→ eval baseline
→ infer upper
→ eval upper
→ timer 汇总
```

### 16.2 全量批跑顺序
确认 `btcv` 跑通后，再批量扩到 11 个数据集：

1. 全部数据集 `processed`
2. 全部数据集 `generate_prompts`
3. 全部数据集 `generate_pseudo_labels --split train --vis_limit 0`
4. 逐个 spot check `pseudo_student/tri_train`
5. 跑 `run_train_all_x2.sh`
6. 跑 `run_infer_eval_all.sh`
7. 跑总表汇总脚本
8. 跑 timer 汇总

---

## 17. IdeaX 扩展合同

针对一些新加的模块功能，IdeaX 采用如下命名方式，并使用独立目录保存结果，避免覆盖当前 frozen baseline。

### 17.1 全局根目录

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

### 17.2 方法名命名规则

以后每一个实验先定义一个唯一的方法名：

```
METHOD=idea1_<variant>
```

### 17.3 每个数据集的基础目录

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

### 17.4 Prompt 文件设计

Prompt 文件放在：

```
$PROCESSED_ROOT/<dataset>/fold_0/prompts/
```

命名规则：

```
prompts_train_${METHOD}.json
```

### 17.5 Prompt 审计和统计文件

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

### 17.6 伪标签目录设计

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

在当前 frozen baseline 根目录中：

```
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/pseudo_student/tri_train/
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/pseudo_teacher/tri_train/
```

是 baseline 的正式伪标签目录，不是临时目录，IdeaX 不允许覆盖。

在 IdeaX 独立根目录 `$IDEA_ROOT` 中：

```
$PROCESSED_ROOT/<dataset>/fold_0/pseudo_student/tri_train/
$PROCESSED_ROOT/<dataset>/fold_0/pseudo_teacher/tri_train/
```

可以作为当前方法运行时的中间输出目录。

为了保存不同方法结果，正式保留时必须另存为：

```
pseudo_student/tri_train_${METHOD}/
pseudo_teacher/tri_train_${METHOD}/
```

这样可以避免不同 IdeaX 变体互相覆盖，也可以避免污染 frozen baseline。

### 17.7 Box-only baseline 目录

在 `$IDEA_ROOT` 中，无论新 Idea1 怎么改，都建议额外保留一份 box-only 对照伪标签：

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

`tri_train_boxonly/` 本质上就是 box-only baseline pseudo。若从 frozen baseline 的 `pseudo_student/tri_train/` 复制得到，则二者内容一致。单独保存它只是为了让 IdeaX 实验目录自包含，并避免后续生成新方法时覆盖正式 baseline。

### 17.8 分析结果目录设计

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

### 17.9 指标输出目录

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

### 17.10 日志目录设计

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

---

## 18. 实验数据记录与历史结果

本章只记录实验结果和历史表格，不定义当前 baseline 的流程、目录或执行规则。当前正式流程以 §0–§17 为准。

本节总表来源于以下自动汇总文件：

- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_metrics_2d.csv`
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_metrics_3d.csv`
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_metrics_3d_per_organ.csv`
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_time_12datasets_hdotm.csv`

### 18. 2 2D 总指标表（7 datasets）

| dataset | baseline_dice | upper_dice | delta_dice_upper_minus_baseline | baseline_iou | upper_iou | delta_iou_upper_minus_baseline | baseline_mae_fg | upper_mae_fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kvasirseg | 0.8426 | 0.9242 | 0.0816 | 0.7416 | 0.8703 | 0.1287 | 0.0547 | 0.0215 |
| cvc_clinicdb | 0.8394 | 0.9008 | 0.0614 | 0.7498 | 0.8548 | 0.1049 | 0.0270 | 0.0078 |
| tn3k | 0.8004 | 0.8318 | 0.0314 | 0.6924 | 0.7371 | 0.0447 |  |  |
| tg3k | 0.5790 | 0.7706 | 0.1916 | 0.4334 | 0.6855 | 0.2521 |  |  |
| ddti | 0.7641 | 0.7990 | 0.0349 | 0.6521 | 0.6960 | 0.0438 |  |  |
| otu_2d | 0.8153 | 0.8493 | 0.0340 | 0.7206 | 0.7733 | 0.0527 |  |  |
| ph2 | 0.8985 | 0.9487 | 0.0503 | 0.8176 | 0.9052 | 0.0876 |  |  |

### 18. 3 3D 总指标表（4 datasets）

| dataset | baseline_macro_dice | upper_macro_dice | delta_dice_upper_minus_baseline | baseline_macro_hd95_mm | upper_macro_hd95_mm | baseline_macro_assd_mm | upper_macro_assd_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| btcv | 0.5213 | 0.7906 | 0.2694 | 28.669 | 13.001 | 6.641 | 2.944 |
| synapse | 0.7263 | 0.8564 | 0.1301 | 17.020 | 9.531 | 3.376 | 1.856 |
| acdc | 0.5834 | 0.8813 | 0.2980 | 14.109 | 3.616 | 3.979 | 0.842 |
| prostate158 | 0.7570 | 0.8204 | 0.0634 | 4.615 | 4.583 | 1.433 | 1.074 |

### 18.4  3D 分器官指标表（class-wise）

| dataset | class_id | label_name | baseline_dice | upper_dice | delta_dice_upper_minus_baseline | baseline_hd95_mm | upper_hd95_mm | delta_hd95_upper_minus_baseline | baseline_assd_mm | upper_assd_mm | delta_assd_upper_minus_baseline | n_cases_baseline | n_cases_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| btcv | 1 | spleen | 0.7984 | 0.9385 | 0.1401 | 19.594 | 10.694 | -8.900 | 4.695 | 2.167 | -2.527 | 6 | 6 |
| btcv | 2 | right_kidney | 0.6520 | 0.7501 | 0.0981 | 23.776 | 17.315 | -6.461 | 8.660 | 5.999 | -2.661 | 6 | 6 |
| btcv | 3 | left_kidney | 0.7170 | 0.8108 | 0.0937 | 21.995 | 16.759 | -5.236 | 4.740 | 3.123 | -1.617 | 6 | 6 |
| btcv | 4 | gallbladder | 0.1126 | 0.6338 | 0.5212 | 28.656 | 12.632 | -16.024 | 6.834 | 3.754 | -3.080 | 6 | 6 |
| btcv | 5 | esophagus | 0.6099 | 0.7947 | 0.1848 | 20.995 | 7.465 | -13.530 | 3.610 | 1.444 | -2.166 | 6 | 6 |
| btcv | 6 | liver | 0.7879 | 0.9652 | 0.1773 | 38.790 | 30.631 | -8.158 | 9.197 | 3.187 | -6.010 | 6 | 6 |
| btcv | 7 | stomach | 0.7081 | 0.8680 | 0.1599 | 32.744 | 13.253 | -19.491 | 7.072 | 2.758 | -4.314 | 6 | 6 |
| btcv | 8 | aorta | 0.8928 | 0.9239 | 0.0311 | 37.560 | 11.018 | -26.542 | 5.516 | 1.366 | -4.150 | 6 | 6 |
| btcv | 9 | ivc | 0.7259 | 0.8707 | 0.1448 | 11.483 | 7.270 | -4.213 | 2.872 | 1.585 | -1.288 | 6 | 6 |
| btcv | 10 | portal_vein_splenic_vein | 0.2351 | 0.7328 | 0.4977 | 56.191 | 10.825 | -45.366 | 13.875 | 3.073 | -10.802 | 6 | 6 |
| btcv | 11 | pancreas | 0.5371 | 0.7011 | 0.1640 | 23.572 | 12.595 | -10.977 | 5.980 | 3.131 | -2.848 | 6 | 6 |
| btcv | 12 | right_adrenal_gland | 0.0000 | 0.7242 | 0.7242 |  | 5.799 |  |  | 1.067 |  | 6 | 6 |
| btcv | 13 | left_adrenal_gland | 0.0000 | 0.5646 | 0.5646 |  | 12.762 |  |  | 5.617 |  | 6 | 6 |
| synapse | 1 | aorta | 0.8652 | 0.9165 | 0.0513 | 8.199 | 7.111 | -1.088 | 1.389 | 1.119 | -0.270 | 12 | 12 |
| synapse | 2 | gallbladder | 0.4116 | 0.6587 | 0.2471 | 10.383 | 14.832 | 4.449 | 3.075 | 2.801 | -0.273 | 12 | 12 |
| synapse | 3 | left_kidney | 0.7990 | 0.8966 | 0.0977 | 10.698 | 4.394 | -6.304 | 2.348 | 0.988 | -1.359 | 12 | 12 |
| synapse | 4 | right_kidney | 0.7767 | 0.8671 | 0.0904 | 24.655 | 7.038 | -17.618 | 4.281 | 2.163 | -2.118 | 12 | 12 |
| synapse | 5 | liver | 0.8020 | 0.9623 | 0.1603 | 32.558 | 10.641 | -21.917 | 6.038 | 1.553 | -4.485 | 12 | 12 |
| synapse | 6 | pancreas | 0.5862 | 0.7381 | 0.1519 | 16.061 | 12.310 | -3.751 | 3.227 | 2.210 | -1.018 | 12 | 12 |
| synapse | 7 | spleen | 0.7806 | 0.9360 | 0.1554 | 17.593 | 7.328 | -10.265 | 3.601 | 1.711 | -1.890 | 12 | 12 |
| synapse | 8 | stomach | 0.7892 | 0.8760 | 0.0869 | 16.010 | 12.593 | -3.417 | 3.047 | 2.305 | -0.742 | 12 | 12 |
| acdc | 1 | rv | 0.7590 | 0.8859 | 0.1270 | 8.552 | 4.373 | -4.180 | 2.138 | 0.989 | -1.149 | 100 | 100 |
| acdc | 2 | myo | 0.5484 | 0.8422 | 0.2938 | 11.666 | 3.087 | -8.579 | 3.156 | 0.716 | -2.440 | 100 | 100 |
| acdc | 3 | lv | 0.4427 | 0.9158 | 0.4731 | 22.109 | 3.388 | -18.721 | 6.643 | 0.820 | -5.823 | 100 | 100 |
| prostate158 | 1 | central_gland | 0.8302 | 0.8788 | 0.0486 | 4.720 | 4.786 | 0.066 | 1.503 | 1.158 | -0.345 | 19 | 19 |
| prostate158 | 2 | peripheral_zone | 0.6838 | 0.7620 | 0.0782 | 4.511 | 4.380 | -0.130 | 1.363 | 0.989 | -0.374 | 19 | 19 |

### 18.5  时间统计表（单位：小时.分钟）

Full pipeline active wall-clock: 11.55 h.mm = 11.9227 hours

---

### 18.6 冻结baselineV1

**frozen_baseline_v1 已经冻结成功**。

你的目录里已经包含了应有的 9 个正式冻结文件：

```
metrics/
├── SHA256SUMS.txt
├── summary_metrics_2d.csv
├── summary_metrics_3d.csv
├── summary_metrics_3d_per_organ.csv
└── summary_metrics_3d_per_organ.md

time/
├── SHA256SUMS.txt
├── summary_time_11datasets_4gpu_wallclock.csv
├── summary_time_11datasets_4gpu_wallclock.json
└── summary_time_11datasets_detail_records.csv
```

这说明当前 **MedSAM→Swin-UMamba baseline v1** 已经有：

```
1. 2D 指标总表
2. 3D 指标总表
3. 3D 分器官指标表
4. 4GPU 全链路时间统计
5. 明细计时记录
6. hash 校验文件
```

可以正式作为 baseline v1 结果保存。
