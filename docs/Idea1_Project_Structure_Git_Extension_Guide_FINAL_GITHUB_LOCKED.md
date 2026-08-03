# Idea1 Hard-Full MedSAM FT：项目结构、Git 与后续扩充指南

> 最后更新：2026-08-03 17:02（UTC+8）  
> 当前正式方法：`idea1_hard_full_medsam_ft`  
> 当前正式 Git 分支：`idea1-hard-full-medsam-ft`  
> GitHub 归档状态：MedSAM 与 Swin-UMamba 两个仓库均已推送远端分支

## 1. 当前只保留两个正式系统

### 1.1 Frozen Baseline V1

冻结管线内部包含两个模式：

- **Frozen Baseline**：Student 使用 `pseudo_student/tri_train`；标签 `255` 作为 ignore。
- **Frozen Upper**：Student 使用 `student_npy/gts/train` 完全监督。

正式结果：

```text
/storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline
/storage/baiyuting/data/Swin-UMamba-main/work_dir/upper
```

### 1.2 Idea1 Hard-Full MedSAM FT

Idea1 是一套完整实验，内部包含两个实验臂：

1. Box-only rerun：
   当前 Idea1 代码环境下的对照臂，不进行 Hard Full MedSAM 微调。

2. Idea1 method：
   选择少量 Hard Full 样本微调 MedSAM mask decoder，
   再为剩余 Box 样本生成更新后的伪标签并训练 Student。

两个实验臂的物理结果位置分别为：

- Box-only rerun：
  `/storage/baiyuting/data/out_data_idea1/Swin-UMamba-main/work_dir/baseline_boxonly`

- Idea1 method：
  `/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft/Swin-UMamba-main/work_dir/idea1_hard_full_medsam_ft`

统一访问入口：

`/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft/result_links/current_run/`

其中：

- `boxonly` 指向 Box-only rerun；
- `idea1` 指向 Idea1 method。

---


## 2. 当前目录结构：物理结构冻结，访问结构精简

### 2.1 当前实际物理结构

当前 Idea1 的真实物理路径保持不动，以避免破坏已完成实验中的绝对路径、日志、JSON 和脚本引用。日常只需要记住以下五类入口：

```text
/storage/baiyuting/data/
├── MedSAM-main/                                  # Frozen Teacher runtime/data
├── Swin-UMamba-main/                             # Frozen Student runtime/results
├── git_repos/                                    # 两个正式 Git clone
│   ├── MedSAM-research/
│   └── Swin-UMamba-research/
└── out_data_idea1/
    ├── code/                                     # Baseline 核心代码 + Idea1 代码
    ├── baseline_reference/                       # Frozen Baseline/Upper 的只读引用
    ├── MedSAM-main/data/processed/               # Box-only 对照数据
    ├── Swin-UMamba-main/work_dir/baseline_boxonly/
    └── formal_runs/idea1_hard_full_medsam_ft/    # Idea1 正式运行、结果与审计
```

其中最重要的真实路径是：

```text
Baseline 核心代码：
/storage/baiyuting/data/out_data_idea1/code/baseline_v1_core

Idea1 完整代码：
/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft

Frozen Baseline：
/storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline

Frozen Upper：
/storage/baiyuting/data/Swin-UMamba-main/work_dir/upper

Box-only rerun：
/storage/baiyuting/data/out_data_idea1/Swin-UMamba-main/work_dir/baseline_boxonly

Idea1 正式结果：
/storage/baiyuting/data/out_data_idea1/formal_runs/idea1_hard_full_medsam_ft/
Swin-UMamba-main/work_dir/idea1_hard_full_medsam_ft
```

### 2.2 推荐的精简访问视图

不移动现有文件，只增加软链接入口：

```text
/storage/baiyuting/data/out_data_idea1/project_view/
├── code/
│   ├── baseline
│   └── method
├── refs/
│   ├── frozen_data
│   ├── frozen_baseline
│   ├── frozen_upper
│   └── boxonly
├── run/
│   ├── data
│   ├── views
│   ├── student
│   └── logs
├── reports/
│   ├── metrics
│   └── figures
└── manifest
```

映射关系：

| 精简入口 | 真实目标 |
|---|---|
| `code/baseline` | `code/baseline_v1_core` |
| `code/method` | `code/idea1_hard_full_medsam_ft` |
| `refs/frozen_data` | Frozen `MedSAM-main/data/processed` |
| `refs/frozen_baseline` | Frozen Baseline 结果 |
| `refs/frozen_upper` | Frozen Upper 结果 |
| `refs/boxonly` | Idea1-side Box-only 结果 |
| `run/data` | Idea1 正式 processed data |
| `run/views` | Idea1 `student_views` |
| `run/student` | Idea1 Student 结果 |
| `run/logs` | Idea1 日志 |
| `reports/metrics` | 定档指标与审计 |
| `reports/figures` | Teacher-space 可视化 |
| `manifest` | `00_manifest` |

该视图只用于人工访问，不改变任何训练、评估和复现实验路径。

### 2.3 可选：创建精简访问视图

```bash
export IDEA_ROOT=/storage/baiyuting/data/out_data_idea1
export FORMAL_ROOT=$IDEA_ROOT/formal_runs/idea1_hard_full_medsam_ft
export VIEW_ROOT=$IDEA_ROOT/project_view

mkdir -p \
  "$VIEW_ROOT/code" \
  "$VIEW_ROOT/refs" \
  "$VIEW_ROOT/run" \
  "$VIEW_ROOT/reports"

ln -sfn "$IDEA_ROOT/code/baseline_v1_core" \
  "$VIEW_ROOT/code/baseline"

ln -sfn "$IDEA_ROOT/code/idea1_hard_full_medsam_ft" \
  "$VIEW_ROOT/code/method"

ln -sfn /storage/baiyuting/data/MedSAM-main/data/processed \
  "$VIEW_ROOT/refs/frozen_data"

ln -sfn /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline \
  "$VIEW_ROOT/refs/frozen_baseline"

ln -sfn /storage/baiyuting/data/Swin-UMamba-main/work_dir/upper \
  "$VIEW_ROOT/refs/frozen_upper"

ln -sfn "$IDEA_ROOT/Swin-UMamba-main/work_dir/baseline_boxonly" \
  "$VIEW_ROOT/refs/boxonly"

ln -sfn "$FORMAL_ROOT/MedSAM-main/data/processed" \
  "$VIEW_ROOT/run/data"

ln -sfn "$FORMAL_ROOT/student_views" \
  "$VIEW_ROOT/run/views"

ln -sfn \
  "$FORMAL_ROOT/Swin-UMamba-main/work_dir/idea1_hard_full_medsam_ft" \
  "$VIEW_ROOT/run/student"

ln -sfn "$FORMAL_ROOT/logs" \
  "$VIEW_ROOT/run/logs"

ln -sfn "$FORMAL_ROOT/summaries/final" \
  "$VIEW_ROOT/reports/metrics"

ln -sfn "$FORMAL_ROOT/visualizations/final_teacher_space" \
  "$VIEW_ROOT/reports/figures"

ln -sfn "$FORMAL_ROOT/00_manifest" \
  "$VIEW_ROOT/manifest"
```

验证：

```bash
find "$VIEW_ROOT" -maxdepth 2 -type l \
  -printf '%p -> %l
' | sort
```

---

## 3. 三类目录的职责

### `baseline_v1_core`

- 保存可运行 Baseline 核心代码、环境、血缘和 SHA256；
- 是 Idea2、Idea3 等独立方法的唯一父代码；
- 原则上只读；
- 不放数据、checkpoint、预测、`work_dir`、缓存。

### `idea1_hard_full_medsam_ft`

- 当前 Idea1 的开发代码；
- 包含 Teacher 和 Student 两侧；
- 后续仅做必要修复、汇总和归档；
- 不混入独立 Idea2。

### `formal_runs/idea1_hard_full_medsam_ft`

- 保存正式实验的数据副本、训练结果、日志、指标和可视化；
- 是论文复核和实验审计入口；
- 不初始化为 Git 仓库。

---


## 4. Git 仓库与当前远端归档状态

正式 Git clone：

```text
/storage/baiyuting/data/git_repos/MedSAM-research
/storage/baiyuting/data/git_repos/Swin-UMamba-research
```

正式分支：

```text
idea1-hard-full-medsam-ft
```

### 4.1 已完成的提交与推送

| Repository | Local/remote branch | Locked commit | Push status |
|---|---|---|---|
| MedSAM-research | `idea1-hard-full-medsam-ft` | `0dbc2da6adf1b4f875af7d09a31cdee12b59ad98` | 已推送 |
| Swin-UMamba-research | `idea1-hard-full-medsam-ft` | `48462cd184f56205873682f97c8e77079710d6c4` | 已推送 |

Swin-UMamba 已验证：

```text
HEAD:
48462cd184f56205873682f97c8e77079710d6c4

origin/idea1-hard-full-medsam-ft:
48462cd184f56205873682f97c8e77079710d6c4
```

因此 Swin-UMamba 本地分支与远端分支完全一致。

MedSAM 推送输出已确认远端分支创建成功；提交时的锁定 commit 为：

```text
0dbc2da6adf1b4f875af7d09a31cdee12b59ad98
```

两个仓库提交后的 `status --short` 均为空。

### 4.2 当前 Git 状态检查

```bash
export MEDSAM_GIT=/storage/baiyuting/data/git_repos/MedSAM-research
export SWIN_GIT=/storage/baiyuting/data/git_repos/Swin-UMamba-research

git -C "$MEDSAM_GIT" status -sb
git -C "$SWIN_GIT" status -sb

git -C "$MEDSAM_GIT" rev-parse HEAD
git -C "$MEDSAM_GIT" rev-parse origin/idea1-hard-full-medsam-ft

git -C "$SWIN_GIT" rev-parse HEAD
git -C "$SWIN_GIT" rev-parse origin/idea1-hard-full-medsam-ft
```

预期：

```text
两个仓库均位于 idea1-hard-full-medsam-ft
本地 HEAD 与 origin/idea1-hard-full-medsam-ft 相同
工作区无未提交修改
```

### 4.3 重要说明：此前 rsync dry-run 未真正执行

终端中执行的：

```bash
rsync -avhn --delete ...
```

会把 `...` 当成真实文件名，因此报错：

```text
link_stat "...": No such file or directory
```

这不影响已经完成的 Git commit/push，但说明当时没有通过该命令完成“live code 与 Git clone 的目录级一致性审计”。

最终归档前建议补做一次只读 dry-run，不能包含省略号：

```bash
export IDEA_CODE=/storage/baiyuting/data/out_data_idea1/code/idea1_hard_full_medsam_ft
export MEDSAM_SRC=$IDEA_CODE/MedSAM-main
export SWIN_SRC=$IDEA_CODE/Swin-UMamba-main

rsync -avhn --delete \
  --exclude='.git/' --exclude='data/' --exclude='work_dir/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='.idea/' --exclude='.vscode/' --exclude='.claude/' \
  --exclude='_archive/' --exclude='*.egg-info/' \
  --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='*.npy' --exclude='*.npz' \
  --exclude='*.nii' --exclude='*.nii.gz' \
  --exclude='*.log' --exclude='*.bak*' --exclude='*.rej' \
  --exclude='*.code-workspace' \
  "$MEDSAM_SRC/" "$MEDSAM_GIT/"

rsync -avhn --delete \
  --exclude='.git/' --exclude='data/' --exclude='work_dir/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='.idea/' --exclude='.vscode/' --exclude='.claude/' \
  --exclude='_archive/' --exclude='*.egg-info/' \
  --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
  --exclude='*.npy' --exclude='*.npz' \
  --exclude='*.nii' --exclude='*.nii.gz' \
  --exclude='*.log' --exclude='*.bak*' --exclude='*.rej' \
  "$SWIN_SRC/" "$SWIN_GIT/"
```

若 dry-run 输出仍有源码差异，应先人工审查，再决定是否产生新的补充 commit；不要直接去掉 `-n`。

### 4.4 后续文档更新提交

本指南是在上述两个代码 commit 推送完成后更新的。若需要让本指南也出现在 GitHub 上，应作为独立的 documentation-only commit 提交，避免修改已锁定的代码提交语义。

---


## 5. 后续 Idea2 / Idea3：采用精简物理结构

独立新方法必须从 `baseline_v1_core` 开始，不从 Idea1 叠加。未来不再复制 Idea1 当前的深层目录，而采用五个一级目录：

```text
/storage/baiyuting/data/out_data_idea2/
├── code/
│   ├── MedSAM-main/
│   └── Swin-UMamba-main/
├── refs/
│   ├── baseline_code
│   ├── frozen_data
│   ├── frozen_baseline
│   ├── frozen_upper
│   └── boxonly
├── run/
│   ├── data/
│   ├── views/
│   ├── student/
│   └── logs/
├── reports/
│   ├── metrics/
│   ├── audits/
│   └── figures/
└── manifest/
    ├── paths.env
    ├── experiment_contract.md
    ├── code_lineage.md
    └── result_lock.md
```

职责：

| Directory | Responsibility |
|---|---|
| `code/` | 当前方法源码 |
| `refs/` | Baseline、Upper、Frozen data 等只读引用 |
| `run/` | 当前方法的数据、Student view、checkpoint、预测和日志 |
| `reports/` | 指标、协议审计、表格和可视化 |
| `manifest/` | 路径合同、代码血缘和最终定档 |

### 5.1 初始化 Idea2

示例名称：

```text
idea2_selective_low_response_calibration
```

初始化代码：

```bash
export BASELINE_CORE=/storage/baiyuting/data/out_data_idea1/code/baseline_v1_core
export IDEA2_ROOT=/storage/baiyuting/data/out_data_idea2

mkdir -p \
  "$IDEA2_ROOT/code" \
  "$IDEA2_ROOT/refs" \
  "$IDEA2_ROOT/run/data" \
  "$IDEA2_ROOT/run/views" \
  "$IDEA2_ROOT/run/student" \
  "$IDEA2_ROOT/run/logs" \
  "$IDEA2_ROOT/reports/metrics" \
  "$IDEA2_ROOT/reports/audits" \
  "$IDEA2_ROOT/reports/figures" \
  "$IDEA2_ROOT/manifest"

rsync -a \
  --chmod=Du+rwx,Dgo+rx,Fu+rw,Fgo+r \
  "$BASELINE_CORE/MedSAM-main/" \
  "$IDEA2_ROOT/code/MedSAM-main/"

rsync -a \
  --chmod=Du+rwx,Dgo+rx,Fu+rw,Fgo+r \
  "$BASELINE_CORE/Swin-UMamba-main/" \
  "$IDEA2_ROOT/code/Swin-UMamba-main/"
```

创建只读引用：

```bash
ln -sfn "$BASELINE_CORE" \
  "$IDEA2_ROOT/refs/baseline_code"

ln -sfn /storage/baiyuting/data/MedSAM-main/data/processed \
  "$IDEA2_ROOT/refs/frozen_data"

ln -sfn /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline \
  "$IDEA2_ROOT/refs/frozen_baseline"

ln -sfn /storage/baiyuting/data/Swin-UMamba-main/work_dir/upper \
  "$IDEA2_ROOT/refs/frozen_upper"
```

新 Git 分支建议：

```text
idea2-selective-low-response-calibration
```

---

## 6. 全部测试数据集的完整对比结果

当前正式评估协议统一为：

- **2D 数据集**：报告 DSC 与 IoU；
- **3D 数据集**：报告 DSC、HD95(mm) 与 ASSD(mm)；
- 主结果比较对象固定为 Frozen Baseline、Idea1、Frozen Upper；
- DSC、IoU 越高越好；HD95、ASSD 越低越好。


### 6.1 2D Student test：审计锁定后的 DSC 与 IoU

以下数值直接来自三套系统当前正式的：

```text
<work_root>/<dataset>/fold_0/eval_2d/eval_summary.json
```

统一读取字段：

```text
dice_macro.mean
iou_macro.mean
```

并已逐数据集确认：

- `split = test`；
- `evaluation_space = native`；
- `geometry_source = geometry_meta.json`；
- `debug_resize_enabled = false`；
- Frozen Baseline、Idea1、Frozen Upper 的预测文件数相同；
- 三组预测文件名集合完全一致。

| Dataset | Frozen Baseline DSC | Idea1 DSC | Frozen Upper DSC | ΔDSC | Frozen Baseline IoU | Idea1 IoU | Frozen Upper IoU | ΔIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TG3K | 0.5832 | **0.6226** | 0.7822 | **+0.0394** | 0.4408 | **0.4817** | 0.6986 | **+0.0409** |
| TN3K | **0.8125** | 0.8020 | 0.8423 | -0.0105 | **0.7156** | 0.6943 | 0.7573 | -0.0214 |
| Kvasir-SEG | **0.8565** | 0.8397 | 0.9251 | -0.0168 | **0.7613** | 0.7382 | 0.8736 | -0.0230 |
| CVC-ClinicDB | **0.8416** | 0.8315 | 0.9111 | -0.0101 | **0.7508** | 0.7357 | 0.8609 | -0.0151 |
| DDTI | 0.7680 | **0.7698** | 0.7937 | **+0.0018** | **0.6573** | 0.6555 | 0.6920 | -0.0018 |
| OTU-2D | 0.8126 | **0.8156** | 0.8515 | **+0.0030** | 0.7170 | **0.7190** | 0.7757 | **+0.0020** |
| PH2 | 0.8948 | **0.9116** | 0.9482 | **+0.0168** | 0.8120 | **0.8404** | 0.9048 | **+0.0284** |
| **Macro** | **0.7956** | **0.7990** | **0.8649** | **+0.0034** | **0.6935** | **0.6950** | **0.7947** | **+0.0014** |

说明：

- `ΔDSC = Idea1 - Frozen Baseline`；
- `ΔIoU = Idea1 - Frozen Baseline`；
- 2D macro DSC 从 0.7956 提升至 0.7990，绝对提升 **+0.0034**；
- 2D macro IoU 从 0.6935 提升至 0.6950，绝对提升 **+0.0014**；
- 2D 总体仅小幅提升，且不同数据集存在明显波动；
- 此表替代此前未完成协议审计的旧 2D 汇总表。

#### 6.1.1 2D MAE（仅当前正式 JSON 中提供该指标的数据集）

MAE 越低越好。

| Dataset | Frozen Baseline MAE | Idea1 MAE | Frozen Upper MAE | Idea1 − Baseline |
|---|---:|---:|---:|---:|
| Kvasir-SEG | **0.0508** | 0.0567 | 0.0214 | +0.0059 |
| CVC-ClinicDB | **0.0266** | 0.0312 | 0.0076 | +0.0046 |

当前其余 2D 数据集的 `mae_fg` 为 `None`，因此不构造跨 7 个数据集的 MAE macro。

### 6.2 3D Student test：DSC、HD95 与 ASSD

| Dataset | Frozen Baseline DSC | Idea1 DSC | Frozen Upper DSC | ΔDSC ↑ | Frozen Baseline HD95 | Idea1 HD95 | Frozen Upper HD95 | ΔHD95 ↓ | Frozen Baseline ASSD | Idea1 ASSD | Frozen Upper ASSD | ΔASSD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCV | 0.5221 | **0.5936** | 0.7890 | **+0.0715** | 28.5002 | **20.0188** | 11.7661 | **+8.4814** | 6.6380 | **5.3980** | 2.6126 | **+1.2400** |
| Synapse | 0.7374 | **0.7377** | 0.8516 | **+0.0003** | **18.3378** | 20.0476 | 13.2479 | -1.7098 | **4.4939** | 4.6027 | 2.7733 | -0.1088 |
| ACDC | 0.5699 | **0.7004** | 0.8811 | **+0.1305** | 15.0012 | **12.6409** | 3.5096 | **+2.3603** | 4.2224 | **3.0597** | 0.8291 | **+1.1627** |
| Prostate158 | 0.7488 | **0.7746** | 0.8281 | **+0.0258** | 5.0186 | **4.7839** | 3.9130 | **+0.2347** | 1.5132 | **1.3695** | 0.9974 | **+0.1437** |
| **Macro** | **0.6445** | **0.7016** | **0.8375** | **+0.0571** | **16.7145** | **14.3728** | **8.1092** | **+2.3417** | **4.2169** | **3.6075** | **1.8031** | **+0.6094** |

说明：

- `ΔDSC = Idea1 - Frozen Baseline`，正值表示提升；
- `ΔHD95 = Frozen Baseline - Idea1`，正值表示边界误差下降；
- `ΔASSD = Frozen Baseline - Idea1`，正值表示边界误差下降；
- 3D macro DSC 提升 **+0.0571**；
- 3D macro HD95 改善 **2.3417 mm**；
- 3D macro ASSD 改善 **0.6094 mm**；
- Synapse 的 DSC 基本持平，但 HD95 和 ASSD 略有恶化。

### 6.3 BTCV 正式测试结果

BTCV Idea1 正式测试结果采用 `overall_macro_mean`：

| 指标 | Frozen Baseline | Idea1 | Frozen Upper |
|---|---:|---:|---:|
| DSC ↑ | 0.5221 | **0.5936** | 0.7890 |
| HD95(mm) ↓ | 28.5002 | **20.0188** | 11.7661 |
| ASSD(mm) ↓ | 6.6380 | **5.3980** | 2.6126 |
| Test cases | 6 | 6 | 6 |

Idea1 相对 Frozen Baseline：

- DSC：**+0.0715**；
- HD95：降低 **8.4814 mm**；
- ASSD：降低 **1.2400 mm**。

### 6.4 总体结论

综合 11 个测试数据集：

- 2D：整体仅小幅改善，并存在多个数据集下降；
- 3D：整体提升更明显，尤其是 BTCV 和 ACDC；
- 边界指标并非在所有数据集上都改善，Synapse 是主要反例；
- Idea1 尚未达到 Frozen Upper 的完全监督上限。

### 6.5 结果协议审计与定档状态

#### 2D

7 个 2D 数据集均已完成只读审计，并满足：

```text
split = test
evaluation_space = native
geometry_source = geometry_meta.json
debug_resize_enabled = false
```

对每个数据集，Frozen Baseline、Idea1、Frozen Upper 均满足：

```text
prediction_file_count 相同
预测文件名集合完全一致
dice_macro.n = iou_macro.n = num_eval_slices
正式 summary = eval_2d/eval_summary.json
```

因此第 6.1 节中的 2D DSC、IoU 和可用 MAE 可以定档。

#### 3D

4 个 3D 数据集均已确认：

```text
evaluation_space = native
volume_reconstruction = all slices sorted by slice_idx and stacked in native space
distance_unit = mm
debug_resize_enabled = false
connectivity = 1
empty_distance_policy = nan
```

同一数据集下三套系统的 `spacing_zyx` 完全一致。Synapse 已使用真实 spacing，而不是 `(1,1,1)`。

因此第 6.2 节中的 3D DSC、HD95 和 ASSD 可以定档。

#### 定档范围

本次定档覆盖：

- 11 个正式数据集的 Student test 指标；
- Frozen Baseline、Idea1、Frozen Upper 三组主结果；
- 当前代码、目录血缘与正式运行路径；
- Teacher-space 伪标签可视化输出。

本次定档不表示已经完成：

- 多随机种子统计；
- Random / Hard-NoFT 等完整机制消融；
- 所有可视化的论文级人工解释；
- 新 Idea2 的设计与实验。

---


## 7. BTCV 已完成并定档

正式结果文件：

```text
$FORMAL_ROOT/Swin-UMamba-main/work_dir/
idea1_hard_full_medsam_ft/btcv/fold_0/eval_3d/eval_3d_summary.json
```

当前锁定结果：

```text
DSC   = 0.5936
HD95  = 20.0188 mm
ASSD  = 5.3980 mm
Cases = 6
```

BTCV 已完成训练、推理、native-space 3D 重建和正式评估，不需要重新训练或重新评估。

---

## 8. 全部数据集正确可视化协议

### 8.1 伪标签质量：统一使用 Teacher Space

```text
Teacher image:
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/teacher_npy/imgs

Teacher GT:
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/teacher_npy/gts

Box-only pseudo:
/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed/<dataset>/fold_0/
pseudo_teacher/tri_train_boxonly

Idea1 pseudo:
$FORMAL_ROOT/MedSAM-main/data/processed/<dataset>/fold_0/
pseudo_teacher/tri_train_idea1_hard_full_medsam_ft
```

禁止：

```text
teacher image + native GT
teacher pseudo + student GT
native image + teacher pseudo
```

### 8.2 Student 预测：使用 Native/Test Space

```text
<work_dir>/<dataset>/fold_0/pred_test/
```

- 2D：预测与 test GT 对齐；
- 3D：恢复 native slice 尺寸，按 `slice_idx` 重建 volume；
- 使用真实 spacing；
- evaluation 不依赖 prompt 或 pseudo。

### 8.3 本执行包输出

`visualize_all_teacher_space.py` 会对 11 个数据集生成：

```text
visualizations/final_teacher_space/
├── global_summary.csv
├── global_summary.json
└── pseudo_quality/<dataset>/
    ├── discovery_audit.json
    ├── per_sample_metrics.csv
    ├── summary.json
    ├── 01_dice_distribution.png
    ├── 02_delta_dice_distribution.png
    ├── 03_baseline_vs_idea_scatter.png
    ├── 04_error_profile.png
    └── qualitative/
        ├── top_gain/
        ├── top_loss/
        ├── hard_cases/
        └── representative/
```

可视化标签固定为：

```text
Box-only rerun
Idea1
```

---


## 9. 最终定档、Git 提交与后续可视化审阅

### 9.1 定档前必须保存的审计文件

建议将以下文件保存到：

```text
$FORMAL_ROOT/summaries/final/
```

至少包括：

```text
audit_2d_protocol.txt
comparison_2d_locked.csv
comparison_2d_locked.md
comparison_3d_overall_macro_mean.csv
comparison_3d_overall_macro_mean.md
```

同时在 `00_manifest/` 中增加：

```text
FINAL_RESULT_LOCK.md
```

记录：

- 定档日期；
- 2D/3D 正式字段；
- 结果源 JSON；
- 2D prediction set 一致性结论；
- 3D spacing/protocol 一致性结论；
- 当前 Git commit ID。


### 9.2 Git 推送完成状态

已完成：

```text
MedSAM-research
branch: idea1-hard-full-medsam-ft
commit: 0dbc2da6adf1b4f875af7d09a31cdee12b59ad98
remote branch: 已创建并推送

Swin-UMamba-research
branch: idea1-hard-full-medsam-ft
commit: 48462cd184f56205873682f97c8e77079710d6c4
remote branch: 已创建并推送
```

Swin-UMamba 已确认：

```text
HEAD == origin/idea1-hard-full-medsam-ft
```

当前代码提交阶段完成。后续只允许：

- 补充文档；
- 修复明确发现的代码问题；
- 增加可视化分析记录；
- 新建独立 Idea2 分支。

不要直接改写或 force-push 已推送提交。

若将本指南加入 GitHub，使用单独的 documentation-only commit：

```bash
git -C "$MEDSAM_GIT" add docs/
git -C "$MEDSAM_GIT" commit \
  -m "docs: add final Idea1 structure and result lock guide"
git -C "$MEDSAM_GIT" push

git -C "$SWIN_GIT" add docs/
git -C "$SWIN_GIT" commit \
  -m "docs: add final Idea1 structure and result lock guide"
git -C "$SWIN_GIT" push
```

### 9.3 可视化人工审阅顺序

统一从：

```text
$FORMAL_ROOT/visualizations/final_teacher_space/pseudo_quality/
```

开始。

每个数据集按以下顺序查看：

```text
qualitative/top_loss/montage.png
qualitative/hard_cases/montage.png
qualitative/representative/montage.png
qualitative/top_gain/montage.png
01_dice_distribution.png
02_delta_dice_distribution.png
03_baseline_vs_idea_scatter.png
04_error_profile.png
```

优先级：

1. TG3K：伪标签增益异常大，优先排查 unknown、前景扩张和配准问题；
2. Synapse：Student DSC 基本持平且边界指标变差；
3. BTCV：检查多器官粘连、小器官漏分和二值前景并集掩盖问题；
4. Kvasir-SEG、CVC-ClinicDB：2D Student 指标下降；
5. ACDC：确认大幅提升是否来自真实结构改善；
6. 其余数据集。

每张图记录：

```text
dataset
sample
category
Box-only 主要错误
Idea1 改善
Idea1 新增错误
是否空间对齐
是否存在前景扩张
是否存在漏分
是否存在类别混淆
是否可作为论文案例
```

## 10. 当前最终锁定记录

```text
Method:
idea1_hard_full_medsam_ft

Branch:
idea1-hard-full-medsam-ft

MedSAM commit:
0dbc2da6adf1b4f875af7d09a31cdee12b59ad98

Swin-UMamba commit:
48462cd184f56205873682f97c8e77079710d6c4

2D protocol:
native-space test evaluation
dice_macro.mean / iou_macro.mean

3D protocol:
native-space volume reconstruction
real spacing
overall_macro_mean

Visualization:
11/11 Teacher-space pseudo-label visualizations completed
```

当前阶段：

```text
代码与正式结果已定档
两个 Git 远端分支已创建
下一阶段进入可视化人工审阅与失败模式分析
```

## 11. 禁止事项

- 不在 Frozen runtime 中开发 Idea2；
- 不把 `formal_runs` 初始化成 Git 仓库；
- 不提交 `work_dir`、checkpoint、`.npy`、`.npz`；
- 不把 `baseline_boxonly` 与 Frozen Baseline 混称；
- 不使用 Native GT 评价 Teacher-space pseudo；
- 不生成 test pseudo；
- 不在 BTCV 未完成时提交最终结果版本；
- 不从 Idea1 直接派生独立 Idea2，除非 Idea2 明确扩展 Idea1。
