# Task 2/3/4 脚本准备报告

**日期**: 2026-06-26  
**状态**: ✅ 已完成，等待 Task 1 完成后启动  
**修复人员**: AI Assistant

---

## 执行摘要

已全面检查并修复 Task 2/3/4 相关脚本的所有已知 bug，确保与 Task 1（night-related baselines）使用相同的修复方案。**所有脚本已就绪，但尚未启动**（GPU 正在运行 Task 1）。

---

## 修复的脚本清单

### 1. `scripts/run_bdd_clear_seq3_darth.sh`
**用途**: Task 2/3/4 - Sequential 3-Target Domain Adaptation (clear → dawn_dusk → night → partly_cloudy)  
**修复类型**: 训练参数 + 测试加速 + GPU 配置

### 2. `scripts/run_bdd_clear_seq3_episode.sh`  
**用途**: Task 2/3/4 - Episode-based Continuous Adaptation  
**修复类型**: 测试加速 + GPU 配置

### 3. `tools/run/train_seq3_episode.py`
**用途**: Continuous episode trainer (单进程连续适应)  
**修复类型**: DDP 参数修复

### 4. `scripts/run_bdd_night_related_baselines_rerun.sh` ✅
**用途**: Task 1 - 正在运行中  
**状态**: 已于之前修复，预计明天（6/27）晚上 21:00 完成

---

## 详细修复内容

### 修复 1: 训练 `total_epochs` bug（与 Task 1 同根因）

**问题**: mmtrack `apis/train.py:175` 使用 `cfg.total_epochs` 驱动训练循环，**完全不读** `cfg.runner.max_epochs`。

**影响**: 
- 脚本传 `--cfg-options runner.max_epochs=20` 是空操作
- DARTH 配置自带 `total_epochs=12` → 训练被截断到 12 epoch
- `epoch_20.pth` 永远不生成 → 后续评测缺失

**修复**: 所有训练调用增加 `total_epochs=20`（保留 `runner.max_epochs=20` 兼容）

```bash
# 修复前（无效）
--cfg-options runner.max_epochs=20 ...

# 修复后（有效）
--cfg-options total_epochs=20 runner.max_epochs=20 ...
```

**修复文件**:
- ✅ `run_bdd_clear_seq3_darth.sh` (训练调用)
- ✅ `train_seq3_episode.py` (已正确处理 `cfg.total_epochs`)

---

### 修复 2: DDP `find_unused_parameters` 缺失（与 Task 1 同根因）

**问题**: DARTH teacher-student 结构有部分参数在某些 iteration 不参与 loss 计算，多卡 DDP 默认要求所有参数参与梯度规约 → 报错崩溃：

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
```

**修复**: 所有训练调用/配置生成处注入 `find_unused_parameters=True`

```bash
# 训练调用
--cfg-options find_unused_parameters=True ...
```

```python
# train_seq3_episode.py
cfg.find_unused_parameters=True
```

**修复文件**:
- ✅ `run_bdd_clear_seq3_darth.sh` (训练调用)
- ✅ `train_seq3_episode.py` (stage_cfg 函数)

---

### 修复 3: 测试加速参数缺失

**问题**: 测试时 GPU 闲置（0-21%），显存只用 2.5G/40G，瓶颈在数据加载。

**根因**: 
- `workers_per_gpu=2`（训练用的值，测试应该更高）
- 缺少 `--fuse-conv-bn`（conv+bn 融合加速）
- 缺少 `cudnn_benchmark=True`（固定输入尺寸选最优算法）

**修复**: 所有测试调用增加：
1. 新增独立变量 `TEST_WORKERS_PER_GPU=8`（不影响训练）
2. 加 `--fuse-conv-bn` flag
3. 加 `cudnn_benchmark=True` 到 `--cfg-options`

```bash
# 修复前
tools/run/test.py ... \
  --cfg-options data.workers_per_gpu=${WORKERS_PER_GPU}

# 修复后
tools/run/test.py ... \
  --fuse-conv-bn \
  --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True
```

**精度影响**: 
- `workers` 提升：**零影响**（逐 bit 相同）
- `fuse-conv-bn`：数学等价，浮点末位误差（不可见）
- `cudnn_benchmark`：末位浮点（不可见）
- **不使用 fp16**（保精度要求）

**修复文件**:
- ✅ `run_bdd_clear_seq3_darth.sh` (source-only eval + stage eval，共 3 处)
- ✅ `run_bdd_clear_seq3_episode.sh` (snapshot eval)

---

### 修复 4: GPU 默认配置不匹配

**问题**: 脚本默认配置与实际硬件和 Task 1 标准不一致

| 参数 | 修复前 (seq3) | 修复前 (episode) | 修复后（统一） | Task 1 标准 |
|------|--------------|-----------------|--------------|-----------|
| `GPU_IDS` | 无 | 9,8,7,6,5,4,3,2 (8张) | 9,8,7,6,5,4,3,2,1,0 | ✓ |
| `GPU_NUM` | 2 | 8 | **10** | ✓ |
| `SAMPLES_PER_GPU` | 1 | 10 | **14** | ✓ |
| `WORKERS_PER_GPU` | 2 | 2 | 2 (训练) | ✓ |
| `TEST_WORKERS_PER_GPU` | 无 | 无 | **8** (测试) | ✓ |

**修复**: 所有脚本顶部配置块统一为：

```bash
GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-14}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
```

所有训练/测试调用前显式加：

```bash
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch ...
```

**修复文件**:
- ✅ `run_bdd_clear_seq3_darth.sh`
- ✅ `run_bdd_clear_seq3_episode.sh`

---

## 语法验证

所有修改的脚本已通过语法检查：

```bash
✓ scripts/run_bdd_clear_seq3_darth.sh
✓ scripts/run_bdd_clear_seq3_episode.sh  
✓ tools/run/train_seq3_episode.py
✓ scripts/run_bdd_night_related_baselines_rerun.sh (Task 1)
```

---

## 修复前后对比示例

### 训练调用（run_bdd_clear_seq3_darth.sh Stage 适应）

**修复前**:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  tools/run/train.py ${adapt_cfg} \
  --cfg-options \
    runner.max_epochs=20 \
    load_from=${load_ckpt} \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=2
```

**修复后**:
```bash
CUDA_VISIBLE_DEVICES=9,8,7,6,5,4,3,2,1,0 python -m torch.distributed.launch \
  --nproc_per_node=10 \
  tools/run/train.py ${adapt_cfg} \
  --cfg-options \
    total_epochs=20 \
    runner.max_epochs=20 \
    find_unused_parameters=True \
    load_from=${load_ckpt} \
    data.samples_per_gpu=14 \
    data.workers_per_gpu=2
```

### 测试调用（run_bdd_clear_seq3_darth.sh Stage 评估）

**修复前**:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  tools/run/test.py ${eval_cfg} \
  --checkpoint ${current_ckpt} \
  --eval bbox track \
  --cfg-options data.workers_per_gpu=2
```

**修复后**:
```bash
CUDA_VISIBLE_DEVICES=9,8,7,6,5,4,3,2,1,0 python -m torch.distributed.launch \
  --nproc_per_node=10 \
  tools/run/test.py ${eval_cfg} \
  --checkpoint ${current_ckpt} \
  --eval bbox track \
  --fuse-conv-bn \
  --cfg-options data.workers_per_gpu=8 cudnn_benchmark=True
```

---

## 预估耗时（Task 2/3/4）

基于 Task 1 的实测数据推算：

| 任务 | 配置 | 单次耗时 | 合计 |
|------|------|---------|------|
| **Seq3 DARTH** (run_bdd_clear_seq3_darth.sh) | | | |
| - Source-only eval | 3 域 × epoch20 | ~1.5 分/域 | ~5 分钟 |
| - Stage 1: dawn_dusk | 20 epoch 适应 | ~2h20m | 2h20m |
| - Stage 1 eval | 3 域 | ~1.5 分/域 | ~5 分钟 |
| - Stage 2: night | 20 epoch 适应 | ~2h20m | 2h20m |
| - Stage 2 eval | 3 域 | ~1.5 分/域 | ~5 分钟 |
| - Stage 3: partly_cloudy | 20 epoch 适应 | ~2h20m | 2h20m |
| - Stage 3 eval | 3 域 | ~1.5 分/域 | ~5 分钟 |
| **Seq3 DARTH 小计** | | | **~7 小时** |

| **Seq3 Episode** (run_bdd_clear_seq3_episode.sh) | | | |
| - Continuous training | 3 stage × 20 epoch | ~2h20m/stage | ~7 小时 |
| - Snapshot eval | 4 snapshot × 3 域 | ~1.5 分/域 | ~20 分钟 |
| **Seq3 Episode 小计** | | | **~7.5 小时** |

**Task 2/3/4 总计**: ~14-15 小时（串行执行，每次吃满 10 卡）

---

## 启动顺序建议

等 Task 1 完成后（预计明天 6/27 晚上 21:00），按以下顺序启动：

### 1. 验收 Task 1 结果
```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
ROOT="work_dirs/bdd_baseline_rerun/night_related_all"

# 检查完成标记
find "$ROOT" -name "*.marker" | wc -l  # 应该 13 (source-only) + 12 (tta train) + 36 (tta eval) = 61

# 查看合并表
cat "$ROOT/summaries/complete_metrics_all_epochs_night_rerun_merged.csv" | wc -l

# 查看摘要
cat "$ROOT/summaries/night_related_summary.md"
```

### 2. 启动 Seq3 DARTH baseline
```bash
# 默认配置（10 卡 × 14 batch）
bash scripts/run_bdd_clear_seq3_darth.sh

# 或 dry-run 预览
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh
```

预计耗时：~7 小时  
预计完成：6/28 凌晨 04:00

### 3. 启动 Seq3 Episode（连续训练消融）
```bash
bash scripts/run_bdd_clear_seq3_episode.sh

# 或自定义 source checkpoint
SOURCE_CKPT=path/to/clear_epoch20.pth bash scripts/run_bdd_clear_seq3_episode.sh
```

预计耗时：~7.5 小时  
预计完成：6/28 中午 11:30

---

## 故障排查

### 如果训练崩溃，检查：

1. **DDP reduction 报错** → 确认 `find_unused_parameters=True` 生效
   ```bash
   grep "find_unused_parameters = True" work_dirs/.../train.log
   ```

2. **只跑到 12 epoch** → 确认 `total_epochs=20` 生效
   ```bash
   grep "workflow.*max:.*20" work_dirs/.../train.log
   ```

3. **显存 OOM** → 降低 batch
   ```bash
   SAMPLES_PER_GPU=10 bash scripts/run_bdd_clear_seq3_darth.sh
   ```

### 如果测试很慢，检查：

```bash
# 应该看到这些参数
ps aux | grep test.py | grep -oE "workers_per_gpu=[0-9]+|fuse-conv-bn"
# 预期输出: workers_per_gpu=8  fuse-conv-bn
```

---

## 与 Task 1 的一致性确认

| 配置项 | Task 1 (night-related) | Task 2/3/4 (seq3) | 状态 |
|--------|----------------------|------------------|------|
| `GPU_IDS` | 9,8,7,6,5,4,3,2,1,0 | 9,8,7,6,5,4,3,2,1,0 | ✅ 一致 |
| `GPU_NUM` | 10 | 10 | ✅ 一致 |
| `SAMPLES_PER_GPU` (train) | 14 | 14 | ✅ 一致 |
| `WORKERS_PER_GPU` (train) | 2 | 2 | ✅ 一致 |
| `TEST_WORKERS_PER_GPU` (eval) | 8 | 8 | ✅ 一致 |
| `total_epochs` 修复 | ✅ 已修复 | ✅ 已修复 | ✅ 一致 |
| `find_unused_parameters` 修复 | ✅ 已修复 | ✅ 已修复 | ✅ 一致 |
| 测试加速 (fuse/cudnn) | ✅ 已加 | ✅ 已加 | ✅ 一致 |

---

## 下一步行动

- [x] 修复 Task 2/3/4 所有已知 bug
- [x] 语法检查通过
- [x] 生成修复报告
- [ ] **等待 Task 1 完成**（明天 6/27 晚上 21:00）
- [ ] 验收 Task 1 结果和合并表
- [ ] 启动 `run_bdd_clear_seq3_darth.sh`（~7h）
- [ ] 启动 `run_bdd_clear_seq3_episode.sh`（~7.5h）
- [ ] 收集 Task 2/3/4 结果
- [ ] 最终实验报告

---

**备注**: 所有脚本已就绪，**无需等待即可启动**（GPU 空闲后）。修复方案与 Task 1 完全一致，确保结果可比性和稳定性。
