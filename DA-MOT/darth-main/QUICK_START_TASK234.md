# Task 2/3/4 快速启动指南

## 当前状态

✅ **所有脚本已修复并就绪**  
⏸️ **等待 Task 1 完成**（预计明天 6/27 晚上 21:00）  
📍 **GPU 正在运行 Task 1**: Night-related baselines rerun

---

## 修复内容摘要

| Bug | 影响 | 状态 |
|-----|------|------|
| 训练用 `runner.max_epochs`（无效） | 只跑 12 epoch，`epoch_20.pth` 缺失 | ✅ 已修复为 `total_epochs` |
| 缺少 `find_unused_parameters` | DDP 崩溃 | ✅ 已注入 |
| 测试 worker=2，无加速参数 | GPU 闲置，测试慢 | ✅ 改为 worker=8 + fuse-conv-bn + cudnn |
| GPU_NUM=2, batch=1 | 资源浪费 | ✅ 改为 GPU_NUM=10, batch=14 |

**详细报告**: 见 `TASK_2_3_4_PREPARATION_REPORT.md`

---

## 启动命令（Task 1 完成后）

### 1️⃣ Seq3 DARTH baseline (Task 2)

**用途**: 固定顺序域适应 clear → dawn_dusk → night → partly_cloudy

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main

# Dry-run 预览（推荐先跑）
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh

# 正式运行（后台）
nohup bash scripts/run_bdd_clear_seq3_darth.sh > work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/run.log 2>&1 &
echo "Started PID: $!"
```

**预计耗时**: ~7 小时  
**结果路径**: `work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/`

---

### 2️⃣ Seq3 Episode (Task 3 消融)

**用途**: 连续 episode 训练（teacher keep across stages）

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main

# Dry-run 预览
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_episode.sh

# 正式运行（后台）
nohup bash scripts/run_bdd_clear_seq3_episode.sh > work_dirs/bdd_seq3_episode/run.log 2>&1 &
echo "Started PID: $!"
```

**预计耗时**: ~7.5 小时  
**结果路径**: `work_dirs/bdd_seq3_episode/clear_to_dawn_dusk_night_partly_cloudy/`

---

## 监控进度

### 查看当前训练日志
```bash
# Seq3 DARTH
tail -f work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/adapt/stage01_dawn_dusk/train.log

# Seq3 Episode
tail -f work_dirs/bdd_seq3_episode/clear_to_dawn_dusk_night_partly_cloudy/logs/train_episode.log
```

### 检查 GPU 利用率
```bash
watch -n 5 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

### 查看已保存的 checkpoint
```bash
# Seq3 DARTH
ls work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/adapt/stage*/epoch_*.pth

# Seq3 Episode
ls work_dirs/bdd_seq3_episode/clear_to_dawn_dusk_night_partly_cloudy/stage_snapshots/*.pth
```

---

## 自定义参数

### 如果显存不足（OOM）
```bash
SAMPLES_PER_GPU=10 bash scripts/run_bdd_clear_seq3_darth.sh
```

### 强制重跑（忽略已有结果）
```bash
FORCE=1 bash scripts/run_bdd_clear_seq3_darth.sh
```

### 自定义 GPU
```bash
GPU_IDS="7,6,5,4" GPU_NUM=4 SAMPLES_PER_GPU=7 bash scripts/run_bdd_clear_seq3_darth.sh
```

### 指定 clear source checkpoint
```bash
SOURCE_CKPT="path/to/clear_epoch_20.pth" bash scripts/run_bdd_clear_seq3_darth.sh
```

---

## 结果收集

### Seq3 DARTH
```bash
python tools/collect_bdd_seq3_results.py work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy

# 查看摘要
cat work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/summaries/seq3_results_summary.md
```

### Seq3 Episode
```bash
python tools/collect_bdd_seq3_results.py work_dirs/bdd_seq3_episode/clear_to_dawn_dusk_night_partly_cloudy

# 查看摘要
cat work_dirs/bdd_seq3_episode/clear_to_dawn_dusk_night_partly_cloudy/summaries/seq3_results_summary.md
```

---

## 故障排查

### 训练崩溃：DDP reduction 报错
```bash
# 检查 find_unused_parameters 是否生效
grep "find_unused_parameters = True" work_dirs/.../train.log

# 如果未生效，检查脚本版本
grep "find_unused_parameters" scripts/run_bdd_clear_seq3_darth.sh
```

### 训练只跑到 12 epoch
```bash
# 检查 total_epochs 是否生效
grep "workflow.*max:.*20" work_dirs/.../train.log

# 应该看到: workflow: [('train', 1)], max: 20 epochs
```

### 测试很慢（GPU 闲置）
```bash
# 检查测试进程的参数
ps aux | grep test.py | grep -oE "workers_per_gpu=[0-9]+|fuse-conv-bn"

# 应该看到: workers_per_gpu=8 和 fuse-conv-bn
```

---

## 时间线预估

| 时间点 | 事件 |
|--------|------|
| **现在** | Task 1 运行中（TTA 训练第 1/12 个） |
| **明天 6/27 21:00** | Task 1 完成 |
| **明天 6/27 21:30** | 启动 Seq3 DARTH |
| **6/28 凌晨 04:30** | Seq3 DARTH 完成，启动 Seq3 Episode |
| **6/28 中午 12:00** | Seq3 Episode 完成 |
| **6/28 下午** | 收集结果、生成报告 |

---

## 下一步（Task 4：顺序消融）

如果需要测试不同顺序（例如 clear → night → dawn_dusk → partly_cloudy），可以修改脚本顶部：

```bash
# run_bdd_clear_seq3_darth.sh
TARGET_ORDER=("night" "dawn_dusk" "partly_cloudy")  # 改变顺序
```

或者复制脚本创建新的顺序变体。

---

**准备完毕！等待 Task 1 完成后即可启动。** 🚀
