# DARTH BDD 7-Domain Full-Matrix Experiment — Agent Context

> 最后更新：2026-06-14  
> 供 AI Agent 直接读取并继续工作，不要跳过任何小节。

---

## 1. 项目目标

在 BDD100K 的 7 个天气/光照域（clear、dawn_dusk、night、partly_cloudy、overcast、rainy、snowy）上跑**全矩阵域自适应跟踪实验**：

1. 每个域训练一个 source-only QDTrack（20 epoch）
2. 每个 source 模型在**所有 7 个目标域**（含自身）上测评，得到 source-only baseline（7×7=49 个）
3. 每对跨域（6×7=42 对）用 DARTH 做 TTA 自适应训练（12 epoch），在 epoch 6 和 epoch 12 各测评一次（42×2=84 个）
4. 汇总全部结果，输出跨域性能矩阵

框架：MMTracking + DARTH (Domain-Adaptive Real-Time Tracking with Hierarchical distillation)

---

## 2. 关键路径

| 类型 | 路径 |
|------|------|
| 项目根目录 | `/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main` |
| 虚拟环境 | `conda activate DARTH`（Python 3.7，PyTorch 1.13.1+cu117）|
| BDD100K 数据根目录 | `/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_<domain>/` |
| 标注文件 | `data/bdd_da/annotations/box_track_20/box_track_<domain>_<split>_cocofmt.json` |
| Source 模型 checkpoint | `work_dirs/bdd_7domain/source/<domain>/qdtrack_20e/**/epoch_20.pth` |
| TTA checkpoint | `work_dirs/bdd_7domain/tta/<src>_to_<tgt>/**/epoch_<N>.pth` |
| Pipeline 状态 | `work_dirs/bdd_7domain/state/jobs.csv` |
| Pipeline 日志 | `work_dirs/bdd_7domain/auto_run_logs/auto_run_all_*.log` |
| 实时日志 | `/tmp/bdd_7domain_pipeline.log` |

---

## 3. 关键文件

| 文件 | 作用 |
|------|------|
| `tools/bdd_7domain_manager.py` | 核心 pipeline 管理器（任务状态读写、命令生成、执行） |
| `scripts/auto_run_all_pending.sh` | 自动串行跑所有 pending/interrupted 任务的主循环脚本 |
| `work_dirs/bdd_7domain/state/jobs.csv` | 所有 187 个任务的状态表（唯一数据源） |
| `configs/mot/qdtrack/bdd_7domain/` | 各域的 source QDTrack 训练/测评配置 |
| `configs/adapters/darth/bdd_7domain/` | 各域的 DARTH TTA 训练配置 |
| `tools/run/train.py` | 训练入口 |
| `tools/run/test.py` | 测评入口 |
| `tools/analysis/parse_bdd_7domain_results.py` | 汇总结果 |

---

## 4. 验证集视频数量（关键约束）

每个域的验证集视频数量决定了最大可用 GPU 数，**GPU 数必须 ≤ 视频数**，否则报 `ValueError: only N videos loaded, but M gpus were given`：

| 域 | 验证视频数 | 最大 GPU 数 |
|----|-----------|------------|
| clear | 36 | 10 |
| dawn_dusk | 20 | 8 |
| night | 71 | 10 |
| partly_cloudy | 9 | 8 |
| overcast | 22 | 8 |
| rainy | 7 | **6** |
| snowy | 15 | 8 |

`build_cmd()` 中已为 `tta_adapt`、`source_train`、`source_only_eval`、`tta_eval` 都按此配置了 `gpu_map`。

---

## 5. 任务结构（共 187 个）

```
prepare_data        (1)   ✅
generate_config     (1)   ✅
batch_probe_source  (1)   ✅
batch_probe_tta     (1)   ✅
source_train        (7)   ✅  7个域各训练20epoch
source_only_eval   (49)   ✅  7×7全矩阵（含同域self-eval）
tta_adapt          (42)   🔄  6✅ 1运行中 34待执行 1失败
tta_eval           (84)   🔄  3✅ 81待执行（依赖tta_adapt）
summary             (1)   ✅（需最后重跑）
```

**截至 2026-06-14 18:54：**
- 已完成：70/187
- 运行中：`tta_dawn_dusk_to_night_12e`
- 失败：1个（`tta_dawn_dusk_to_clear_12e`，需排查）
- 待执行：115个

已完成的 tta_adapt（6个，均为 clear 作为 source）：
- `tta_clear_to_dawn_dusk_12e`
- `tta_clear_to_night_12e`
- `tta_clear_to_partly_cloudy_12e`
- `tta_clear_to_overcast_12e`
- `tta_clear_to_rainy_12e`
- `tta_clear_to_snowy_12e`

---

## 6. 本次 Session 已完成的所有修改

### 6.1 `tools/bdd_7domain_manager.py`

**问题 1：tta_adapt 全部失败 — GPU 数超过验证集视频数**

原来 `tta_adapt` 硬编码 `NGPU=10`，导致 `ValueError: only 7 videos loaded, but 10 gpus were given`。

修复：在 `build_cmd()` 的 `tta_adapt` 分支加入 `gpu_map`，按目标域动态限制 GPU 数。

**问题 2：孤儿进程导致 OOM Kill**

原来用 `subprocess.run(shell=True, ...)`，`conda activate` 执行时会 `exec` 替换 shell，导致进程树断裂。shell 退出后 GPU worker 被 `init` 接管变成孤儿，占满显存，下一个任务启动时被 OOM Killer 用 SIGKILL (-9) 杀掉。

修复：改用 `subprocess.Popen(..., preexec_fn=os.setsid)`，任务结束后在 `finally` 块中调用 `os.killpg()` 杀掉整个进程组，确保 GPU 显存 100% 释放。

**问题 3：同域测评缺失**

原 `init()` 中 `if s!=t` 跳过了同域测评（如 `clear_to_clear`）。

修复：移除该条件，现在 source_only_eval 是完整的 7×7=49 个任务。已向 jobs.csv 追加 7 个同域任务并重新排序。

**问题 4：`interrupted` 状态被 pipeline 跳过**

`get_next_job()` 只查 `pending`，`interrupted` 的任务会被跳过导致孤儿 checkpoint 无法续训。

修复：`get_next_job()` 同时处理 `pending` 和 `interrupted` 状态。

**新增 `recover` 子命令**

```bash
python3 tools/bdd_7domain_manager.py recover
```

检查 checkpoint 是否实际存在来识别被误判为 failed 的任务，自动恢复状态。

### 6.2 `scripts/auto_run_all_pending.sh`

- 移除了顶层 `set -e`（它会在任务失败时直接退出整个脚本，GPU 子进程变孤儿）
- 改用 `get_next_job()` 函数同时处理 `pending` 和 `interrupted`
- 新增 `wait_gpu_free()` 函数：每个任务启动前等待上一个任务的 GPU worker 全部退出（最多等 120 秒，超时则强制 kill）
- 新增连续失败计数器，连续失败 5 次才真正停止 pipeline

---

## 7. 已知问题和根因

### 孤儿进程（已修复）

根因：`subprocess.run(shell=True)` + `conda activate` 的组合导致进程树断裂。修复方案见 6.1。

### OOM Kill / SIGKILL (-9)（已修复）

根因：孤儿进程残留显存 → 新任务启动时被系统 OOM Killer 杀掉。随孤儿进程问题修复而消除。

### `tta_dawn_dusk_to_clear_12e` 当前 failed（待处理）

当前有 1 个 failed 任务。在重启 pipeline 前，应运行：

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
python3 tools/bdd_7domain_manager.py recover
```

若 recover 无法自动处理，手动重置：

```bash
python3 -c "
import csv
from pathlib import Path
STATE = Path('work_dirs/bdd_7domain/state')
FIELDS = 'job_id,stage,source,target,source_epoch,tta_epoch,mode,status,start_time,end_time,gpu_ids,samples_per_gpu,work_dir,checkpoint_path,log_path,result_path,done_marker,running_marker,fail_marker,pid_path,error_summary_path,command'.split(',')
jobs = list(csv.DictReader(open(STATE/'jobs.csv')))
for j in jobs:
    if j['status'] in ('failed', 'interrupted'):
        for mk in (j['fail_marker'], j['running_marker']):
            p = Path(mk)
            if p.exists(): p.unlink()
        j['status'] = 'pending'
        j['end_time'] = ''
        print(f'Reset: {j[\"job_id\"]}')
with (STATE/'jobs.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, FIELDS); w.writeheader()
    for j in jobs: w.writerow({k: j.get(k,'') for k in FIELDS})
"
```

---

## 8. 下一步任务（按顺序）

pipeline 会自动完成，**无需手动干预**，除非 pipeline 停止：

1. 运行中：`tta_dawn_dusk_to_night_12e`（约 2 小时）
2. 自动接续：`tta_dawn_dusk_to_*`（5个）→ `tta_night_to_*`（6个）→ `tta_partly_cloudy_to_*`（6个）→ `tta_overcast_to_*`（6个）→ `tta_rainy_to_*`（6个）→ `tta_snowy_to_*`（6个）
3. 每个 `tta_adapt` 完成后，pipeline 自动接续对应的 `tta_eval_epoch6` 和 `tta_eval_epoch12`
4. 全部完成后，手动重跑 summary：`python3 tools/bdd_7domain_manager.py run-job --job-id summary_parse_all_results`

---

## 9. 中断后如何恢复

### 检查状态

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
python3 tools/bdd_7domain_manager.py status
```

### 清理孤儿进程

```bash
ps aux | grep -E "(train\.py|test\.py|auto_run)" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

### 重置 failed/interrupted 任务

```bash
python3 tools/bdd_7domain_manager.py recover
```

若 recover 不够用，使用第 7 节中的手动重置脚本。

### 重启 pipeline

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
nohup bash scripts/auto_run_all_pending.sh > /tmp/bdd_7domain_pipeline.log 2>&1 &
```

### 查看实时进度

```bash
tail -f /tmp/bdd_7domain_pipeline.log
# 或查看最新 auto_run 日志
tail -f work_dirs/bdd_7domain/auto_run_logs/auto_run_all_$(ls -t work_dirs/bdd_7domain/auto_run_logs/ | head -1)
```

### 查看某个任务的训练日志

```bash
# 查状态，找到 log 路径
python3 tools/bdd_7domain_manager.py status
# 直接看最新的 tta 日志
ls -t work_dirs/bdd_7domain/tta/<src>_to_<tgt>/*.log | head -1 | xargs tail -f
```

---

## 10. 所有重要命令

```bash
# 环境准备
conda activate DARTH
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main

# 查看 pipeline 状态
python3 tools/bdd_7domain_manager.py status

# 恢复被误判为失败的任务
python3 tools/bdd_7domain_manager.py recover

# 手动跑单个任务
python3 tools/bdd_7domain_manager.py run-job --job-id <job_id>

# 启动/重启整个 pipeline（后台）
nohup bash scripts/auto_run_all_pending.sh > /tmp/bdd_7domain_pipeline.log 2>&1 &

# 清理所有孤儿 GPU 进程
ps aux | grep -E "(train\.py|test\.py)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 查看实时日志
tail -f /tmp/bdd_7domain_pipeline.log

# 查看当前 GPU 使用情况
nvidia-smi

# 查看各域验证集视频数
python3 -c "
import json; from pathlib import Path
ann = Path('data/bdd_da/annotations/box_track_20')
for d in ['clear','dawn_dusk','night','partly_cloudy','overcast','rainy','snowy']:
    data = json.loads((ann/f'box_track_{d}_val_cocofmt.json').read_text())
    print(f'{d}: {len(data[\"videos\"])} videos')
"
```

---

## 11. 绝对不要做的事

1. **不要手动修改 `jobs.csv`**（除非按照第 7 节的脚本格式），否则会破坏 pipeline 状态
2. **不要同时启动多个 `auto_run_all_pending.sh` 实例**，否则两个 pipeline 会竞争跑同一个任务，产生大量重复进程
3. **不要用 `set -e` 的 shell 脚本包裹 manager**，会导致孤儿进程（已修复，不要回滚）
4. **不要在 tta_adapt 任务还在运行时手动 kill 再重启**，会导致 checkpoint 损坏；应等任务自然结束或让 pipeline 标记为 interrupted 后自动续训
5. **不要直接跑 `python3 tools/bdd_7domain_manager.py init --force`**，这会清空所有任务状态
6. **不要改变 GPU map 设置**（`rainy` 最多 6 GPU，`partly_cloudy`/`snowy`/`dawn_dusk`/`overcast` 最多 8 GPU），否则会触发 `ValueError: only N videos loaded`
7. **summary 任务虽然显示 completed，但它是在实验还没全部完成时跑的**，最后所有任务完成后必须重新跑一次

---

## 12. 结果目录结构（完成后）

```
work_dirs/bdd_7domain/
├── source/<domain>/qdtrack_20e/          # source 训练 checkpoints
├── source_only_eval/<src>_epoch20_to_<tgt>/  # source-only 测评结果
├── tta/<src>_to_<tgt>/                   # TTA 训练 checkpoints (epoch_1..12)
├── tta_eval/<src>_to_<tgt>_epoch<N>/     # TTA 测评结果
├── summaries/                             # 汇总结果
└── state/                                 # pipeline 状态
    ├── jobs.csv                           # 任务状态表
    └── pipeline_state.json
```

---

## 13. 代码修改摘要（diff 级别）

### `tools/bdd_7domain_manager.py` 核心改动

**`build_cmd()` - tta_adapt 分支**：原来 `NGPU=10` 硬编码，现在：
```python
gpu_map={'rainy':6,'partly_cloudy':8,'snowy':8,'dawn_dusk':8,'overcast':8,'clear':10,'night':10}
ngpu=gpu_map.get(t,8)
resume_ckpt=get_resume_ckpt(wd)
resume_flag=f' --resume-from {resume_ckpt}' if resume_ckpt else ''
```

**`run_job()` - 进程管理**：原来 `subprocess.run(shell=True)`，现在：
```python
proc=subprocess.Popen(shell_cmd,shell=True,...,preexec_fn=os.setsid)
try:
    proc.wait()
finally:
    pgid=os.getpgid(proc.pid)
    os.killpg(pgid, SIGTERM); time.sleep(3); os.killpg(pgid, SIGKILL)
```

**`init()` - source_only_eval**：移除了 `if s!=t` 条件，现在包含同域测评。

**新增 `recover_failed_jobs()` 函数**：检查 checkpoint 实际存在情况来恢复误判任务。

### `scripts/auto_run_all_pending.sh` 核心改动

- 移除 `set -e`
- `get_next_job()` 同时处理 `pending` 和 `interrupted`
- 新增 `wait_gpu_free()` 函数（每任务启动前等 GPU worker 全部退出）
- 新增连续失败计数（超过 5 次才停止）
