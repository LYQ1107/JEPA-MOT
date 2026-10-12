# Baseline 备份和清理方案

生成时间: 2026-06-24 17:28

当前磁盘: **58GB 可用 (99% 已用)**

---

## 一、需要备份的内容

### 1. 代码文件 (约 50MB)

**核心方法实现:**
- `mmtrack/models/mot/ema_track.py` - EMA tracker
- `mmtrack/models/losses/distillation_loss.py` - 蒸馏loss
- `configs/adapters/darth/` - 所有DARTH配置
- `tools/bdd_7domain_manager.py` - 任务管理器

**备份位置:** `baseline_snapshot_20260624/code/`

### 2. 关键结果文件 (约 35GB)

**必须保留的结果:**
- `work_dirs/bdd_7domain/summaries/*.csv` - 所有汇总指标
- `work_dirs/bdd_7domain/summaries/*.md` - 统计报告
- `work_dirs/bdd_7domain/tta_eval/` - 420个TTA评测结果 (34GB)
- `work_dirs/bdd_7domain/source_self_eval/` - 7个自测结果 (1GB)

**备份位置:** `baseline_snapshot_20260624/results/`

### 3. 最终checkpoint (约 30GB)

**保留epoch:**
- Source: epoch20, epoch25, epoch30 (已训完的)
- TTA: epoch6, epoch12, epoch20

**备份位置:** `baseline_snapshot_20260624/checkpoints/`

**总备份量: ~65GB**

---

## 二、可以安全删除的内容

### 删除优先级1: TTA早期checkpoint (释放 133GB)

```bash
# epoch 1-5 对你的分析没用，全部删除
find work_dirs/bdd_7domain/tta -name "epoch_[1-5].pth" -delete
```

**收益: 133GB**

### 删除优先级2: TTA中间checkpoint (释放 107GB)

```bash
# epoch 7-11, 13-19 可删除，保留 6, 12, 20
find work_dirs/bdd_7domain/tta -name "epoch_[7-9].pth" -delete
find work_dirs/bdd_7domain/tta -name "epoch_1[01].pth" -delete
find work_dirs/bdd_7domain/tta -name "epoch_1[3-9].pth" -delete
```

**收益: 107GB**

### 删除优先级3: Source早期checkpoint (释放 64GB)

```bash
# epoch 1-10 可删除
find work_dirs/bdd_7domain/source -name "epoch_[1-9].pth" -delete
find work_dirs/bdd_7domain/source -name "epoch_10.pth" -delete
```

**收益: 64GB**

### 删除优先级4: Source中间checkpoint (释放 30GB)

```bash
# epoch 11-19, 21-24, 26-29 可删除，保留 20, 25, 30
find work_dirs/bdd_7domain/source -name "epoch_1[1-9].pth" -delete
find work_dirs/bdd_7domain/source -name "epoch_2[1-4].pth" -delete
find work_dirs/bdd_7domain/source -name "epoch_2[6-9].pth" -delete
```

**收益: 30GB**

### 删除优先级5: 训练日志 (释放 5-10GB)

```bash
# 训练日志很大，eval结果已经在单独目录
find work_dirs/bdd_7domain/tta -name "*.log" -size +50M -delete
find work_dirs/bdd_7domain/source -name "*.log" -size +50M -delete
```

**收益: 5-10GB**

### 删除优先级6: source_only_eval (可选，释放 4GB)

这些是 source→其他域 的直接测试结果，不做TTA。如果你后续不需要对比可以删。

```bash
rm -rf work_dirs/bdd_7domain/source_only_eval
```

**收益: 4GB**

---

## 三、执行步骤

### Step 1: 创建备份

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
mkdir -p baseline_snapshot_20260624/{code,results,checkpoints}

# 备份代码
tar -czf baseline_snapshot_20260624/code/darth_baseline_code.tar.gz \
  mmtrack/models/mot/ema_track.py \
  mmtrack/models/losses/distillation_loss.py \
  configs/adapters/darth/ \
  tools/bdd_7domain_manager.py \
  tools/analysis/parse_bdd_7domain_results.py

# 备份结果
cp -r work_dirs/bdd_7domain/summaries baseline_snapshot_20260624/results/
tar -czf baseline_snapshot_20260624/results/tta_eval.tar.gz work_dirs/bdd_7domain/tta_eval/
tar -czf baseline_snapshot_20260624/results/source_self_eval.tar.gz work_dirs/bdd_7domain/source_self_eval/

# 备份关键checkpoint (epoch 6,12,20 for TTA; 20,25,30 for source)
# 这一步会比较慢，约30GB数据
python3 << 'EOF'
import shutil, re
from pathlib import Path
base=Path('baseline_snapshot_20260624/checkpoints')
base.mkdir(exist_ok=True)
# TTA epoch 6,12,20
for p in Path('work_dirs/bdd_7domain/tta').rglob('epoch_*.pth'):
    m=re.search(r'epoch_([0-9]+)\.pth$', p.name)
    if m and int(m.group(1)) in [6,12,20] and p.stat().st_size>300*1024**2:
        rel=p.relative_to('work_dirs/bdd_7domain/tta')
        dst=base/'tta'/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p, dst)
        print('copied',rel)
# Source epoch 20,25,30
for p in Path('work_dirs/bdd_7domain/source').rglob('epoch_*.pth'):
    m=re.search(r'epoch_([0-9]+)\.pth$', p.name)
    if m and int(m.group(1)) in [20,25,30] and p.stat().st_size>300*1024**2:
        rel=p.relative_to('work_dirs/bdd_7domain/source')
        dst=base/'source'/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p, dst)
        print('copied',rel)
EOF
```

### Step 2: 验证备份完整性

```bash
du -sh baseline_snapshot_20260624/*
tar -tzf baseline_snapshot_20260624/code/darth_baseline_code.tar.gz | head -20
ls -lh baseline_snapshot_20260624/results/summaries/*.csv
```

### Step 3: 清理checkpoint

**建议从优先级1开始逐步删，删完一轮看磁盘空间再决定是否继续。**

```bash
# 优先级1: TTA epoch1-5 (释放 133GB)
find work_dirs/bdd_7domain/tta -name "epoch_[1-5].pth" -exec rm -v {} \;

# 检查释放空间
df -h .

# 如果还不够，继续删优先级2: TTA epoch7-11,13-19
find work_dirs/bdd_7domain/tta -name "epoch_[7-9].pth" -exec rm -v {} \;
find work_dirs/bdd_7domain/tta -name "epoch_1[01].pth" -exec rm -v {} \;
find work_dirs/bdd_7domain/tta -name "epoch_1[3-9].pth" -exec rm -v {} \;

df -h .
```

---

## 四、推荐方案

考虑到磁盘只剩58GB，建议：

1. **立即执行 Step 1 + 优先级1清理**
   - 备份会额外占用约65GB (但很快会通过删除释放133GB)
   - 净收益: 释放约 **68GB**，剩余空间 -> 126GB

2. **如果还需要更多空间，再执行优先级2**
   - 再释放 107GB -> 剩余空间 233GB

3. **优先级3-6根据实际需求决定**

---

## 五、恢复方法

如果将来需要恢复baseline运行:

```bash
cd baseline_snapshot_20260624
tar -xzf code/darth_baseline_code.tar.gz -C /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/
tar -xzf results/tta_eval.tar.gz -C /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/work_dirs/bdd_7domain/
# checkpoint手动拷回即可
```

---

## 六、你需要手动决定的删除项

| 内容 | 大小 | 是否保留? | 备注 |
|---|---:|---|---|
| TTA epoch 1-5 | 133GB | **建议删** | 对分析无用 |
| TTA epoch 7-11, 13-19 | 107GB | **建议删** | 已有6,12,20足够 |
| Source epoch 1-10 | 64GB | **建议删** | 已有20,25,30 |
| Source epoch 11-19, 21-24, 26-29 | 30GB | 可删 | 如果只关心最终结果 |
| source_only_eval 结果 | 4GB | 保留 | 是baseline对比 |
| 大日志文件 | 5-10GB | 可删 | 结果已在eval目录 |

**我的推荐: 至少删除优先级1+2，释放240GB，让你有足够空间做新实验。**
