# Baseline 备份方案（可执行版本）

生成时间: 2026-06-24 18:15
当前可用空间: 185GB

---

## 备份策略

**不压缩成tar包，而是保持目录结构直接拷贝**，这样你可以：
- 随时切换回baseline运行
- 做消融/对比实验时直接读取baseline checkpoint
- 查看详细的eval log而不需要解压

---

## 备份目录结构

```
baseline_darth_20260624/
├── code/                          # 核心代码文件
│   ├── mmtrack/
│   │   ├── models/mot/ema_track.py
│   │   └── models/losses/distillation_loss.py
│   ├── configs/adapters/darth/    # 所有DARTH配置
│   ├── tools/bdd_7domain_manager.py
│   └── tools/analysis/parse_bdd_7domain_results.py
│
├── checkpoints/                   # 关键epoch checkpoint (89GB)
│   ├── tta/                       # TTA epoch 6,12,20
│   │   ├── clear_to_dawn_dusk/
│   │   │   └── .../epoch_{6,12,20}.pth
│   │   └── ...                    # 42 pairs
│   └── source/                    # Source epoch 20,25,30
│       ├── clear/
│       │   └── .../epoch_{20,25,30}.pth
│       └── ...                    # 7 domains
│
├── results/                       # 评测结果 (35GB)
│   ├── summaries/                 # CSV/MD汇总
│   │   ├── complete_metrics_all_epochs.csv
│   │   ├── tta_best_epoch_per_pair.csv
│   │   ├── complete_metrics_summary.md
│   │   └── final_run_statistics.md
│   ├── tta_eval/                  # 420个TTA评测目录
│   │   ├── clear_to_dawn_dusk_epoch6/
│   │   ├── clear_to_dawn_dusk_epoch12/
│   │   └── ...
│   └── source_self_eval/          # 7个自测目录
│       ├── clear/
│       └── ...
│
└── README.md                      # 备份说明文档
```

**总大小: ~124GB**

---

## 执行步骤

### Step 1: 创建备份根目录

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
mkdir -p baseline_darth_20260624/{code,checkpoints,results}
```

### Step 2: 备份代码（约1分钟）

```bash
# 拷贝完整目录结构，保持可运行性
mkdir -p baseline_darth_20260624/code/mmtrack/models/{mot,losses}
mkdir -p baseline_darth_20260624/code/configs/adapters
mkdir -p baseline_darth_20260624/code/tools/analysis

cp mmtrack/models/mot/ema_track.py baseline_darth_20260624/code/mmtrack/models/mot/
cp mmtrack/models/losses/distillation_loss.py baseline_darth_20260624/code/mmtrack/models/losses/
cp -r configs/adapters/darth baseline_darth_20260624/code/configs/adapters/
cp tools/bdd_7domain_manager.py baseline_darth_20260624/code/tools/
cp tools/analysis/parse_bdd_7domain_results.py baseline_darth_20260624/code/tools/analysis/

echo "Code backup complete"
```

### Step 3: 备份结果（约5分钟）

```bash
# 直接拷贝整个目录，保持结构
cp -r work_dirs/bdd_7domain/summaries baseline_darth_20260624/results/
cp -r work_dirs/bdd_7domain/tta_eval baseline_darth_20260624/results/
cp -r work_dirs/bdd_7domain/source_self_eval baseline_darth_20260624/results/

echo "Results backup complete"
du -sh baseline_darth_20260624/results/*
```

### Step 4: 备份关键checkpoint（约15-20分钟，89GB）

```bash
# 使用Python脚本智能拷贝，保持目录结构
python3 << 'EOF'
import shutil, re
from pathlib import Path

base_src = Path('work_dirs/bdd_7domain')
base_dst = Path('baseline_darth_20260624/checkpoints')
base_dst.mkdir(parents=True, exist_ok=True)

count = 0
total_size = 0

# TTA epoch 6,12,20
print('Copying TTA checkpoints (epoch 6,12,20)...')
for pair_dir in (base_src / 'tta').iterdir():
    if not pair_dir.is_dir(): continue
    
    for ep in [6, 12, 20]:
        for ckpt in pair_dir.rglob(f'epoch_{ep}.pth'):
            if ckpt.stat().st_size < 300*1024**2: continue
            
            # 保持相对路径结构
            rel_path = ckpt.relative_to(base_src / 'tta')
            dst_path = base_dst / 'tta' / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(ckpt, dst_path)
            count += 1
            total_size += ckpt.stat().st_size
            
            if count % 10 == 0:
                print(f'  Copied {count} files, {total_size/1024**3:.1f} GB')

# Source epoch 20,25,30
print('Copying Source checkpoints (epoch 20,25,30)...')
for domain_dir in (base_src / 'source').iterdir():
    if not domain_dir.is_dir(): continue
    
    for ep in [20, 25, 30]:
        for ckpt in domain_dir.rglob(f'epoch_{ep}.pth'):
            if ckpt.stat().st_size < 300*1024**2: continue
            
            # 保持相对路径结构
            rel_path = ckpt.relative_to(base_src / 'source')
            dst_path = base_dst / 'source' / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(ckpt, dst_path)
            count += 1
            total_size += ckpt.stat().st_size
            
            if count % 5 == 0:
                print(f'  Copied {count} files, {total_size/1024**3:.1f} GB')

print(f'\nCheckpoint backup complete!')
print(f'Total: {count} files, {total_size/1024**3:.1f} GB')
EOF

echo "All backups complete"
```

### Step 5: 生成备份说明文档

```bash
cat > baseline_darth_20260624/README.md << 'DOC'
# DARTH Baseline Backup

备份时间: 2026-06-24
方法版本: DARTH with EMA + Distillation

## 包含内容

1. **代码** (code/)
   - EMA tracker: mmtrack/models/mot/ema_track.py
   - 蒸馏loss: mmtrack/models/losses/distillation_loss.py
   - DARTH配置: configs/adapters/darth/
   - 任务管理: tools/bdd_7domain_manager.py

2. **Checkpoint** (checkpoints/)
   - TTA: 42 pairs × 3 epochs (6,12,20) = 126 files
   - Source: 7 domains × 3 epochs (20,25,30) = 21 files
   - 总计: 147 checkpoint files, ~89GB

3. **结果** (results/)
   - 汇总表: summaries/*.csv, *.md
   - TTA评测: tta_eval/ (420个目录)
   - 自测: source_self_eval/ (7个目录)

## 性能汇总

### TTA Average MOTA (42 pairs)
- epoch 6:  15.47
- epoch 12: 15.79
- epoch 20: 15.69

### Source Self-eval Average MOTA (7 domains)
- epoch 20: 18.41

## 如何使用

### 1. 查看完整结果
```bash
cat results/summaries/complete_metrics_all_epochs.csv
```

### 2. 复现某个checkpoint的评测
```bash
# 使用备份的checkpoint
python tools/run/test.py \
  configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py \
  --checkpoint baseline_darth_20260624/checkpoints/tta/clear_to_night/*/epoch_12.pth \
  --eval bbox track
```

### 3. 对比新方法与baseline
```bash
# 新方法checkpoint
NEW_CKPT="work_dirs/new_method/clear_to_night/epoch_12.pth"
# Baseline checkpoint
BASE_CKPT="baseline_darth_20260624/checkpoints/tta/clear_to_night/*/epoch_12.pth"

# 分别测试对比
python tools/run/test.py CONFIG --checkpoint $NEW_CKPT --eval bbox track
python tools/run/test.py CONFIG --checkpoint $BASE_CKPT --eval bbox track
```

### 4. 恢复baseline代码到主工作区
```bash
cp -r baseline_darth_20260624/code/* /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/
```

DOC

cat baseline_darth_20260624/README.md
```

### Step 6: 验证备份完整性

```bash
# 检查备份大小
du -sh baseline_darth_20260624/*

# 检查checkpoint数量
echo "TTA checkpoints:"
find baseline_darth_20260624/checkpoints/tta -name "*.pth" | wc -l
echo "Should be: 126 (42 pairs × 3 epochs)"

echo "Source checkpoints:"
find baseline_darth_20260624/checkpoints/source -name "*.pth" | wc -l
echo "Should be: 21 (7 domains × 3 epochs)"

# 检查结果文件
echo "Result directories:"
ls -lh baseline_darth_20260624/results/
```

---

## 优点

1. **可直接运行** - checkpoint保持原有目录结构，配置文件中的路径无需修改
2. **易于对比** - 新实验可以直接与这个baseline对比
3. **便于查看** - 所有log和CSV都可以直接打开，不需要解压
4. **支持增量** - 后续可以继续往这个备份里添加其他epoch

---

## 空间占用

- 代码: < 0.1 GB
- 结果: ~35 GB
- Checkpoint: ~89 GB
- **总计: ~124 GB**

当前可用: 185GB，备份后剩余约 61GB

---

## 清理建议

备份完成后，如果需要更多空间做新实验，可以删除：

1. **TTA epoch 1-5** (释放 133GB)
   ```bash
   find work_dirs/bdd_7domain/tta -name "epoch_[1-5].pth" -delete
   ```

2. **Source epoch 1-10, 11-19, 21-24, 26-29** (释放 94GB)
   ```bash
   find work_dirs/bdd_7domain/source -name "epoch_[1-9].pth" -delete
   find work_dirs/bdd_7domain/source -name "epoch_1[0-9].pth" -delete
   find work_dirs/bdd_7domain/source -name "epoch_2[1-4].pth" -delete
   find work_dirs/bdd_7domain/source -name "epoch_2[6-9].pth" -delete
   ```

这样可以释放约 227GB，加上当前剩余的 185GB，总共约 412GB 可用空间。
