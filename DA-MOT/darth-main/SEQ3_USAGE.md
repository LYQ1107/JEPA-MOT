# BDD Sequential 3-Target Domain Adaptation - Usage Guide

## Overview

This implements a **fixed-order sequential adaptation** pipeline:
- **Source**: clear
- **Targets (in order)**: dawn_dusk → night → partly_cloudy
- **Method**: DARTH baseline (no CGCT/curriculum)
- **Epochs per stage**: 20

Each stage adapts on one target domain, then evaluates on all three targets.

---

## Files Created

### 1. Main Script
- `scripts/run_bdd_clear_seq3_darth.sh` - Sequential adaptation pipeline

### 2. Results Collection
- `tools/collect_bdd_seq3_results.py` - Parse logs and generate summaries

---

## Quick Start

### Step 1: Dry-run (check without executing)

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
conda activate DARTH
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh
```

This will:
- ✓ Check if configs exist
- ✓ Find clear source checkpoint
- ✓ Print all commands that would be executed
- ✗ NOT actually run training/evaluation

### Step 2: Run with 2 GPUs (recommended for testing)

```bash
GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

### Step 3: Run with more GPUs (production)

```bash
GPU_NUM=8 SAMPLES_PER_GPU=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

---

## Configuration Options

### Environment Variables

```bash
# Required
GPU_NUM=2                    # Number of GPUs to use

# Optional
SAMPLES_PER_GPU=1           # Batch size per GPU
WORKERS_PER_GPU=2           # Data loader workers
FORCE=0                     # 1=rerun even if complete
RESET_TEACHER_FROM_STUDENT=1 # 1=reset teacher at each stage
RUN_SOURCE_ONLY_EVAL=1      # 1=run stage0 evaluation
DRY_RUN=0                   # 1=print commands only

# Auto-detected (can override)
SOURCE_CKPT=<path>          # Path to clear source checkpoint
```

### Example: Force re-run stage 2

```bash
# This will skip completed stages but force re-run any failed ones
FORCE=1 GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

---

## Output Structure

```
work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/
├── adapt/
│   ├── stage01_dawn_dusk/
│   │   ├── epoch_1.pth ... epoch_20.pth
│   │   └── train.log
│   ├── stage02_night/
│   │   ├── epoch_1.pth ... epoch_20.pth
│   │   └── train.log
│   └── stage03_partly_cloudy/
│       ├── epoch_1.pth ... epoch_20.pth
│       └── train.log
├── eval/
│   ├── stage00_source_only/
│   │   ├── dawn_dusk/eval.log
│   │   ├── night/eval.log
│   │   └── partly_cloudy/eval.log
│   ├── stage01_after_dawn_dusk/
│   │   ├── dawn_dusk/eval.log
│   │   ├── night/eval.log
│   │   └── partly_cloudy/eval.log
│   ├── stage02_after_night/
│   │   └── ... (same structure)
│   └── stage03_after_partly_cloudy/
│       └── ... (same structure)
└── summaries/
    ├── seq3_results_long.csv
    ├── seq3_results_summary.md
    ├── seq3_matrix_hota.csv
    ├── seq3_matrix_mota.csv
    ├── seq3_matrix_idf1.csv
    ├── seq3_matrix_assa.csv
    └── seq3_matrix_deta.csv
```

---

## Checking Progress

### Check if stages are complete

```bash
ls -lh work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/adapt/*/epoch_20.pth
```

### Check current training log

```bash
tail -f work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/adapt/stage01_dawn_dusk/train.log
```

### Check evaluation status

```bash
find work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/eval -name "scalabel_results.json" | wc -l
# Should be 12 when complete (4 stages × 3 eval domains)
```

---

## Collecting Results

### After experiment completes

```bash
python tools/collect_bdd_seq3_results.py \
    work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy
```

This generates:
- `summaries/seq3_results_long.csv` - All metrics in long format
- `summaries/seq3_matrix_*.csv` - Metric matrices (rows=stages, cols=domains)
- `summaries/seq3_results_summary.md` - Markdown report

### View summary

```bash
cat work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/summaries/seq3_results_summary.md
```

---

## Resume from Interruption

The script supports **automatic resume**:
- If `epoch_20.pth` exists for a stage → skip training
- If evaluation results exist → skip evaluation
- To force re-run: `FORCE=1`

### Example: Resume after power failure

```bash
# Just re-run the same command
GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
# It will automatically skip completed stages
```

---

## Troubleshooting

### Problem: Source checkpoint not found

**Solution**: Set explicit path

```bash
SOURCE_CKPT=baseline_darth_20260624/work_dirs/.../epoch_20.pth \
    GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

### Problem: Training fails with CUDA OOM

**Solution**: Reduce batch size

```bash
GPU_NUM=2 SAMPLES_PER_GPU=1 bash scripts/run_bdd_clear_seq3_darth.sh
```

### Problem: Config not found

**Solution**: Check config exists

```bash
ls configs/adapters/darth/bdd_7domain/darth_qdtrack_*_{dawn_dusk,night,partly_cloudy}.py
ls configs/mot/qdtrack/bdd_7domain/qdtrack_*_{dawn_dusk,night,partly_cloudy}.py
```

### Problem: Evaluation results not parsed

**Solution**: Check log format

```bash
grep -A 5 "OVERALL" work_dirs/bdd_seq3/.../eval/stage01_*/dawn_dusk/eval.log
```

---

## Expected Runtime

With 2 GPUs (SAMPLES_PER_GPU=1):
- Stage 1 (dawn_dusk): ~2-3 hours
- Stage 2 (night): ~4-5 hours (larger dataset)
- Stage 3 (partly_cloudy): ~2-3 hours
- Evaluations: ~30 min total (12 evaluations × 2-3 min each)

**Total: ~10-12 hours**

With 8 GPUs (SAMPLES_PER_GPU=2):
- **Total: ~3-4 hours**

---

## Next Steps

After this baseline works, you can:
1. Modify order: Try different target sequences
2. Add CGCT: Implement graph co-teaching
3. Add curriculum: Implement closest-domain selection
4. Compare: Run multiple orders and compare results

---

## Notes

- This version uses `load_from` (not `resume_from`) to avoid epoch conflicts
- Teacher is reset at each new stage (RESET_TEACHER_FROM_STUDENT=1)
- Each stage trains for exactly 20 epochs from epoch 1
- All checkpoints are saved (max_keep_ckpts=-1)
- Mid-training evaluation is disabled (evaluation.interval=21)

---

## Contact

For issues with this implementation, check:
1. Dry-run output: `DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh`
2. Training logs: `work_dirs/bdd_seq3/.../adapt/*/train.log`
3. Eval logs: `work_dirs/bdd_seq3/.../eval/*/eval.log`
