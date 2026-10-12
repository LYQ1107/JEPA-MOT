# Sequential 3-Target Domain Adaptation - Status Report

**Date**: 2026-06-25  
**Status**: ✅ **Phase 1 Complete - Ready for Testing**

---

## ✅ What Has Been Completed

### 1. Core Implementation

| Component | Status | File | Lines |
|---|:---:|---|---:|
| Main pipeline script | ✅ | `scripts/run_bdd_clear_seq3_darth.sh` | 293 |
| Results collector | ✅ | `tools/collect_bdd_seq3_results.py` | 211 |
| Usage guide | ✅ | `SEQ3_USAGE.md` | 264 |
| Implementation summary | ✅ | `SEQ3_IMPLEMENTATION_SUMMARY.md` | 287 |

**Total**: 4 new files, 1055 lines, **no modifications to DARTH core**

### 2. Features Implemented

✅ **Sequential Adaptation Pipeline**
- Fixed order: clear → dawn_dusk → night → partly_cloudy
- 20 epochs per stage
- Automatic checkpoint passing between stages
- Uses `load_from` (not `resume_from`) to avoid epoch conflicts

✅ **Comprehensive Evaluation**
- Stage 0: Source-only evaluation on all 3 targets
- After each stage: Evaluate on all 3 targets
- Total: 4 stages × 3 domains = 12 evaluations

✅ **Production Features**
- Dry-run mode for validation
- Resume support (skip completed stages)
- Configurable GPUs, batch size, workers
- All checkpoints saved (max_keep=-1)
- Teacher reset option at each stage

✅ **Results Collection**
- Parse all evaluation logs
- Generate CSV tables (long format + matrices)
- Generate markdown summary
- Extract HOTA, MOTA, IDF1, AssA, DetA

### 3. Verification Results

```
✓ All files created and executable
✓ Source checkpoint found automatically
✓ All config files verified
✓ Dry-run generates correct commands
✓ Error in dry-run is expected (no training was run)
```

---

## 🎯 Next Steps (Phase 2)

### Immediate Testing

```bash
# 1. Verify dry-run (already done ✓)
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh

# 2. Test with 2 GPUs (recommended)
GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh

# 3. Collect results after completion
python tools/collect_bdd_seq3_results.py \
    work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy
```

### Expected Timeline

**With 2 GPUs**:
- Stage 1 (dawn_dusk): 2-3 hours
- Stage 2 (night): 4-5 hours  
- Stage 3 (partly_cloudy): 2-3 hours
- Evaluations: 30 minutes
- **Total: ~10-12 hours**

**With 8 GPUs**:
- **Total: ~3-4 hours**

---

## 📊 What Will Be Generated

### After First Stage

```
work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/
├── eval/stage00_source_only/{dawn_dusk,night,partly_cloudy}/
├── eval/stage01_after_dawn_dusk/{dawn_dusk,night,partly_cloudy}/
└── adapt/stage01_dawn_dusk/epoch_{1..20}.pth
```

### After All Stages

```
work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/
├── adapt/
│   ├── stage01_dawn_dusk/epoch_{1..20}.pth
│   ├── stage02_night/epoch_{1..20}.pth
│   └── stage03_partly_cloudy/epoch_{1..20}.pth
├── eval/
│   └── [12 evaluation directories: 4 stages × 3 domains]
└── summaries/
    ├── seq3_results_long.csv
    ├── seq3_results_summary.md
    └── seq3_matrix_{hota,mota,idf1,assa,deta}.csv
```

---

## 🔧 Configuration Quick Reference

### Basic Run

```bash
GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

### Production Run

```bash
GPU_NUM=8 SAMPLES_PER_GPU=2 WORKERS_PER_GPU=4 \
    bash scripts/run_bdd_clear_seq3_darth.sh
```

### Force Re-run

```bash
FORCE=1 GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

### Custom Source Checkpoint

```bash
SOURCE_CKPT=/path/to/clear_epoch20.pth \
    GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

---

## 📝 Key Design Decisions

### 1. Stage Transition Strategy

**Each stage**:
- Loads previous stage's `epoch_20.pth`
- Uses `load_from` (NOT `resume_from`)
- Starts from epoch 1 (not epoch 21)
- Trains for exactly 20 epochs

**Why**: Avoids optimizer state and epoch counter conflicts

### 2. Teacher Initialization

**Default**: `RESET_TEACHER_FROM_STUDENT=1`
- Teacher = Student at start of each new stage
- Then updated by EMA hook during training

**Why**: Prevents previous domain's teacher EMA from dominating new domain

### 3. Evaluation Strategy

**No mid-training eval**: `evaluation.interval=21` (> max_epochs)
**Post-stage eval**: Test on all 3 targets after each stage

**Why**: Saves time, focuses on final stage performance

### 4. Checkpoint Management

**Save all**: `checkpoint_config.max_keep_ckpts=-1`
**Save every epoch**: `checkpoint_config.interval=1`

**Why**: Enables fine-grained analysis of adaptation trajectory

---

## ✅ Verification Checklist

Before running:
- [x] Scripts created and executable
- [x] Source checkpoint auto-detected
- [x] All configs verified
- [x] Dry-run successful
- [ ] Conda env activated: `conda activate DARTH`
- [ ] Sufficient disk space: 185GB available
- [ ] GPUs available

After stage 1:
- [ ] `stage01_dawn_dusk/epoch_20.pth` exists (>300MB)
- [ ] 3 evaluation results in `stage01_after_dawn_dusk/`
- [ ] MOTA scores reasonable (not NaN/0)

After completion:
- [ ] 3 final checkpoints exist
- [ ] 12 evaluation directories exist
- [ ] Results collection successful
- [ ] Summary markdown has valid metrics

---

## 🚨 Known Limitations (Intentional)

This baseline version does NOT implement:
1. ❌ CGCT graph co-teaching
2. ❌ Automatic curriculum/order selection
3. ❌ Pseudo-source replay
4. ❌ Causal memory module
5. ❌ Multi-source simultaneous adaptation

These are intentionally excluded to establish a clean baseline.

---

## 📖 Documentation Files

| File | Purpose |
|---|---|
| `SEQ3_USAGE.md` | Quick start guide, troubleshooting |
| `SEQ3_IMPLEMENTATION_SUMMARY.md` | Technical details, design choices |
| `STATUS_REPORT.md` | This file - current status |

---

## 🎓 What This Achieves

This implementation provides a **reproducible baseline** for:

1. **Fixed-order sequential adaptation** experiments
2. **Catastrophic forgetting** analysis (compare stage N performance on domain from stage N-1)
3. **Forward transfer** analysis (does adapting on A help B?)
4. **Order sensitivity** studies (try different sequences)

It establishes the foundation for:
- CGCT implementation
- Curriculum learning
- Order optimization
- Multi-target DA methods

---

## 🚀 Ready to Run

The implementation is **complete and ready for testing**.

**Recommended first run**:

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
conda activate DARTH

# Test run with 2 GPUs
GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

**Monitor progress**:

```bash
# Watch training
tail -f work_dirs/bdd_seq3/.../adapt/stage01_dawn_dusk/train.log

# Check completed stages
ls -lh work_dirs/bdd_seq3/.../adapt/*/epoch_20.pth
```

**Collect results** (after completion):

```bash
python tools/collect_bdd_seq3_results.py \
    work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy

cat work_dirs/bdd_seq3/.../summaries/seq3_results_summary.md
```

---

**Implementation Status**: ✅ **READY FOR TESTING**

**Phase 1 Complete!** 🎉
