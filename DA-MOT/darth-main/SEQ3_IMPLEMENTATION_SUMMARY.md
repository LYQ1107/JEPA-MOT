# Sequential 3-Target Domain Adaptation - Implementation Summary

## ✅ Completed Components

### 1. Main Pipeline Script
**File**: `scripts/run_bdd_clear_seq3_darth.sh`

**Features**:
- ✓ Fixed-order sequential adaptation: clear → dawn_dusk → night → partly_cloudy
- ✓ 20 epochs per adaptation stage
- ✓ Stage 0 source-only evaluation
- ✓ After each stage: evaluate on all 3 target domains
- ✓ Automatic checkpoint detection and passing between stages
- ✓ Dry-run mode for validation
- ✓ Resume support (skip completed stages)
- ✓ All checkpoints saved (interval=1, max_keep=-1)
- ✓ Teacher reset option (RESET_TEACHER_FROM_STUDENT)
- ✓ Uses `load_from` (not `resume_from`) to avoid epoch conflicts
- ✓ Mid-training eval disabled to save time

**Configuration**: Supports all environment variables (GPU_NUM, SAMPLES_PER_GPU, FORCE, etc.)

### 2. Results Collection Script
**File**: `tools/collect_bdd_seq3_results.py`

**Features**:
- ✓ Parses all evaluation logs
- ✓ Extracts HOTA, MOTA, IDF1, AssA, DetA metrics
- ✓ Generates long-format CSV
- ✓ Generates metric matrices (one per metric)
- ✓ Generates markdown summary report
- ✓ Handles missing results gracefully

### 3. Usage Documentation
**File**: `SEQ3_USAGE.md`

**Content**:
- ✓ Quick start guide
- ✓ Configuration options
- ✓ Output structure
- ✓ Progress checking
- ✓ Resume instructions
- ✓ Troubleshooting
- ✓ Runtime estimates

---

## 📝 Usage Summary

### Dry-Run (Recommended First Step)

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
conda activate DARTH
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh
```

**What it does**:
- Checks all configs exist
- Finds clear source checkpoint
- Prints all commands without executing
- Validates experiment setup

### Actual Run

```bash
# Small test (2 GPUs)
GPU_NUM=2 bash scripts/run_bdd_clear_seq3_darth.sh

# Production (8 GPUs)
GPU_NUM=8 SAMPLES_PER_GPU=2 bash scripts/run_bdd_clear_seq3_darth.sh
```

### Collect Results

```bash
python tools/collect_bdd_seq3_results.py \
    work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy

# View summary
cat work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/summaries/seq3_results_summary.md
```

---

## 📊 Expected Outputs

### Training Checkpoints
```
work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/adapt/
├── stage01_dawn_dusk/epoch_{1..20}.pth
├── stage02_night/epoch_{1..20}.pth
└── stage03_partly_cloudy/epoch_{1..20}.pth
```

### Evaluation Results
```
work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/eval/
├── stage00_source_only/{dawn_dusk,night,partly_cloudy}/
├── stage01_after_dawn_dusk/{dawn_dusk,night,partly_cloudy}/
├── stage02_after_night/{dawn_dusk,night,partly_cloudy}/
└── stage03_after_partly_cloudy/{dawn_dusk,night,partly_cloudy}/
```

### Summary Files
```
work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy/summaries/
├── seq3_results_long.csv          # All metrics
├── seq3_results_summary.md        # Markdown report
├── seq3_matrix_hota.csv           # 4×3 HOTA matrix
├── seq3_matrix_mota.csv           # 4×3 MOTA matrix
├── seq3_matrix_idf1.csv           # 4×3 IDF1 matrix
├── seq3_matrix_assa.csv           # 4×3 AssA matrix
└── seq3_matrix_deta.csv           # 4×3 DetA matrix
```

---

## ⏱️ Runtime Estimates

**2 GPUs** (SAMPLES_PER_GPU=1):
- Stage 1 (dawn_dusk): ~2-3 hours
- Stage 2 (night): ~4-5 hours
- Stage 3 (partly_cloudy): ~2-3 hours
- All evaluations: ~30 min
- **Total: ~10-12 hours**

**8 GPUs** (SAMPLES_PER_GPU=2):
- **Total: ~3-4 hours**

---

## 🔍 Key Implementation Details

### Stage Transition Strategy
1. **Stage 1**: Load clear source checkpoint → adapt on dawn_dusk
2. **Stage 2**: Load stage1/epoch_20.pth → adapt on night
3. **Stage 3**: Load stage2/epoch_20.pth → adapt on partly_cloudy

### Critical Design Choices

✅ **Use `load_from` instead of `resume_from`**
- Reason: Avoids epoch counter conflicts
- Each stage starts from epoch 1 and trains to epoch 20

✅ **Reset teacher at each new stage** (default)
- Reason: Prevents previous target's teacher EMA from dominating
- Teacher initialized from current student, then updated by EMA hook

✅ **Disable mid-training evaluation**
- Reason: Saves time, only evaluate after each stage completes
- Set `evaluation.interval=21` (> max_epochs=20)

✅ **Save all checkpoints**
- Reason: Allows fine-grained analysis of adaptation trajectory
- Set `checkpoint_config.max_keep_ckpts=-1`

---

## 🚀 Next Steps

### Immediate (Validation)
1. Run dry-run to verify setup
2. Run with 2 GPUs to test full pipeline
3. Check results collection works
4. Verify metrics are reasonable

### Short-term (Experiments)
1. Compare different target orders
2. Ablate teacher reset strategy
3. Try different epoch counts per stage
4. Analyze forgetting on previous targets

### Long-term (Extensions)
1. Implement CGCT graph co-teaching
2. Add automatic curriculum (closest-domain selection)
3. Add pseudo-source replay
4. Add causal memory module

---

## 📋 Files Modified/Created

### Created
- `scripts/run_bdd_clear_seq3_darth.sh` (293 lines)
- `tools/collect_bdd_seq3_results.py` (211 lines)
- `SEQ3_USAGE.md` (264 lines)
- `SEQ3_IMPLEMENTATION_SUMMARY.md` (this file)

### Modified
- None (all new files, no changes to DARTH core)

---

## ✅ Verification Checklist

Before running experiments:

- [ ] Dry-run passes without errors
- [ ] Clear source checkpoint found and valid
- [ ] All 3 target adapt configs exist
- [ ] All 3 target eval configs exist
- [ ] Conda environment is DARTH
- [ ] At least 185GB disk space available
- [ ] GPUs are available and not in use

After first stage completes:

- [ ] stage01_dawn_dusk/epoch_20.pth exists and is >300MB
- [ ] 3 evaluation results exist in stage01_after_dawn_dusk/
- [ ] MOTA scores are reasonable (not NaN or 0)

After experiment completes:

- [ ] All 3 stage final checkpoints exist
- [ ] 12 evaluation result directories exist (4 stages × 3 domains)
- [ ] Results collection script runs successfully
- [ ] Summary markdown displays valid metrics

---

## 🐛 Known Limitations

1. **Fixed order only** - This version doesn't implement curriculum/order selection
2. **No CGCT** - Graph co-teaching not implemented yet
3. **Single source** - Only supports clear as source
4. **No intermediate epoch eval** - Only evaluates after full stage completion
5. **Manual teacher reset** - Teacher initialization not fully automated (uses EMA hook default)

These are intentional simplifications for the baseline version.

---

## 📖 Additional Documentation

- **Quick start**: `SEQ3_USAGE.md`
- **Troubleshooting**: See `SEQ3_USAGE.md` troubleshooting section
- **DARTH original docs**: Check `configs/adapters/darth/` for config examples

---

## 🎯 Success Criteria

This implementation is successful if:

1. ✅ Script runs without crashes
2. ✅ Checkpoints are saved at each stage
3. ✅ Sequential loading works (stage2 uses stage1 output)
4. ✅ Evaluation runs on all 3 targets after each stage
5. ✅ Results can be collected and summarized
6. ✅ Metrics show expected adaptation behavior (MOTA improves on adapted domain)

Expected behavior:
- MOTA on dawn_dusk should improve after stage 1
- MOTA on night should improve after stage 2
- MOTA on partly_cloudy should improve after stage 3
- Possible catastrophic forgetting on earlier domains

---

## 💡 Tips

**For debugging**:
```bash
# Check what commands will run
DRY_RUN=1 bash scripts/run_bdd_clear_seq3_darth.sh | grep "CMD"

# Monitor training
tail -f work_dirs/bdd_seq3/.../adapt/stage01_dawn_dusk/train.log

# Check checkpoints
ls -lh work_dirs/bdd_seq3/.../adapt/*/epoch_20.pth
```

**For analysis**:
```bash
# Extract MOTA from all evals
grep "OVERALL" work_dirs/bdd_seq3/.../eval/*/*/eval.log

# Compare stages
python tools/collect_bdd_seq3_results.py work_dirs/bdd_seq3/...
```

---

**Implementation complete! Ready for testing.**
