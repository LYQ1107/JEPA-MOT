#!/bin/bash
# Sequential 3-Target Domain Adaptation: clear -> dawn_dusk -> night -> partly_cloudy
# Fixed order, DARTH baseline, no CGCT/graph/curriculum learning

set -e

#=============================================================================
# Configuration
#=============================================================================

SOURCE_DOMAIN="clear"
TARGET_ORDER=("dawn_dusk" "night" "partly_cloudy")
EVAL_DOMAINS=("dawn_dusk" "night" "partly_cloudy")

ADAPT_EPOCHS=20
GPU_IDS=${GPU_IDS:-"9,8,7,6,5,4,3,2,1,0"}
GPU_NUM=${GPU_NUM:-10}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
TEST_WORKERS_PER_GPU=${TEST_WORKERS_PER_GPU:-8}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-14}

FORCE=${FORCE:-0}
RESET_TEACHER_FROM_STUDENT=${RESET_TEACHER_FROM_STUDENT:-1}
RUN_SOURCE_ONLY_EVAL=${RUN_SOURCE_ONLY_EVAL:-1}
DRY_RUN=${DRY_RUN:-0}

# Auto-detect source checkpoint
SOURCE_CKPT=${SOURCE_CKPT:-""}
if [ -z "$SOURCE_CKPT" ]; then
    CANDIDATES=(
        "baseline_darth_20260624/work_dirs/bdd_7domain/source/clear/qdtrack_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/20260612_173506/epoch_20.pth"
        "work_dirs/bdd_7domain/source/clear/qdtrack_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/20260612_173506/epoch_20.pth"
        "baseline_darth_20260624/work_dirs/bdd_7domain/source/clear/**/epoch_20.pth"
        "work_dirs/bdd_7domain/source/clear/**/epoch_20.pth"
    )
    for candidate in "${CANDIDATES[@]}"; do
        found=$(find . -path "./$candidate" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            SOURCE_CKPT="$found"
            break
        fi
    done
fi

ROOT_WORK_DIR="work_dirs/bdd_seq3/clear_to_dawn_dusk_night_partly_cloudy"

#=============================================================================
# Environment Check
#=============================================================================

echo "========================================="
echo "BDD Sequential 3-Target Domain Adaptation"
echo "========================================="
echo "Source: $SOURCE_DOMAIN"
echo "Targets (in order): ${TARGET_ORDER[@]}"
echo "Eval domains: ${EVAL_DOMAINS[@]}"
echo "Epochs per stage: $ADAPT_EPOCHS"
echo "GPU IDs: $GPU_IDS"
echo "GPUs: $GPU_NUM"
echo "SAMPLES_PER_GPU: $SAMPLES_PER_GPU"
echo "WORKERS_PER_GPU: $WORKERS_PER_GPU (train)"
echo "TEST_WORKERS_PER_GPU: $TEST_WORKERS_PER_GPU (eval)"
echo ""

# Check project root
if [ ! -f "tools/run/train.py" ] || [ ! -d "configs/adapters/darth" ]; then
    echo "ERROR: Must run from DARTH project root"
    exit 1
fi

# Check conda env
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda env: $CONDA_DEFAULT_ENV"
    if [ "$CONDA_DEFAULT_ENV" != "DARTH" ]; then
        echo "WARNING: Expected conda env 'DARTH', got '$CONDA_DEFAULT_ENV'"
    fi
else
    echo "WARNING: No conda environment detected"
fi

# Check source checkpoint
if [ ! -f "$SOURCE_CKPT" ]; then
    echo "ERROR: Source checkpoint not found: $SOURCE_CKPT"
    echo "Please set SOURCE_CKPT=<path> or ensure clear source checkpoint exists"
    exit 1
fi
echo "Source checkpoint: $SOURCE_CKPT"
echo ""

# Check configs
for target in "${TARGET_ORDER[@]}"; do
    adapt_cfg="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_${target}.py"
    eval_cfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${target}.py"
    
    if [ ! -f "$adapt_cfg" ]; then
        echo "ERROR: Adapt config not found: $adapt_cfg"
        exit 1
    fi
    if [ ! -f "$eval_cfg" ]; then
        echo "ERROR: Eval config not found: $eval_cfg"
        exit 1
    fi
done
echo "All configs verified ✓"
echo ""

#=============================================================================
# Helper Functions
#=============================================================================

get_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

log_cmd() {
    echo "[CMD] $@"
    if [ $DRY_RUN -eq 0 ]; then
        eval "$@"
    fi
}

check_checkpoint() {
    local ckpt=$1
    if [ -f "$ckpt" ]; then
        local size=$(stat -f%z "$ckpt" 2>/dev/null || stat -c%s "$ckpt" 2>/dev/null)
        if [ $size -gt $((300*1024*1024)) ]; then
            return 0
        fi
    fi
    return 1
}

#=============================================================================
# Stage 0: Source-only Evaluation
#=============================================================================

if [ $RUN_SOURCE_ONLY_EVAL -eq 1 ]; then
    echo "========================================="
    echo "Stage 0: Source-only Evaluation"
    echo "========================================="
    
    for eval_domain in "${EVAL_DOMAINS[@]}"; do
        eval_dir="${ROOT_WORK_DIR}/eval/stage00_source_only/${eval_domain}"
        eval_log="${eval_dir}/eval.log"
        
        if [ -f "${eval_dir}/scalabel_results.json" ] && [ $FORCE -eq 0 ]; then
            echo "SKIP: ${eval_dir} (already evaluated)"
            continue
        fi
        
        mkdir -p "${eval_dir}"
        eval_cfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${eval_domain}.py"
        PORT=$(get_free_port)
        
        echo "Evaluating source on ${eval_domain}..."
        cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=127.0.0.1 \
            --master_port=${PORT} \
            tools/run/test.py \
            ${eval_cfg} \
            --checkpoint ${SOURCE_CKPT} \
            --launcher pytorch \
            --work-dir ${eval_dir} \
            --eval bbox track \
            --fuse-conv-bn \
            --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
            > ${eval_log} 2>&1"
        
        log_cmd "$cmd"
        echo "  ✓ Saved to ${eval_dir}"
    done
    echo ""
fi

#=============================================================================
# Sequential Adaptation Stages
#=============================================================================

current_ckpt="${SOURCE_CKPT}"

for stage_idx in {1..3}; do
    target="${TARGET_ORDER[$((stage_idx-1))]}"
    stage_name=$(printf "stage%02d_%s" $stage_idx $target)
    adapt_dir="${ROOT_WORK_DIR}/adapt/${stage_name}"
    adapt_log="${adapt_dir}/train.log"
    final_ckpt="${adapt_dir}/epoch_${ADAPT_EPOCHS}.pth"
    
    echo "========================================="
    echo "Stage ${stage_idx}: Adapt to ${target}"
    echo "========================================="
    echo "Input checkpoint: ${current_ckpt}"
    echo "Output dir: ${adapt_dir}"
    
    # Check if already trained
    if check_checkpoint "$final_ckpt" && [ $FORCE -eq 0 ]; then
        echo "SKIP: Training already complete"
        current_ckpt="$final_ckpt"
    else
        # Prepare checkpoint
        load_ckpt="$current_ckpt"
        if [ $RESET_TEACHER_FROM_STUDENT -eq 1 ] && [ $stage_idx -gt 1 ]; then
            echo "RESET_TEACHER_FROM_STUDENT=1: Using checkpoint as-is"
            echo "  (Teacher will be reset by EMA hook initialization)"
        fi
        
        # Training
        mkdir -p "${adapt_dir}"
        adapt_cfg="configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_${target}.py"
        PORT=$(get_free_port)
        
        echo "Training on ${target} for ${ADAPT_EPOCHS} epochs..."
        cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=127.0.0.1 \
            --master_port=${PORT} \
            tools/run/train.py \
            ${adapt_cfg} \
            --launcher pytorch \
            --work-dir ${adapt_dir} \
            --cfg-options \
                total_epochs=${ADAPT_EPOCHS} \
                runner.max_epochs=${ADAPT_EPOCHS} \
                checkpoint_config.interval=1 \
                checkpoint_config.max_keep_ckpts=-1 \
                evaluation.interval=$((ADAPT_EPOCHS+1)) \
                find_unused_parameters=True \
                load_from=${load_ckpt} \
                data.samples_per_gpu=${SAMPLES_PER_GPU} \
                data.workers_per_gpu=${WORKERS_PER_GPU} \
            > ${adapt_log} 2>&1"
        
        log_cmd "$cmd"
        
        if check_checkpoint "$final_ckpt"; then
            echo "  ✓ Training complete"
            current_ckpt="$final_ckpt"
        else
            echo "  ✗ ERROR: Training failed or checkpoint missing"
            echo "  Check log: ${adapt_log}"
            exit 1
        fi
    fi
    
    # Evaluation after this stage
    echo ""
    echo "Evaluating after ${stage_name}..."
    for eval_domain in "${EVAL_DOMAINS[@]}"; do
        eval_dir="${ROOT_WORK_DIR}/eval/${stage_name}/${eval_domain}"
        eval_log="${eval_dir}/eval.log"
        
        if [ -f "${eval_dir}/scalabel_results.json" ] && [ $FORCE -eq 0 ]; then
            echo "  SKIP: ${eval_domain} (already evaluated)"
            continue
        fi
        
        mkdir -p "${eval_dir}"
        eval_cfg="configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_${eval_domain}.py"
        PORT=$(get_free_port)
        
        cmd="CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m torch.distributed.launch \
            --nproc_per_node=${GPU_NUM} \
            --master_addr=127.0.0.1 \
            --master_port=${PORT} \
            tools/run/test.py \
            ${eval_cfg} \
            --checkpoint ${current_ckpt} \
            --launcher pytorch \
            --work-dir ${eval_dir} \
            --eval bbox track \
            --fuse-conv-bn \
            --cfg-options data.workers_per_gpu=${TEST_WORKERS_PER_GPU} cudnn_benchmark=True \
            > ${eval_log} 2>&1"
        
        log_cmd "$cmd"
        echo "  ✓ ${eval_domain}"
    done
    echo ""
done

#=============================================================================
# Summary
#=============================================================================

echo "========================================="
echo "Sequential Adaptation Complete!"
echo "========================================="
echo "Results saved to: ${ROOT_WORK_DIR}"
echo ""
echo "Next steps:"
echo "1. Collect results:"
echo "   python tools/collect_bdd_seq3_results.py ${ROOT_WORK_DIR}"
echo ""
echo "2. View summaries:"
echo "   cat ${ROOT_WORK_DIR}/summaries/seq3_results_summary.md"
echo ""

if [ $DRY_RUN -eq 1 ]; then
    echo "========================================="
    echo "DRY RUN MODE - No commands were executed"
    echo "========================================="
fi
