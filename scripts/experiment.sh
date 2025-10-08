#!/bin/bash

# Meta-Learning Experimental Comparison Script
# Usage: ./scripts/experiment.sh [options]

# Default parameters
NUM_ROUNDS=10
NUM_EPOCHS=300
NUM_TASKS=10
TARGET_EPOCH=25
NOISE_SCALE=0.05
SEED=2025
OUTPUT_PREFIX="meta_learning_comparison"
VERBOSE=""

# Algorithm hyperparameters
INNER_LR=8e-5
OUTER_LR_HF=5e-3
OUTER_LR_ZO=5e-2
PERTURBATION_SCALE=1.1e-3
REPTILE_WEIGHT=0.5

# Environment parameters
MAX_STEPS=100
NUM_TEST_TASKS=8
TEST_PERTURBATION_SCALE=1e-4
EVAL_INTERVAL=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --num_tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --target_epoch)
            TARGET_EPOCH="$2"
            shift 2
            ;;
        --noise_scale)
            NOISE_SCALE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --inner_lr)
            INNER_LR="$2"
            shift 2
            ;;
        --outer_lr_hf)
            OUTER_LR_HF="$2"
            shift 2
            ;;
        --outer_lr_zo)
            OUTER_LR_ZO="$2"
            shift 2
            ;;
        --perturbation_scale)
            PERTURBATION_SCALE="$2"
            shift 2
            ;;
        --reptile_weight)
            REPTILE_WEIGHT="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --num_test_tasks)
            NUM_TEST_TASKS="$2"
            shift 2
            ;;
        --test_perturbation_scale)
            TEST_PERTURBATION_SCALE="$2"
            shift 2
            ;;
        --eval_interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --quick)
            # Quick test configuration
            NUM_ROUNDS=3
            NUM_EPOCHS=10
            TARGET_EPOCH=5
            shift
            ;;
        --full)
            # Full experiment configuration
            NUM_ROUNDS=10
            NUM_EPOCHS=300
            TARGET_EPOCH=50
            shift
            ;;
        --help|-h)
            echo "Meta-Learning Experimental Comparison"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  Training Parameters:"
            echo "    --num_rounds N        Number of training rounds per algorithm (default: 10)"
            echo "    --num_epochs N        Number of training epochs per round (default: 300)"
            echo "    --num_tasks N         Number of tasks per training round (default: 10)"
            echo "    --target_epoch N      Epoch to use for few-shot evaluation (default: 50)"
            echo "    --eval_interval N     Evaluation interval during training (default: 1)"
            echo ""
            echo "  Environment Parameters:"
            echo "    --noise_scale F       Environment noise scale (default: 0.05)"
            echo "    --max_steps N         Maximum steps per episode (default: 100)"
            echo "    --num_test_tasks N    Number of test tasks for meta-testing (default: 8)"
            echo "    --test_perturbation_scale F  Perturbation scale for test environments (default: 1e-4)"
            echo ""
            echo "  Algorithm Hyperparameters:"
            echo "    --inner_lr F          Inner loop learning rate (default: 8e-5)"
            echo "    --outer_lr_hf F       Outer loop learning rate for HessianFree (default: 5e-3)"
            echo "    --outer_lr_zo F       Outer loop learning rate for ZOMAML (default: 5e-2)"
            echo "    --perturbation_scale F Perturbation scale for gradients (default: 1.1e-3)"
            echo "    --reptile_weight F    Weight for Reptile in combined method (default: 0.5)"
            echo ""
            echo "  Output Parameters:"
            echo "    --seed N              Random seed (default: 2025)"
            echo "    --output_prefix STR   Output filename prefix (default: meta_learning_comparison)"
            echo "    --verbose             Enable verbose output"
            echo ""
            echo "  Presets:"
            echo "    --quick               Quick test: 3 rounds, 10 epochs, target epoch 5"
            echo "    --full                Full experiment: 10 rounds, 300 epochs, target epoch 50"
            echo "    --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                    # Quick test run"
            echo "  $0 --full --verbose           # Full experiment with verbose output"
            echo "  $0 --num_rounds 5 --num_epochs 100  # Custom configuration"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "Meta-Learning Experimental Comparison"
echo "========================================"
echo "Configuration:"
echo "  Training:"
echo "    Rounds per algorithm: $NUM_ROUNDS"
echo "    Epochs per round: $NUM_EPOCHS"
echo "    Tasks per round: $NUM_TASKS"
echo "    Target epoch for few-shot: $TARGET_EPOCH"
echo "    Evaluation interval: $EVAL_INTERVAL"
echo "  Environment:"
echo "    Noise scale: $NOISE_SCALE"
echo "    Max steps: $MAX_STEPS"
echo "    Test tasks: $NUM_TEST_TASKS"
echo "    Test perturbation scale: $TEST_PERTURBATION_SCALE"
echo "  Algorithm Hyperparameters:"
echo "    Inner LR: $INNER_LR"
echo "    Outer LR (HessianFree): $OUTER_LR_HF"
echo "    Outer LR (ZOMAML): $OUTER_LR_ZO"
echo "    Perturbation scale: $PERTURBATION_SCALE"
echo "    Reptile weight: $REPTILE_WEIGHT"
echo "  Output:"
echo "    Random seed: $SEED"
echo "    Output prefix: $OUTPUT_PREFIX"
echo "    Verbose: $([ -n "$VERBOSE" ] && echo "Yes" || echo "No")"
echo "========================================"

# Run the experimental comparison
python experimental_comparison.py \
    --num_rounds $NUM_ROUNDS \
    --num_epochs $NUM_EPOCHS \
    --num_tasks $NUM_TASKS \
    --target_epoch $TARGET_EPOCH \
    --eval_interval $EVAL_INTERVAL \
    --noise_scale $NOISE_SCALE \
    --max_steps $MAX_STEPS \
    --num_test_tasks $NUM_TEST_TASKS \
    --test_perturbation_scale $TEST_PERTURBATION_SCALE \
    --inner_lr $INNER_LR \
    --outer_lr_hf $OUTER_LR_HF \
    --outer_lr_zo $OUTER_LR_ZO \
    --perturbation_scale $PERTURBATION_SCALE \
    --reptile_weight $REPTILE_WEIGHT \
    --seed $SEED \
    --output_prefix $OUTPUT_PREFIX \
    $VERBOSE

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Experiment completed successfully!"
    echo "Output plot: ${OUTPUT_PREFIX}_multifold.png"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Experiment failed with exit code $?"
    echo "========================================"
    exit 1
fi
