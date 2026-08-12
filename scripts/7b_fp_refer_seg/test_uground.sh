#!/bin/bash

DEFAULT_GPUS="1,2"
GPU_LIST="${1:-${DEFAULT_GPUS}}"

if [ -z "${GPU_LIST}" ]; then
    echo "❌ Please specify GPUs."
    echo "Usage:"
    echo "  ./scripts/7b_fp_refer_seg/test_uground copy.sh 1,2"
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
WORLD_SIZE="${#GPU_ARRAY[@]}"

echo "========================================"
echo "Using GPUs      : ${GPU_ARRAY[@]}"
echo "World size      : ${WORLD_SIZE}"
echo "========================================"

function run_inference() {
    MODEL_KEY="${1}"
    CUDA_DEVICE="${2}"
    PROCESS_NUM="${3}"
    WORLD_SIZE="${4}"
    DATASET="${5}"
    INFERENCE_CMD="${6:-inference}"

    echo "[GPU ${CUDA_DEVICE}] process_num=${PROCESS_NUM}, cmd=${INFERENCE_CMD}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    python test_ds.py \
        --model_key="${MODEL_KEY}" \
        --cmd="${INFERENCE_CMD}" \
        --local_rank=0 \
        --process_num="${PROCESS_NUM}" \
        --world_size="${WORLD_SIZE}" \
        --dataset_dir="../dataset_sesame" \
        --version="./runs/UGround-7b_fp_refer_seg_llava1.5_ema/hf-UGround-7b_fp_refer_seg_llava1.5_ema" \
        --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
        --separate_mm_projector \
        --pad_train_clip_images \
        --preprocessor_config="./configs/preprocessor_336.json" \
        --resize_vision_tower \
        --resize_vision_tower_size=336 \
        --vision_tower_for_mask \
        --model_max_length=2048 \
        --val_dataset="${DATASET}" \
        --vis_save_path="./inference_results/${DATASET}_inference_cvpr" \
        --num_layers=33 \
        --strategy="policy_walker" \
        --mode=0 \
        --eval_legacy
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dataset_selector.sh"

MODEL_KEY="UGround"
VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

IFS=$'\n' read -rd '' -a SELECTED_DATASETS <<< "${VAL_DATASET//||/$'\n'}"

for dataset in "${SELECTED_DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "🚀 Running inference for dataset: ${dataset}"
    echo "========================================"

    # ---- parallel inference ----
    for idx in "${!GPU_ARRAY[@]}"; do
        GPU_ID="${GPU_ARRAY[$idx]}"

        run_inference \
            "${MODEL_KEY}" \
            "${GPU_ID}" \
            "${idx}" \
            "${WORLD_SIZE}" \
            "${dataset}" \
            "inference" &
    done

    echo "⏳ Waiting for all inference processes..."
    wait

    echo "✅ Inference finished for ${dataset}"

    # ---- metrics: rank 0 only ----
    echo "📊 Running metrics (rank 0 only)..."

    run_inference \
        "${MODEL_KEY}" \
        "${GPU_ARRAY[0]}" \
        0 \
        "${WORLD_SIZE}" \
        "${dataset}" \
        "metrics"

    echo "🎉 Inference + metrics done for ${dataset}"
done

echo "========================================"
echo "✅ ALL DATASETS FINISHED"
echo "========================================"


# #!/bin/bash

# function run_inference() {
#     MODEL_KEY="${1}"
#     CUDA_DEVICE="${2}"
#     PROCESS_NUM="${3}"
#     WORLD_SIZE="${4}"
#     DATASET="${5}"
#     INFERENCE_CMD="${6:-inference}"
#     CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python test_ds.py \
#         --model_key="${MODEL_KEY}" \
#         --cmd="${INFERENCE_CMD}" \
#         --local_rank=0 \
#         --process_num="${PROCESS_NUM}" \
#         --world_size="${WORLD_SIZE}" \
#         --dataset_dir="../dataset_sesame" \
#         --version="./runs/UGround-7b_fp_refer_seg_llava1.5_ema/hf-UGround-7b_fp_refer_seg_llava1.5_ema" \
#         --vision_tower="../dataset_sesame/clip-vit-large-patch14-336" \
#         --separate_mm_projector \
#         --pad_train_clip_images \
#         --preprocessor_config='./configs/preprocessor_336.json' \
#         --resize_vision_tower \
#         --resize_vision_tower_size=336 \
#         --vision_tower_for_mask \
#         --model_max_length=2048 \
#         --val_dataset="${DATASET}" \
#         --vis_save_path="./inference_results/${DATASET}_inference_cvpr" \
#         --num_layers=33 \
#         --strategy="policy_walker" \
#         --mode=0 \
#         --eval_legacy 
# }

# # Get script directory and source the dataset selector
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# source "$SCRIPT_DIR/dataset_selector.sh"

# # Interactive dataset selection
# MODEL_KEY="UGround"
# VAL_DATASET=$(select_datasets_interactive "$MODEL_KEY")

# # Split the selected datasets
# IFS=$'\n' read -rd '' -a SELECTED_DATASETS <<< "${VAL_DATASET//||/$'\n'}"
# # declare -a datasets=("fprefcoco|val" "fprefcoco+|val" "fprefcocog|val" "refcoco|val" "refcoco+|val" "refcocog|val")
# # for dataset in "${datasets[@]}"; do
# for dataset in "${SELECTED_DATASETS[@]}"; do
#     echo "Running inference for ${dataset}..."
#     run_inference "${MODEL_KEY}" 1 0 1 "${dataset}" "inference"
#     echo "Waiting for background inference processes to finish... for ${dataset}..."
#     wait
#     echo "Background processes for ${dataset} finished. Running metrics..."
#     run_inference "${MODEL_KEY}" 1 0 1 "${dataset}" "metrics" 
#     echo "Inference and metrics for ${dataset} finished."
# done
