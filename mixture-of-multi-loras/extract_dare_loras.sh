#!/bin/bash
SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co
BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
MIXTURE_ROOT_DIR=${MODELS_ROOT_DIR}/mixture-of-multi-loras

DARE_MODELS="bhenrym14/mistral-7b-platypus-fp16 Intel/neural-chat-7b-v3-1 jondurbin/airoboros-m-7b-3.1.2 migtissera/SynthIA-7B-v1.3 teknium/CollectiveCognition-v1.1-Mistral-7B uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b HuggingFaceH4/zephyr-7b-alpha"

LORA_R=64
LORA_MODELS_ROOT_DIR=${MIXTURE_ROOT_DIR}/loras-r${LORA_R}
mkdir -p ${LORA_MODELS_ROOT_DIR}

for DARE_MODEL in ${DARE_MODELS}; do
    DARE_MODEL_PATH=${MIXTURE_ROOT_DIR}/${DARE_MODEL}-dare-0.85
    LORA_SAVE_PATH=${LORA_MODELS_ROOT_DIR}/${DARE_MODEL}-lora
    echo "Extracting LORA from ${DARE_MODEL_PATH} to ${LORA_SAVE_PATH}"
    PYTHONPATH=${SCRIPT_PATH}/.. \
    python -m multi_loras \
        extract_lora \
        --base_model_name_or_path ${BASE_MODEL_PATH} \
        --tuned_model_name_or_path ${DARE_MODEL_PATH} \
        --save_path ${LORA_SAVE_PATH} \
        --bf16 \
        --bits 4 \
        --lora_r ${LORA_R}
done
