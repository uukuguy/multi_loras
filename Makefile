MODELS_ROOT_DIR=/opt/local/llm_models/huggingface.co

BASE_MODEL_PATH=${MODELS_ROOT_DIR}/mistralai/Mistral-7B-v0.1
# LORA_R=64
# # ---------- Open-Orca/Mistral-7B-SlimOrca ----------
# TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/Open-Orca/Mistral-7B-SlimOrca

# # ---------- Open-Orca/Mistral-7B-OpenOrca ----------
# TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/Open-Orca/Mistral-7B-OpenOrca

# # ---------- uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b ----------
# TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/speechlessai/speechless-mistral-dolphin-orca-platypus-samantha-7b


# LORA_SAVE_PATH=${MODELS_ROOT_DIR}/$(shell basename ${TUNED_MODEL_PATH})-lora
# TASK_MODEL_PATH=${MODELS_ROOT_DIR}/$(shell basename ${TUNED_MODEL_PATH})-lora-merged

# ---------- Open-Orca/Mistral-7B-OpenOrca ----------
# TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/Open-Orca/Mistral-7B-OpenOrca

# ---------- HuggingFaceH4/zephyr-7b-beta ----------
# TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/HuggingFaceH4/zephyr-7b-beta 

# ---------- meetkai/functionary-small-v2.2 ----------
# TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/meetkai/functionary-small-v2.2

# # ---------- uukuguy/speechless-code-mistral-7b-v1.0 ----------
TUNED_MODEL_PATH=${MODELS_ROOT_DIR}/uukuguy/speechless-code-mistral-7b-v1.0

BITS=4
LORA_R=64
LORA_SAVE_PATH=${MODELS_ROOT_DIR}/$(shell basename ${TUNED_MODEL_PATH})-${BITS}bit-r${LORA_R}-lora
TASK_MODEL_PATH=${MODELS_ROOT_DIR}/$(shell basename ${TUNED_MODEL_PATH})-${BITS}bit-r${LORA_R}-lora-merged

help:
	@echo "Usage: make [extract_lora | merge_lora]" 

extract_lora:
	PYTHONPATH=${PWD} \
	python -m multi_loras \
		extract_lora \
		--base_model_name_or_path ${BASE_MODEL_PATH} \
		--tuned_model_name_or_path ${TUNED_MODEL_PATH} \
		--save_path ${LORA_SAVE_PATH} \
		--bf16 \
		--bits ${BITS} \
		--lora_r ${LORA_R}

merge_lora:
	PYTHONPATH=${PWD} \
	python -m multi_loras \
		merge_lora \
		--base_model_name_or_path ${BASE_MODEL_PATH} \
		--lora_model_path ${LORA_SAVE_PATH} \
		--merged_model_name_or_path ${TASK_MODEL_PATH}

slora_server:
	python multi_loras/slora/slora_server.py \
		--port 8000 \
		--model_dir ${MODELS_ROOT_DIR}/meta-llama/Llama-2-7b-hf \

include ../Makefile.remote
