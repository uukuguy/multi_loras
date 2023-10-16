# Multi-LoRAs

Multi-LoRAs is a LLM toolkit that can simultaneously load multiple LoRA modules and automatically switch to the appropriate combination of LoRA modules based on user queries to generate the best answer. It includes tools such as extracting LoRA modules from efficiently parameters fine-tuning models, merging base models with LoRA models, and routing multiple LoRA models.

Tools:

- Extract the LoRA module from a model that has undergone efficient parameter fine-tuning.
- Tool for merging LoRA module into the base model.
- Multi LoRAs router (Under development)

## Install

```bash
pip install multi_loras
# - or -
pip install git+https://github.com/uukuguy/multi_loras.git
```

## Quick Start

Extract LoRA model from a model.

```bash
python -m multi_loras \
    extract_lora \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --tuned_model_name_or_path ${TUNED_MODEL_PATH} \
    --save_path ${LORA_SAVE_PATH} \
    --bf16 \
    --bits 4 \
    --lora_r 64
```

Merge the extracted LoRA model with the base model.

```bash
python -m multi_loras \
    merge_lora \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --lora_model_path ${LORA_SAVE_PATH} \
    --merged_model_name_or_path ${TASK_MODEL_PATH}
```

## References

- Gradio GUI for Kohya’s Stable Diffusion Trainer

[](https://github.com/bmaltais/kohya_ss)
> networks/extract_lora_from_models.py
> networks/merge_lora.py
> networks/resize_lora.py
> network/lora.py
> network/lora_fa.py
> network/dylora.py

- LoRA for Text-to-Image

[](https://github.com/cloneofsimo/lora)
> lora_diffusion/cli_svd.py

- Microsoft LoRA
[](https://github.com/microsoft/LoRA)
