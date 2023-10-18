# Multi-LoRAs

Multi-LoRAs is a LLM toolkit that can simultaneously load multiple LoRA modules and automatically switch to the appropriate combination of LoRA modules based on user queries to generate the best answer. It includes tools such as extracting LoRA modules from efficiently parameters fine-tuning models, merging base models with LoRA models, and routing multiple LoRA models.

Tools:

- Extract the LoRA module from a model that has undergone efficient parameter fine-tuning.
- Tool for merging LoRA module into the base model.
- Multi LoRAs router (Under development)

## Experiments

### Mistral-7B-OpenOrca

- Extract lora model [Mistral-7B-OpenOrca-lora](https://huggingface.co/uukuguy/Mistral-7B-OpenOrca-lora) from [Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca);

- Merge the base model [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) with lora model to [Mistral-7B-OpenOrca-lora-merged](https://huggingface.co/uukuguy/Mistral-7B-OpenOrca-lora-merged)

- LLM Evaluation ...

#### Local Test

| | ARC_acc_norm (25-shot) | HellaSwag_acc_norm (10-shot) | MMLU_acc (5-shot) | TruthfulQA_mc2 (0-shot) | GSM8K_acc (8-shot) | Open LLM Score |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Mistral-7B-OpenOrca | **71** | 83 | 61.42 | 45 | 40 | 65.11 |
| r=256 | 68 | **84** |  |  |  |  |
| **r=64** | 67 | 84 | **64.26** | **47.32** | **41** | **65.65** |
| *r=16* | *65* | *83* | *62.84* | *46.95* | *38* | *64.45* |

#### Open LLM Leaderboard
| | ARC_acc_norm (25-shot) | HellaSwag_acc_norm (10-shot) | MMLU_acc (5-shot) | TruthfulQA_mc2 (0-shot) | Open LLM Score |
| ------ | ------ | ------ | ------ | ------ | ------ |
| Mistral-7B-SlimOrca | 62.54 | 83.86 | **62.77** | **54.23** |  **65.85** |
| Mistral-7B-OpenOrca | **64.08** | **83.99** | 62.24 | 53.05 |  65.84 |


## Install

```bash
pip install multi-loras
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

[bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
> networks/extract_lora_from_models.py
> networks/merge_lora.py
> networks/resize_lora.py
> network/lora.py
> network/lora_fa.py
> network/dylora.py

- LoRA for Text-to-Image

[cloneofsimo/lora](https://github.com/cloneofsimo/lora)
> lora_diffusion/cli_svd.py

- Microsoft LoRA
[LoRA](https://github.com/microsoft/LoRA)
