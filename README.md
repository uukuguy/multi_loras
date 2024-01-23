# Multi-LoRAs

> Load multiple LoRA modules simultaneously and automatically switch the appropriate combination of LoRA modules to generate the best answer based on user queries.

![Multi-LoRAs](imgs/multi-loras.png)

Multi-LoRAs is a LLM toolkit that can simultaneously load multiple LoRA modules and automatically switch to the appropriate combination of LoRA modules based on user queries to generate the best answer. It includes tools such as extracting LoRA modules from efficiently parameters fine-tuning models, merging base models with LoRA models, and routing multiple LoRA models.

Tools:

- Extract the LoRA module from a model that has undergone efficient parameter fine-tuning.
- Tool for merging LoRA module into the base model.
- Multi LoRAs router (A gradient-free learning implement inspired by [lorahub](https://github.com/sail-sg/lorahub))

## Experiments

### Mixture-of-Multi-LoRAs

2024.01.23

[speechless-zephyr-code-functionary-7b](https://huggingface.co/uukuguy/speechless-zephyr-code-functionary-7b)

Extract LoRA modules from below models (all based Mistral-7B-v0.1), each LoRA module has its own unique skills. By using multi-loras, they can be combined together statically or dynamically to form a versatile new model.

- HuggingFaceH4/zephyr-7b-beta (Uncensored Model)
- meetkai/functionary-small-v2.2 (Execute functions/plugins)
- uukuguy/speechless-code-mistral-7b-v1.0 (Enhance Coding)

2024.01.05

[speechless-mistral-moloras-7b](https://huggingface.co/uukuguy/speechless-mistral-moloras-7b)

The goal of the work is to combine multiple professional models based on the same base model in order to achieve comparable fine-tuning effects on unknown tasks without training from scratch. This will be achieved by using Multi-LoRAs routers to automatically combine dedicated models.

Implemented a Multi-LoRAs routing inspired by lorahub's gradient-free learning.

Continuing from previous experiments, extract LoRA modules from 6 DARE models (base model Mistral-7B-v0.1). The optimal weight ratio of the LoRA modules was calculated using a gradient-free algorithm, and then fused to generate the final model.

#### Base model Mistral-7B-v0.1 and 6 LoRA modules

Download all LoRA modules from [speechless-multi-loras-r64](https://huggingface.co/uukuguy/speechless-multi-loras-r64).

```python
lora_module_list = [
    f"{multi_loras_dir}/Intel/neural-chat-7b-v3-1-lora",
    f"{multi_loras_dir}/migtissera/SynthIA-7B-v1.3-lora",
    f"{multi_loras_dir}/jondurbin/airoboros-m-7b-3.1.2-lora",
    f"{multi_loras_dir}/bhenrym14/mistral-7b-platypus-fp16-lora",
    f"{multi_loras_dir}/teknium/CollectiveCognition-v1.1-Mistral-7B-lora",
    f"{multi_loras_dir}/uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b-lora",
]
```

#### Dataset

Use the popular Alpaca dataset as learning samples. For the purpose of quick validation, only 64 random learning samples were selected.

It can be considered that these selected data with relatively small quantities are typical samples of a task. Our goal is to learn and optimize a LoRA for the task based on the existing 6 "professional" LoRA modules, without fine-tuning the task model from scratch.

```python
    dataset_file = "/opt/local/datasets/alpaca_data_cleaned.json"
    dataset = load_dataset("json", data_files=dataset_file)['train']
    print(f"Loaded {len(dataset)} samples")

    if max_learning_samples:
        dataset = dataset.select(range(max_learning_samples))
        print(f"Selected {len(dataset)} samples")
```

#### Gradient-free Learning

For 6 LoRA modules, running a total of 100 steps, obtain the optimized lora_ratios. With this allocation ratio, it is possible to merge the LoRA modules to generate the final static task model or implement dynamic task routing and allocation on the server side.

```bash
python multi_loras/lorahub.py \
    --max_inference_step 100 \
    --max_learning_samples 64
```

#### LM-Evaluation-Harness

| Metric | Value |
| --- | --- |
| ARC | 59.98  |
| HellaSwag | 83.29 |
| MMLU | 64.12 |
| TruthfulQA | 42.15 |
| Winogrande | 78.37 |
| GSM8K | 37.68 |
| Average | 60.93 |

TODO:

- How to determine if route learning has reached optimization. max_inference_step and ?
- Implementation of dynamic routing
- Selection of task sample data

2023.12.04

It seems that there are some issues with the calculation of the GSM8K and DROP metrics on the Open LLM Leaderboard. Currently, the DROP metric has been removed from the official website, while the calculation of GSM8K metric remains chaotic, with significant differences in values among various models. Therefore, I am temporarily using ARC, HellaSwag, MMLU, TruthfulQA, and Winogrande metrics to evaluate the performance of DARE.

| Model                                         | Average| ARC    | HellaSwag | MMLU| TruthfulQA | Winogrande |
| ------                                        | ------ | ------ | ------ | ------ | ------ | ------ |
| zephyr-7b-alpha                               | 68.590 | 61.01  | 84.04  | 61.39  | 57.90  | 78.61  |
| zephyr-7b-alpha-dare-0.85                     | 66.402 | 61.18  | 83.67  | 64.30  | 44.41  | 78.45  |
| CollectiveCognition-v1.1-Mistral-7B           | 68.326 | 62.12  | 84.17  | 62.35  | 57.62  | 75.37  |
| CollectiveCognition-v1.1-Mistral-7B-dare-0.85 | 66.676 | 61.01  | 84.31  | 64.34  | 44.87  | 78.85  |
| airoboros-m-7b-3.1.2                          | 67.722 | 61.86  | 83.51  | 61.91  | 53.75  | 77.58  |
| airoboros-m-7b-3.1.2-dare-0.85                | 66.144 | 61.09  | 83.57  | 64.05  | 43.64  | 78.37  |
| SynthIA-7B-v1.3                               | 67.688 | 62.12  | 83.45  | 62.65  | 51.37  | 78.85  |
| SynthIA-7B-v1.3-dare-0.85                     | 66.340 | 61.01  | 83.50  | 64.49  | 43.77  | 78.93  |
| neural-chat-7b-v3-1                           | 70.002 | 66.21  | 83.64  | 62.37  | 59.65  | 78.14  |
| neural-chat-7b-v3-1-dare-0.85                 | 66.856 | 61.95  | 83.84  | 64.43  | 44.90  | 79.16 |
|                                               |        |        |        |        |        |        |
| [speechless-mistral-7b-dare-0.85](https://huggingface.co/uukuguy/speechless-mistral-7b-dare-0.85) (merge 6 DARE models)| 68.516 | 63.57 | 84.82 | 64.29 | 50.66 | 79.24 |

From the official website evaluation results, after deleting 85% of the incremental parameters, the overall indicators remain above 97.5% of the original performance indicators. Among them, ARC slightly decreases, TruthfulQA significantly decreases, MMLU significantly increases, and HellaSwagt and Winogrande slightly increase. The most significant impact is the significant decrease in TruthfulQA, while other indicators are relatively well maintained, with MMLU showing a noticeable increase.

2023.11.26

DARE (Drop and REscale) was proposed in the paper [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](http://arxiv.org/abs/2311.03099). The insight is that most delta parameters can be directly set to zero without affecting the capabilities of SFT LMs. Based on this, we can use the DARE algorithm to sparsify the delta parameters of multiple parameter efficient fine-tuning models with different capabilities, and further obtain a more powerful new model through model merging algorithm, which preserves the advantages of each sub-model.

By drop the redundant delta parameters, it's possible to mitigate the mutual interference between merging models. What I want to do is try to verify this point. If the verification is successful, then I may have the possibility to merge multiple homologous models and maintain the prominent advantages of each model. And all of this does not require retraining the model, which is the most appealing aspect to me.

The following experiment will select multiple models with strong overall performance and outstanding sub-indicators on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). Each model will be built into its own DARE model, and the existing extract-lora function of multi-loras will be used to extract the LoRA module of each DARE model. It is hoped to ultimately build a new powerful model composed of multiple professional LoRA modules. We will name this Mixture-of-Multi-LoRAs.

Source LLM：mistral/Mistral-7B-v0.1

DARE: weight_mask_rate=0.85 / use_weight_rescale=True / mask_stratery=random / scaling_coefficient=1.0

PEFT Models:

- Intel/neural-chat-7b-v3-1 [DARE Model](https://huggingface.co/uukuguy/neural-chat-7b-v3-1-dare-0.85)
- bhenrym14/mistral-7b-platypus-fp16 [DARE Model](https://huggingface.co/uukuguy/mistral-7b-platypus-fp16-dare-0.9)
- jondurbin/airoboros-m-7b-3.1.2 [DARE Model](https://huggingface.co/uukuguy/airoboros-m-7b-3.1.2-dare-0.85)
- migtissera/SynthIA-7B-v1.3 [DARE Model](https://huggingface.co/uukuguy/SynthIA-7B-v1.3-dare-0.85)
- uukuguy/speechless-code-mistral-orca-7b-v1.0
- teknium/CollectiveCognition-v1.1-Mistral-7B [DARE Model](teknium/CollectiveCognition-v1.1-Mistral-7B)
- ehartford/dolphin-2.2.1-mistral-7b
- uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b [DARE Model](https://huggingface.co/uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b-dare-0.85)
- HuggingFaceH4/zephyr-7b-alpha [DARE Model](https://huggingface.co/uukuguy/zephyr-7b-alpha-dare-0.85)

LoRA Modules:

Use `mixture-of-multi-loras/extract_dare_loras.sh` script to extract LoRA module from all DARE models.

extract parameters: lora_r=64/bits=4/bf16

[huggingface.co/uukuguy/speechless-multi-loras-r64](https://huggingface.co/uukuguy/speechless-multi-loras-r64)

### Mistral-7B-OpenOrca

- Extract lora model [Mistral-7B-OpenOrca-lora](https://huggingface.co/uukuguy/Mistral-7B-OpenOrca-lora) from [Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca);

- Merge the base model [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) with lora model to [Mistral-7B-OpenOrca-lora-merged](https://huggingface.co/uukuguy/Mistral-7B-OpenOrca-lora-merged)

- LLM Evaluation ...

#### Local Test

| | ARC_acc_norm (25-shot) | HellaSwag_acc_norm (10-shot) | MMLU_acc (5-shot) | TruthfulQA_mc2 (0-shot) | GSM8K_acc (8-shot) | Open LLM Score |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Mistral-7B-OpenOrca | **71** | 83 | 61.42 | 45 | 40 | 65.11 |
| r=256 | 68 | 84 | 64.28 | 46.953 | **41** |  65.81 |
| **r=128** | 68 | **84** | **64.368** | 47.239 | **41** |  **65.90** |
| r=64 | 67 | 84 | 64.26 | **47.32** | **41** | 65.65 |
| *r=16* | *65* | *83* | *62.84* | *46.95* | *38* | *64.45* |

#### Open LLM Leaderboard

| | Average | ARC_acc_norm (25-shot) | HellaSwag_acc_norm (10-shot) | MMLU_acc (5-shot) | TruthfulQA_mc2 (0-shot) | Winogrande (5-shot) | GSM8K (5-shot) |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | 
| Mistral-7B-OpenOrca | 60.17 | 64.08 | 83.99 | 62.24 | 53.05 |  77.74 | 19.94 |
| [Mistral-7B-OpenOrca-lora](https://huggingface.co/uukuguy/Mistral-7B-OpenOrca-lora) | 58.14 | 61.95 | 83.62 | 64.16 | 42.74 | 79.08 | 17.29 |

## Install

```bash
pip install git+https://github.com/uukuguy/multi_loras.git
```

## Quick Start

Extract LoRA model from a model.

```bash
# --bits only support 4 or 8
python -m multi_loras \
    extract_lora \
    --base_model_name_or_path ${BASE_MODEL_PATH} \
    --tuned_model_name_or_path ${TUNED_MODEL_PATH} \
    --save_path ${LORA_SAVE_PATH} \
    --bf16 \
    --bits 4 \
    --lora_r 128
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
