import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import LoraConfig
from functools import partial
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_get_loss(task_dataset, model, batch_size=None):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    data_batch_size = len(task_dataset) if batch_size is None else min(len(task_dataset), batch_size)
    # use gpu if available
    task_dataloader = DataLoader(
        task_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(task_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(task_dataset["input"])

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_final_weights(lora_ratios, cached_lora_modules):
    final_state_dict = {}
    for i, (_, lora_state_dict) in enumerate(cached_lora_modules.items()):
        keys = lora_state_dict.keys()
        if i == 0:
            for key in keys:
                final_state_dict[key] = lora_ratios[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + lora_ratios[i] * lora_state_dict[key]
                )
    return final_state_dict


def get_score(
    lora_ratios,
    peft_model,
    cached_lora_modules,
    task_dataset,
    batch_size,
    get_loss=default_get_loss,
    get_regular=default_l1_regularization
):
    # the composed lora state dict
    final_state_dict = get_final_weights(lora_ratios, cached_lora_modules)
    # reload the model with the new adapter config
    set_peft_model_state_dict(peft_model, final_state_dict)

    # minimize the metric
    loss = get_loss(task_dataset, peft_model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(lora_ratios)

    return metric_val


def learn(peft_model, cached_lora_modules, task_dataset, max_inference_step: int = 40, batch_size=None):

    number_of_loras = len(cached_lora_modules)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    get_score_partial = partial(get_score,
                                peft_model=peft_model,
                                cached_lora_modules=cached_lora_modules,
                                task_dataset=task_dataset,
                                batch_size=batch_size)
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )

    print("> Begin to perform gradient-free optimization ...")
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    lora_ratios=recommendation.value
    task_lora = get_final_weights(lora_ratios, cached_lora_modules)

    # # set the final weights
    # set_peft_model_state_dict(peft_model, task_lora)
    # task_model = peft_model.merge_and_unload()
    # return lora_ratios, task_model

    return task_lora

def preloaing_lora_modules(base_model, lora_module_list: List[str]):
    """load base model and lora modules from huggingface model hub

    Args:
        base_model: the base model
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
    """
    print("> Begin to load lora modules")
    cached_lora_modules = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list, ncols=100, desc="Loading LoRA modules"):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cached_lora_modules[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cached_lora_modules[peft_model_id]
        # check whether the LoRA can be merged into one
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cached_lora_modules[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')

    default_peft_model_id = lora_module_list[0]
    peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    return peft_model, cached_lora_modules



import copy
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import Dataset

def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    tokenized_dataset = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return tokenized_dataset


# def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None):
#     """load base model and lora modules from huggingface model hub

#     Args:
#         lora_module_list (List[str]): a list of lora module names available in huggingface model hub
#         model_name_or_path (Optional[str]): base model name, default is None
#     """
#     # use gpu if available
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # load basic model
#     default_peft_model_id = lora_module_list[0]
#     # find the base model
#     if model_name_or_path is None:
#         model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path

#     base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
#     # load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     # 0 is the default model
#     try:
#         peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
#     except:
#         raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')

#     peft_model = peft_model.to(device)
#     peft_model.eval()

#     print("> Begin to load lora modules")
#     cache = {}

#     first_dict = None

#     for peft_model_id in tqdm(lora_module_list):
#         print("> Loading {} ...".format(peft_model_id))
#         cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
#         cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

#         if first_dict is None:
#             first_dict = cache[peft_model_id]
#         # check whether the LoRA can be merged into one
#         try:
#             # detect whether the arch is the same
#             for key in first_dict.keys():
#                 assert first_dict[key].shape == cache[peft_model_id][key].shape
#         except:
#             raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')

#     return peft_model, tokenizer, cache

# def lorahub_inference(example_inputs: List[str],
#                       model_or_name_path: Union[AutoModelForSeq2SeqLM, str],
#                       tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
#                       batch_size: int,
#                       # if not provided, we do not report the accuracy
#                       example_outputs: List[str]=None):

#     def accuracy_score(outputs, ground_truths):
#         correct = 0
#         total = 0
#         for output, truth in zip(outputs, ground_truths):
#             if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
#                 correct += 1
#             total += 1
#         return correct / total * 100

#     example_predictions = []
#     # load model
#     if isinstance(model_or_name_path, str):
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_or_name_path)
#     else:
#         model = model_or_name_path

#     # load tokenizer
#     if isinstance(tokenizer_or_tokenizer_path, str):
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path)
#     else:
#         tokenizer = tokenizer_or_tokenizer_path

#     # process dataset
#     dataset = load_dataset(example_inputs, example_outputs, tokenizer)
#     # use gpu if available
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)

#     for i in range(0, len(dataset["input"]), batch_size):
#         inputs = tokenizer(
#             dataset["input"][i : i + batch_size],
#             max_length=2048,
#             return_tensors="pt",
#             padding=True,
#         ).to(device)
#         outputs = model.generate(
#             input_ids=inputs["input_ids"], max_new_tokens=256
#         )
#         outputs = tokenizer.batch_decode(
#             outputs.to("cpu"), skip_special_tokens=True
#         )
#         example_predictions.extend(outputs)

#     if example_outputs is not None:
#         task_perf = accuracy_score(example_predictions, example_outputs)
#     else:
#         task_perf = None

#     return example_predictions, task_perf

def do_learn(args):
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    lora_module_list = args.lora_module_list

    task_dataset = None

    peft_model, cached_lora_modules = preloaing_lora_modules(base_model, lora_module_list)

    lora_ratios, task_lora = learn(
        peft_model=peft_model,
        cached_lora_modules=cached_lora_modules,
        task_dataset=task_dataset,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size
    )

    print(f"{lora_module_list=}")
    print(f"{lora_ratios=}")

    print(f"Saving the task LoRA to {args.output_dir}")
    lora_config = LoraConfig.from_pretrained(lora_module_list[0])
    lora_config.inference_mode = True
    torch.save(task_lora, f"{args.output_dir}/adapter_model.bin")
    lora_config.save_pretrained(args.output_dir)
    print(f"Saved {args.output_dir}")

    # set_peft_model_state_dict(peft_model, task_lora)
    # task_model = peft_model.merge_and_unload()
    # print(f"Saving the model to {args.output_dir}")
    # task_model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"Saved the model to {args.output_dir}")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_learn", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--lora_module_list", nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--example_inputs", type=str, default=None)
    parser.add_argument("--example_outputs", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_inference_step", type=int, default=40)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.do_learn:
        do_learn(args)
    # # load model
    # model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path)
    # # process dataset
    # dataset = load_dataset(example_inputs, example_outputs, tokenizer)

if __name__ == "__main__":
    main()
