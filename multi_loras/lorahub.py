import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import default_data_collator
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import LoraConfig
from functools import partial
from typing import List, Dict, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, Dataset



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
        for batch in tqdm(task_dataloader, ncols=100, desc="Calculating loss"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    loss = float(loss) / len(task_dataloader)
    print(f"learning loss: {loss:.6f}")
    return loss

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_final_weights(lora_ratios, cached_lora_modules):
    print(f"{lora_ratios=}")
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


def learn(peft_model, cached_lora_modules, task_dataset, max_inference_step: int = 40, batch_size=None, num_workers: int=1):
    print(f"Start learning with {len(task_dataset)} examples")

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

    if num_workers > 1:
        from concurrent import futures
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step, num_workers=num_workers)
        # We use ThreadPoolExecutor for CircleCI but please
        # use the line just below, with ProcessPoolExecutor instead (unless your
        # code is I/O bound rather than CPU bound):
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        #with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
        # With batch_mode=True it will ask the optimizer for num_workers points to evaluate, run the evaluations, then update the optimizer with the num_workers function outputs, and repeat until the budget is all spent. Since no executor is provided, the evaluations will be sequential. num_workers > 1 with no executor is therefore suboptimal but nonetheless useful for evaluation purpose (i.e. we simulate parallelism but have no actual parallelism). batch_mode=False (steady state mode) will ask for a new evaluation whenever a worker is ready.
            recommendation = optimizer.minimize(get_score_partial, executor=executor, batch_mode=False)
    else:
        optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
        recommendation = optimizer.minimize(get_score_partial, verbosity=1)

    print(f"{type(recommendation)=}, {recommendation=}")
    lora_ratios=recommendation.value
    task_lora = get_final_weights(lora_ratios, cached_lora_modules)

    # # set the final weights
    # set_peft_model_state_dict(peft_model, task_lora)
    # task_model = peft_model.merge_and_unload()
    # return lora_ratios, task_model

    return lora_ratios, task_lora

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model = peft_model.to(device)
    peft_model.eval()

    return peft_model, cached_lora_modules



import copy
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import Dataset

def format_alpaca_data(instruction: str,  response: str, input: str=None, system_prompt: str=None):
    if system_prompt is None:
        if input:
            system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        else:
            system_prompt = "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n"

    if input:
        human_input = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ".format(instruction=instruction, input=input)
    else:
        human_input = "### Instruction:\n{instruction}\n\n### Response:".format(instruction=instruction)

    source = f"{system_prompt}{human_input}"
    target = f"{response.strip()}\n"

    # print(f"{source=},{target=}")

    return source, target

IGNORE_INDEX=-100
from torch.nn.utils.rnn import pad_sequence
def tokenize_and_pad_instruction_response_pairs(instructions: List, responses: List, inputs: List, model_max_len, tokenizer, prompt_type="alpaca"):
    print(f"Call tokenize_and_pad_instruction_response_pairs for {len(instructions)} instances, {len(responses)} responses, {len(inputs)} inputs")
    input_ids = []
    labels = []
    for idx, (instruction, response, input) in enumerate(tqdm(zip(instructions, responses, inputs), ncols=100, desc="Tokenizing")):
        if prompt_type == "alpaca":
            source, target = format_alpaca_data(instruction=instruction, response=response, input=input)
        else:
            source, target = format_alpaca_data(instruction=instruction, response=response, input=input)
        source = tokenizer.bos_token + source
        target = target + tokenizer.eos_token

        tokenized_source = tokenizer(source, max_length=model_max_len, truncation=True, add_special_tokens=False)
        tokenized_target = tokenizer(target, max_length=model_max_len, truncation=True, add_special_tokens=False)
        tokenized_input = torch.tensor(tokenized_source['input_ids'] + tokenized_target['input_ids'])
        tokenized_output = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source['input_ids']))] +
                                        copy.deepcopy(tokenized_target['input_ids']))

        example_input_ids = tokenized_input
        example_output_ids = tokenized_output
        # if idx == 0:
        #     example_input_ids = tokenized_input
        #     example_output_ids = tokenized_output
        # else:
        #     example_input_ids = torch.concatenate((example_input_ids, tokenized_input), dim=0)
        #     example_output_ids = torch.concatenate((example_output_ids, tokenized_output), dim=0)

        input_ids.append(example_input_ids)
        labels.append(example_output_ids)

    if tokenizer.padding_side == "left":
        input_ids = [t.flip(-1) for t in input_ids]
        labels = [t.flip(-1) for t in labels]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    if tokenizer.padding_side == "left":
        input_ids = input_ids.flip(-1)
        labels = labels.flip(-1)

    data_dict = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask':input_ids.ne(tokenizer.pad_token_id),
    }
    return data_dict


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: PreTrainedTokenizer
    model_max_len: int
    prompt_type: str = None
    instruction_field: str = "instruction"
    response_field: str = "response"
    input_field: str = "input"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        print(f"Call DataCollatorForCausalLM for {len(instances)} instances")
        instructions = [instance[self.instruction_field] for instance in instances]
        responses = [instance[self.response_field] for instance in instances]
        inputs = [instance[self.input_field] if self.input_field in instance else "" for instance in instances]

        return tokenize_and_pad_instruction_response_pairs(
            instructions=instructions,
            responses=responses,
            inputs=inputs,
            tokenizer=self.tokenizer,
            model_max_len=self.model_max_len,
            prompt_type=self.prompt_type
        )


def load_task_dataset(tokenizer, max_learning_samples=None, model_max_len=2048, prompt_type="alpaca"):
    dataset_file = "/opt/local/datasets/alpaca_data_cleaned.json"
    print(f"Loading dataset {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file)['train']
    print(f"Loaded {len(dataset)} samples")

    if max_learning_samples:
        dataset = dataset.select(range(max_learning_samples))
        print(f"Selected {len(dataset)} samples")

    instructions = dataset['instruction']
    responses = dataset['output']
    df = [
        {"instruction": instructions[i], "response": responses[i]}
        for i in range(len(dataset))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    # preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    # tokenized_dataset = dataset.map(
    #     preprocess_func_with_tokenizer,
    #     batched=True,
    #     num_proc=1,
    #     desc="Running tokenizer on dataset",
    # )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_len=model_max_len,
        prompt_type=prompt_type,
    )
    data_dict = data_collator(dataset)
    tokenized_dataset = Dataset.from_dict(data_dict)
    print(f"tokenized_dataset: {len(tokenized_dataset)} samples.")

    return tokenized_dataset

def fix_special_tokens(tokenizer, model_name_or_path):
    print(f"---------- Original tokens----------")
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

    if "qwen" in model_name_or_path.lower():
        tokenizer.eos_token = "<|endoftext|>"
        # tokenizer.unk_token = "<|extra_3|>"
        tokenizer.bos_token = "<|extra_2|>"
        tokenizer.pad_token = "<|extra_1|>"
    else:
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 1
            tokenizer.bos_token = "<s>"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 2
            tokenizer.eos_token = "</s>"
        # if tokenizer.unk_token_id is None:
        #     tokenizer.unk_token_id = 0
        #     tokenizer.unk_token = "<unk>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0 # tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id) #tokenizer.eos_token
    print(f"---------- Fixed tokens ----------")
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    # print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

def do_learn(args):
    # Tokenizer
    tokenizer_kwargs = {
        "padding_side": "left",
        "use_fast": False,
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    fix_special_tokens(tokenizer, args.model_name_or_path)

    lora_module_list = args.lora_module_list

    task_dataset = load_task_dataset(tokenizer=tokenizer, max_learning_samples=args.max_learning_samples)

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    peft_model, cached_lora_modules = preloaing_lora_modules(base_model, lora_module_list)

    lora_ratios, task_lora = learn(
        peft_model=peft_model,
        cached_lora_modules=cached_lora_modules,
        task_dataset=task_dataset,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"{lora_module_list=}")
    print(f"{lora_ratios=}")

    print(f"Saving the task LoRA to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
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

multi_loras_dir = "/opt/local/llm_models/huggingface.co/mixture-of-multi-loras/speechless-multi-loras-r64"
# type(recommendation)=<class 'nevergrad.parametrization.data.Array'>, recommendation=Array{(6,),Cl([-1.5 -1.5 -1.5 -1.5 -1.5 -1.5],[1.5 1.5 1.5 1.5 1.5 1.5],b)}[sigma=[0.5 0.5 0.5 0.5 0.5 0.5]]:[-8.96515221e-07  2.34189008e-07 -9.26112899e-07 -7.87191754e-07 -1.26186896e-09 -2.01161302e-07]
# lora_ratios=array([-8.96515221e-07,  2.34189008e-07, -9.26112899e-07, -7.87191754e-07, -1.26186896e-09, -2.01161302e-07])
lora_module_list = [
    f"{multi_loras_dir}/Intel/neural-chat-7b-v3-1-lora",
    f"{multi_loras_dir}/migtissera/SynthIA-7B-v1.3-lora",
    # f"{multi_loras_dir}/HuggingFaceH4/zephyr-7b-alpha-lora",
    f"{multi_loras_dir}/jondurbin/airoboros-m-7b-3.1.2-lora",
    f"{multi_loras_dir}/bhenrym14/mistral-7b-platypus-fp16-lora",
    f"{multi_loras_dir}/teknium/CollectiveCognition-v1.1-Mistral-7B-lora",
    f"{multi_loras_dir}/uukuguy/speechless-mistral-dolphin-orca-platypus-samantha-7b-lora",
]

def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_learn", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="/opt/local/llm_models/huggingface.co/mistralai/Mistral-7B-v0.1")
    parser.add_argument("--lora_module_list", nargs="+", default=lora_module_list)
    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--example_inputs", type=str, default=None)
    parser.add_argument("--example_outputs", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_inference_step", type=int, default=100)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_learning_samples", type=int, default=64)

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
