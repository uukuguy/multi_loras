# https://github.com/bigcode-project/starcoder/blob/main/finetune/merge_peft_adapters.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# from utils.model_utils import merge_peft_adapters
def merge_peft_adapters(base_model_name_or_path, lora_model_path, merged_model_name_or_path=None, push_to_hub=False):
    if merged_model_name_or_path is None:
        merged_model_name_or_path = f"{base_model_name_or_path}-merged"

    print(f"Loading base model from {base_model_name_or_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading peft model from {lora_model_path} ...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    print(f"Merging ...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)

    if push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(merged_model_name_or_path, use_temp_dir=False, private=True)
        tokenizer.push_to_hub(merged_model_name_or_path, use_temp_dir=False, private=True)
    else:
        print(f"Saving to {merged_model_name_or_path} ...")
        model.save_pretrained(merged_model_name_or_path)
        tokenizer.save_pretrained(merged_model_name_or_path)
        print(f"Model saved to {merged_model_name_or_path}")
    
def do_merge_lora(args):
    merge_peft_adapters(base_model_name_or_path=args.base_model_name_or_path, 
                        lora_model_path=args.lora_model_path, 
                        merged_model_name_or_path=args.merged_model_name_or_path,
                        push_to_hub=args.push_to_hub
                        )

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--lora_model_path", type=str)
    parser.add_argument("--merged_model_name_or_path", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", default=False)

    args = parser.parse_args()
    do_merge_lora(args)



if __name__ == "__main__" :
    main()