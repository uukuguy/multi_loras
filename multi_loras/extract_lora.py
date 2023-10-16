#!/usr/bin/env python

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model

def find_all_linear_names(model, bits=16):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_lora_modules_count(model, bits):
    n = 0
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    for name, module in model.named_modules():
        if isinstance(module, cls):
            n += 1
    return n


def _iter_lora(model, bits):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    for name, module in model.named_modules():
        if isinstance(module, cls):
            yield name, module


def prepare_model_kwargs(args):
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "load_in_4bit": args.bits == 4,
        "load_in_8bit": args.bits == 8,
        "device_map": "cpu",
        "max_memory": None,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if args.bits in (4, 8) else None,
        "torch_dtype": (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        "trust_remote_code": args.trust_remote_code,
        "use_flash_attention_2": args.flash_attention,
        # "use_auth_token": args.use_auth_token
    }

    return model_kwargs


def create_lora_model(args, model):
    modules = find_all_linear_names(model)
    print(f"{modules=}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def load_model_and_init_lora(args, model_name_or_path, model_kwargs):
    print(f"Loading model from {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = create_lora_model(args, model)
    for k, params in model.named_parameters():
        print(k, params.shape)
    print(f"Loaded model from {model_name_or_path}.")
    return model

# SVD on residual
# Below snippet is from https://github.com/cloneofsimo/lora/lora_diffusion/cli_svd.py
def svd_distill(residual, rank, clamp_quantile=1):
    residual = residual.float()
    U, S, Vh = torch.linalg.svd(residual)
    U = U[:, :rank]
    S = S[:rank]
    U = U @ torch.diag(S)

    Vh = Vh[:rank, :]

    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile)
    low_val = -hi_val

    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    return U, Vh


def do_extract_lora(args):
    # Load base model and tuned model
    model_kwargs = prepare_model_kwargs(args)
    base_model = load_model_and_init_lora(args, args.base_model_name_or_path, model_kwargs)
    tuned_model = load_model_and_init_lora(args, args.tuned_model_name_or_path, model_kwargs)

    bits = args.bits
    num_base_lora_modules = get_lora_modules_count(base_model, bits)
    num_tuned_lora_modules = get_lora_modules_count(tuned_model, bits)
    assert num_base_lora_modules == num_tuned_lora_modules, f"{num_base_lora_modules=}, {num_tuned_lora_modules=}"
    pbar = tqdm(zip(_iter_lora(base_model, bits), _iter_lora(tuned_model, bits)), 
                total=num_base_lora_modules, ncols=120, desc="Run SVD")

    rank = args.lora_r
    clamp_quantile = args.clamp_quantile
    device = base_model.device
    dtype = base_model.dtype

    for (name_base, lora_base), (name_tuned, lora_tune) in pbar:
        assert name_base == name_tuned, f"name_base={name_base} != name_tuned={name_tuned}"

        residual = lora_tune.weight.data - lora_base.weight.data
        pbar.set_postfix({"layer": name_base.replace("base_model.model.", ""), "shape": residual.shape})

        # SVD on residual
        U, Vh = svd_distill(residual, rank=rank, clamp_quantile=clamp_quantile)

        assert lora_base.lora_A.default.weight.shape == Vh.shape
        assert lora_base.lora_B.default.weight.shape == U.shape

        lora_base.lora_A.default.weight.data = Vh.to(device=device, dtype=dtype)
        lora_base.lora_B.default.weight.data = U.to(device=device, dtype=dtype)

    # Save the distilled model
    print(f"Saving peft model to {args.save_path} ...")
    base_model.save_pretrained(args.save_path)
    print(f"Save done.")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--tuned_model_name_or_path", type=str, required=True, help="Path to the tuned model.")
    parser.add_argument("--save_path", type=str, default="svd_distill_model", help="Path to save the distilled model.")
    parser.add_argument("--bits", type=int, default=4, help="Bits to use for quantization.")
    parser.add_argument("--lora_r", type=int, default=64, help="Rank for LORA.")
    parser.add_argument("--lora_alpha", type=float, default=16, help="Alpha for LORA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LORA.")
    parser.add_argument("--double_quant", action="store_true", default=True, help="Compress the quantization statistics through double quantization.")
    parser.add_argument("--quant_type", type=str, default="nf4", help="Quantization data type to use. Should be one of `fp4` or `nf4`.")
    parser.add_argument("--clamp_quantile", type=float, default=1.0, help="Clamp the quantization range to this quantile of the distribution.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use fp16 for training.")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use bf16 for training.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to cache dir.")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="Trust remote code.")
    parser.add_argument("--flash_attention", action="store_true", default=False, help="Use flash attention.")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    do_extract_lora(args)

if __name__ == '__main__':
    main()