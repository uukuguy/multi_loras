#!/usr/bin/env python
from .extract_lora import do_extract_lora
from .merge_peft_adapters import do_merge_lora
from .dare import do_dare
from .delta_weights import do_delta_weights

cmd_functions = {
    "extract_lora": do_extract_lora,
    "merge_lora": do_merge_lora,
    "drop_and_rescale": do_dare,
    "delta_weights": do_delta_weights,
}
available_commands = cmd_functions.keys()


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "cmd",
        type=str,
        choices=available_commands,
        help=f"Command to run. Should be one of {available_commands}",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="Path to the base model.",
    )
    parser.add_argument(
        "--tuned_model_name_or_path", type=str, help="Path to the tuned model."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="svd_distill_model",
        help="Path to save the distilled model.",
    )
    parser.add_argument(
        "--device_map", type=str, default="cpu", help="Path to device map."
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Bits to use for quantization. Only support 4 or 8.",
    )
    parser.add_argument("--lora_r", type=int, default=64, help="Rank for LORA.")
    parser.add_argument("--lora_alpha", type=float, default=16, help="Alpha for LORA.")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="Dropout for LORA."
    )
    parser.add_argument(
        "--double_quant",
        action="store_true",
        default=True,
        help="Compress the quantization statistics through double quantization.",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="nf4",
        help="Quantization data type to use. Should be one of `fp4` or `nf4`.",
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=1.0,
        help="Clamp the quantization range to this quantile of the distribution.",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Use fp16 for training."
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="Use bf16 for training."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Path to cache dir."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code.",
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        default=False,
        help="Use flash attention.",
    )

    parser.add_argument("--lora_model_path", type=str)
    parser.add_argument("--merged_model_name_or_path", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", default=False)

    parser.add_argument("--do_dare", action="store_true")
    parser.add_argument("--dare_weight_mask_rate", type=float, default=0.85)
    parser.add_argument("--dare_use_weight_rescale", action="store_true")
    parser.add_argument("--dare_mask_strategy", type=str, default="random")
    parser.add_argument("--dare_scaling_coefficient", type=float, default=1.0)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cmd_func = cmd_functions.get(args.cmd, None)
    if cmd_func is not None:
        cmd_func(args)
    else:
        print(f"Command {args.cmd} not found. Should be one of {available_commands}")


if __name__ == "__main__":
    main()
