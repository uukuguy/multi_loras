#!/usr/bon/env python
"""
This script is used to do drop and rescale for the tuned model
"""
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .delta_weights import DeltaWeights, copy_params_to_model

default_dare_kwargs = {
    "weight_mask_rate": 0.85,
    "use_weight_rescale": True,
    "mask_strategy": "random",
    "scaling_coefficient": 1.0,
}


# DARE (Drop And REscale)
# Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch
# https://arxiv.org/abs/2311.03099
def drop_and_rescale_tensor(
    input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str
):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert (
        0.0 <= mask_rate <= 1.0
    ), f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(
            torch.full_like(input=input_tensor.float(), fill_value=mask_rate)
        ).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert (
            mask_strategy == "magnitude"
        ), f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ),
        # find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(
            k=num_mask_params, dim=0, keepdim=True
        )
        # Tensor, shape (num_total_params, ),
        # where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


def drop_and_rescale_model(
    tuned_model: nn.Module,
    base_model: nn.Module,
    exclude_param_names_regex: list = None,
    weight_mask_rate: float = 0.85,
    use_weight_rescale: bool = True,
    mask_strategy: str = "random",
    scaling_coefficient: float = 1.0,
):
    """
    mask model weights
    :param tuned_model: nn.Module, the tuned model
    :param base_model: nn.Module, the base model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    # get weights that need to be masked
    delta_weights = DeltaWeights(
        base_model=base_model,
        tuned_model=tuned_model,
        exclude_param_names_regex=exclude_param_names_regex,
    )
    model_param_dict = delta_weights.params_dict

    with torch.no_grad():
        dare_params_dict = {}
        for param_name, param_value in tqdm(model_param_dict.items(), ncols=0):
            dare_params_dict[param_name] = drop_and_rescale_tensor(
                input_tensor=param_value,
                mask_rate=weight_mask_rate,
                use_rescale=use_weight_rescale,
                mask_strategy=mask_strategy,
            )

        new_delta_weights = DeltaWeights(params_dict=dare_params_dict)
        # combine with parameters of the merged model based on scaling coefficient
        dare_model_weights = new_delta_weights.combine_with_pretrained_model(
            base_model=base_model, scaling_coefficient=scaling_coefficient
        )

    return dare_model_weights


def do_dare(args):
    """
    This function is used to do drop and rescale for the tuned model
    """
    print(f"Loading base model from {args.base_model_name_or_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, device_map=args.device_map, trust_remote_code=True
    ).half()
    print(f"Loading tuned model from {args.tuned_model_name_or_path} ...")
    tuned_model = AutoModelForCausalLM.from_pretrained(
        args.tuned_model_name_or_path,
        device_map=args.device_map,
        trust_remote_code=True,
    ).half()
    tokenizer = AutoTokenizer.from_pretrained(args.tuned_model_name_or_path, trust_remote_code=True)

    dare_kwargs = {
        "weight_mask_rate": args.dare_weight_mask_rate,
        "use_weight_rescale": args.dare_use_weight_rescale,
        "mask_strategy": args.dare_mask_strategy,
        "scaling_coefficient": args.dare_scaling_coefficient,
    }
    print(
        f"Do drop and rescale with {dare_kwargs=} with {args.tuned_model_name_or_path} ..."
    )
    model_weights = drop_and_rescale_model(
        tuned_model=tuned_model,
        base_model=base_model,
        **dare_kwargs,
    )
    copy_params_to_model(model_weights, base_model)
    print(f"Saving model to {args.save_path} ...")
    tokenizer.save_pretrained(args.save_path)
    base_model.save_pretrained(args.save_path)

    print(f"Saved model to {args.save_path}")
