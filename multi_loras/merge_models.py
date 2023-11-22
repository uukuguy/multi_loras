#!/usr/bin/env python
# Derived from yule-BUAA/MergeLM/merging_methods.py

import copy
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn
from delta_weights import (
    DeltaWeights,
    copy_params_to_model,
    get_model_params,
    is_excluded_param_name,
)
from dare import drop_and_rescale_model

def apply_dare_to_model(models_to_merge: list, base_model: nn.Module, dare_kwargs: dict):
    if dare_kwargs is not None:
        assert base_model is not None, f"base_model should not be None if dare_kwargs is not None"
        models_to_merge = [
            drop_and_rescale_model(
                base_model=base_model,
                tuned_model=tuned_model,
                **dare_kwargs,
            )
            for tuned_model in models_to_merge
        ]

    return models_to_merge


def average_merging(
    models_to_merge: list,
    base_model: nn.Module = None,
    dare_kwargs: dict = None,
    exclude_param_names_regex: list = None,
):
    models_to_merge_param_dict = defaultdict(list)

    models_to_merge = apply_dare_to_model(models_to_merge, base_model, dare_kwargs)

    for model in models_to_merge:
        params_dict = get_model_params(model)
        for param_name, param_value in params_dict.items():
            if not is_excluded_param_name(param_name, exclude_param_names_regex):
                models_to_merge_param_dict[param_name].append(param_value)

    with torch.no_grad():
        averaged_params = {
            param_name: torch.stack(param_value, dim=0).mean(dim=0)
            for param_name, param_value in models_to_merge_param_dict.items()
        }

    return averaged_params


def task_arithmetic_merging(
    base_model: nn.Module,
    models_to_merge: list,
    dare_kwargs: dict = None,
    exclude_param_names_regex: list = None,
    scaling_coefficient: float = 1.0,
):

    models_to_merge = apply_dare_to_model(models_to_merge, base_model, dare_kwargs)

    delta_weights_list = [
        DeltaWeights(
            base_model=base_model,
            tuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    # iterate each individual model that needs to be merged
    with torch.no_grad():
        # sum up the task vectors
        merged_delta_weights = delta_weights_list[0] + delta_weights_list[1]
        for index in range(2, len(delta_weights_list)):
            merged_delta_weights = merged_delta_weights + delta_weights_list[index]

        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_delta_weights.combine_with_pretrained_model(
            base_model=base_model, scaling_coefficient=scaling_coefficient
        )

    return merged_params


def ties_merging(
    base_model: nn.Module,
    models_to_merge: list,
    exclude_param_names_regex: list = None,
    param_value_mask_rate: float = 0.8,
    scaling_coefficient: float = 1.0,
    dare_kwargs = None,
):
    """
    ties merging method
    :param base_model: nn.Module, the base model
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
    :param scaling_coefficient: float, scaling coefficient to merge the task vectors
    :return:
    """

    def delta_weights_to_single_vector(delta_weights: DeltaWeights):
        """
        convert parameter dictionary in task vector to a single vector
        :param delta_weights: DeltaWeights, delta weights
        :return:
        """
        params_dict = copy.deepcopy(delta_weights.params_dict)
        sorted_params_dict = OrderedDict(sorted(params_dict.items()))

        # Tensor, shape (num_total_params, )
        return nn.utils.parameters_to_vector(
            [param.flatten() for param in sorted_params_dict.values()]
        )

    def single_vector_to_delta_weights(
        single_vector: torch.Tensor, delta_weights: DeltaWeights
    ):
        """
        convert a single vector to parameter dictionary in task vector
        :param single_vector: Tensor, single vector that contain all parameters in delta_weights.params_dict
        :param delta_weights: DeltaWeights, delta weights
        :return:
        """
        params_dict = copy.deepcopy(delta_weights.params_dict)
        sorted_params_dict = OrderedDict(sorted(params_dict.items()))

        nn.utils.vector_to_parameters(single_vector, sorted_params_dict.values())

        return sorted_params_dict

    def mask_smallest_magnitude_param_values(
        flattened_models_to_merge_param: torch.Tensor,
        param_value_mask_rate: float = 0.8,
    ):
        """
        mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :return:
        """
        # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
        num_mask_params = int(
            flattened_models_to_merge_param.shape[1] * param_value_mask_rate
        )

        # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
        kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(
            k=num_mask_params, dim=1, keepdim=True
        )
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        mask = flattened_models_to_merge_param.abs() >= kth_values

        return flattened_models_to_merge_param * mask

    def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
        """
        get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
        # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
        majority_sign = torch.sign(param_signs.sum(dim=0))
        param_signs[param_signs == 0] = majority_sign
        return param_signs

    def disjoint_merge(
        flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor
    ):
        """
        disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
        :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
        :return:
        """
        # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
        param_to_preserve_mask = (
            (param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)
        ) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
        # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
        param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

        # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
        num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
        # Tensor, shape (num_total_params, ), the averaged flattened parameters
        merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(
            num_models_param_preserved, min=1.0
        )

        return merged_flattened_param

    models_to_merge = apply_dare_to_model(models_to_merge, base_model, dare_kwargs)

    models_to_merge_delta_weights = [
        DeltaWeights(
            base_model=base_model,
            tuned_model=model_to_merge,
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for model_to_merge in models_to_merge
    ]

    flattened_models_to_merge_param = [
        delta_weights_to_single_vector(task_vector=task_vector)
        for task_vector in models_to_merge_delta_weights
    ]
    # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    flattened_models_to_merge_param = torch.vstack(flattened_models_to_merge_param)

    with torch.no_grad():
        # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
        flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
            flattened_models_to_merge_param=flattened_models_to_merge_param,
            param_value_mask_rate=param_value_mask_rate,
        )

        # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
        param_signs = get_param_signs(
            flattened_models_to_merge_param=flattened_models_to_merge_param
        )

        # Tensor, shape (num_total_params, ), disjoint merge
        merged_flattened_param = disjoint_merge(
            flattened_models_to_merge_param=flattened_models_to_merge_param,
            param_signs=param_signs,
        )

        # merged parameter dictionary
        merged_params_dict = single_vector_to_delta_weights(
            single_vector=merged_flattened_param,
            task_vector=models_to_merge_delta_weights[0],
        )
        merged_delta_weights = DeltaWeights(params_dict=merged_params_dict)
        # combine with parameters of the merged model based on scaling coefficient
        merged_params = merged_delta_weights.combine_with_pretrained_model(
            base_model=base_model, scaling_coefficient=scaling_coefficient
        )

    return merged_params
