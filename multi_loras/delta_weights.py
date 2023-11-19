# Derived from yule-BUAA/MergeLM/merging_methods.py
import torch.nn as nn


def is_exclude_param_name(param_name: str, exclude_param_names_regex: list):
    if exclude_param_names_regex:
        exclude = any(
            [
                re.match(exclude_pattern, param_name)
                for exclude_pattern in exclude_param_names_regex
            ]
        )
    else:
        exclude = False
    return exclude


def get_model_params(model: nn.Module):
    params_dict = {
        param_name: param_value for param_name, param_value in model.named_parameters()
    }
    return params_dict


def copy_params_to_model(params_dict: dict, model: nn.Module):
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params_dict[param_name])


class DeltaWeights:
    def __init__(
        self,
        base_model: nn.Module = None,
        tuned_model: nn.Module = None,
        exclude_param_names_regex: list = None,
        params_dict: dict = None,
    ):
        """
        Task vector. Initialize the task vector from a pretrained model and a tuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param base_model: nn.Module, base model
        :param tuned_model: nn.Module, tuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param params_dict: dict, prams dict to initialize self.params_dict
        """
        self.params_dict = {}

        if params_dict is not None:
            self.params_dict = params_dict
        else:
            base_params_dict = get_model_params(base_model)
            tuned_params_dict = get_model_params(tuned_model)
            for param_name in base_params_dict:
                self.params_dict[param_name] = (
                    tuned_params_dict[param_name] - base_params_dict[param_name]
                )

    def __add__(self, other):
        """
        add task vector
        :param other: DeltaWeights to add, at right side
        :return:
        """
        assert isinstance(
            other, DeltaWeights
        ), "addition of DeltaWeights can only be done with another DeltaWeights!"
        new_params_dict = {}
        for param_name in self.params_dict:
            assert (
                param_name in other.params_dict.keys()
            ), f"param_name {param_name} is not contained in both params!"
            new_params_dict[param_name] = (
                self.params_dict[param_name] + other.param_dict[param_name]
            )
        return DeltaWeights(params_dict=new_params_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: DeltaWeights to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(
        self, base_model: nn.Module, scaling_coefficient: float = 1.0
    ):
        """
        combine the delta weights with pretrained model
        :param base_model: nn.Module, base model
        :param scaling_coefficient: float, scaling coefficient to merge the delta weights
        :return:
        """
        base_params_dict = get_model_params(base_model)

        with torch.no_grad():
            merged_params = {}
            for param_name in self.params_dict:
                merged_params[param_name] = (
                    base_params_dict[param_name]
                    + scaling_coefficient * self.params_dict[param_name]
                )

        return merged_params
