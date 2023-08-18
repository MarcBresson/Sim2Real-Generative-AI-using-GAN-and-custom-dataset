from typing import Union

from torch import Tensor


def batcher(input_: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
    """
    in case individual images were provided, convert them to batches
    """
    if isinstance(input_, dict):
        if len(input_["streetview"].shape) == 3:
            input_["streetview"] = input_["streetview"].unsqueeze(0)
        if len(input_["simulated"].shape) == 3:
            input_["simulated"] = input_["simulated"].unsqueeze(0)

    elif isinstance(input_, Tensor):
        if len(input_.shape) == 3:
            input_ = input_.unsqueeze(0)

    return input_


class Sample2Batch():
    def __call__(self, input_: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
        """
        transform a single dict sample to a dict batchted sample
        transform a single sample to a batched sample

        Parameters
        ----------
        input_ : Union[dict[str, Tensor], Tensor]
            the single sample

        Returns
        -------
        Union[dict[str, Tensor], Tensor]
            the batched sample
        """
        if isinstance(input_, dict):
            if len(input_["streetview"].shape) == 3:
                input_["streetview"] = input_["streetview"].unsqueeze(0)
            if len(input_["simulated"].shape) == 3:
                input_["simulated"] = input_["simulated"].unsqueeze(0)

        elif isinstance(input_, Tensor):
            if len(input_.shape) == 3:
                input_ = input_.unsqueeze(0)

        else:
            raise TypeError(f"type {type(input_)} is not supported. Please use a Tensor or a "
                            "dict with keys `simulated` and `streetview`.")

        return input_


class Batch2Sample():
    def __call__(self, input_: Union[dict[str, Tensor], Tensor]) -> Union[dict[str, Tensor], Tensor]:
        """
        transform a dict batched sample to a dict single sample if the batch has 1 element
        transform a batched sample to a single sample if the batch has 1 element

        Parameters
        ----------
        input_ : Union[dict[str, Tensor], Tensor]
            batch, either in a dict or in a single Tensor.

        Returns
        -------
        Union[dict[str, Tensor], Tensor]
            single sample
        """
        if isinstance(input_, dict):
            if len(input_["streetview"].shape) == 4 and input_["streetview"].shape[0] == 1:
                input_["streetview"] = input_["streetview"][0]
            if len(input_["simulated"].shape) == 4 and input_["simulated"].shape[0] == 1:
                input_["simulated"] = input_["simulated"][0]

        elif isinstance(input_, Tensor):
            if len(input_.shape) == 4 and input_.shape[0] == 1:
                input_ = input_[0]

        else:
            raise TypeError(f"type {type(input_)} is not supported. Please use a Tensor or a "
                            "dict with keys `simulated` and `streetview`.")

        return input_
