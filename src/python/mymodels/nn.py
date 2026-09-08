from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, TypeVar

import tensordict.nn as thd_nn
import torch as th
import torchinfo as thinfo
from tensordict import NestedKey

ModuleT = TypeVar("ModuleT", bound=th.nn.Module)


def make_lazy_nnet(
    in_features: int,
    out_features: int,
    layers: Sequence[th.nn.Module] | OrderedDict[str, th.nn.Module] | th.nn.Sequential,
    input_unflatten_shape: Optional[tuple[int, ...]] = None,
    verbose: Optional[int] = None,
) -> th.nn.Sequential:
    """Make a neural network with a lazily initialized output layer.

    The given ``layers`` are assembled into a sequential network, optionally
    preceded by an unflatten layer and always followed by a flatten layer and
    a lazily initialized linear output layer. The network is sanity-checked
    by running a ``torchinfo`` summary against an input of size
    ``in_features``, which also initializes the lazy output layer.

    Parameters
    ----------
    in_features : int
        Input feature size of the network. If ``input_unflatten_shape`` is
        given, it must equal its product.
    out_features : int
        Output feature size of the network.
    layers : sequence of torch.nn.Module, OrderedDict, or torch.nn.Sequential
        The hidden layers of the network. An ``OrderedDict`` gives the layers
        names; a ``Sequential`` is used as is.
    input_unflatten_shape : tuple of int, optional
        If given, an ``Unflatten`` layer reshaping the flattened input to
        this shape is prepended to the network.
    verbose : int, optional
        Verbosity of the ``torchinfo`` summary. If not None and not 0, the
        summary is also logged at INFO level.

    Returns
    -------
    torch.nn.Sequential
        The assembled network.
    """
    if isinstance(layers, th.nn.Sequential):
        nnet = layers
    elif isinstance(layers, OrderedDict):
        nnet = th.nn.Sequential(layers)
    else:
        nnet = th.nn.Sequential(*layers)
    if input_unflatten_shape is not None:
        assert in_features == math.prod(input_unflatten_shape)
        nnet.insert(0, th.nn.Unflatten(dim=1, unflattened_size=input_unflatten_shape))
    nnet.extend([th.nn.Flatten(), th.nn.LazyLinear(out_features=out_features)])
    # sanity check
    _summary = thinfo.summary(
        nnet, input_size=(in_features,), batch_dim=0, verbose=verbose
    )
    if verbose is not None and verbose != 0:
        lgr = logging.getLogger(make_lazy_nnet.__name__)
        lgr.addHandler(logging.StreamHandler())
        lgr.setLevel(logging.INFO)
        lgr.info(_summary)
    return nnet


class Cat(th.nn.Module):
    """Concatenate a variable number of tensors into a single tensor.

    Parameters
    ----------
    dim : int
        Dimension along which the tensors are concatenated.
    unsqueeze_if_oor : bool
        If True, unsqueeze any input tensor for which ``dim`` is out of
        range along ``dim`` before concatenation. This is useful to mix
        per-row scalar features of shape ``[N]`` (pass ``dim=1`` so they
        become ``[N, 1]``) with already 2-D features of shape ``[N, k]``.

    Examples
    --------
    Concatenate two 2-D tensors along the feature dimension:

    >>> import torch as th
    >>> cat = Cat(dim=-1, unsqueeze_if_oor=False)
    >>> cat(th.randn(5, 3), th.randn(5, 2)).shape
    torch.Size([5, 5])

    Mix a per-row scalar tensor of shape ``[N]`` with a 2-D tensor by
    unsqueezing the former to ``[N, 1]`` (``dim=1`` is out of range for a
    1-D tensor):

    >>> cat = Cat(dim=1, unsqueeze_if_oor=True)
    >>> cat(th.randn(5), th.randn(5, 2)).shape
    torch.Size([5, 3])
    """

    def __init__(self, dim: int, unsqueeze_if_oor: bool) -> None:
        super().__init__()
        self.dim = dim
        self.unsqueeze_if_oor = unsqueeze_if_oor

    def forward(self, *tensors: th.Tensor) -> th.Tensor:
        if self.unsqueeze_if_oor:
            tensors = tuple(
                (
                    t.unsqueeze(self.dim if self.dim >= 0 else self.dim + t.ndim + 1)
                    if (self.dim >= t.ndim or self.dim < -t.ndim)
                    else t
                )
                for t in tensors
            )
        return th.cat(tensors, dim=self.dim)


class CatTensorDictTensors(thd_nn.TensorDictModule):
    """Concatenate multiple TensorDict entries into a single tensor.

    Parameters
    ----------
    in_keys : sequence of NestedKey
        Keys of the tensors to concatenate, in order. Nested keys (e.g.
        ``("feature", "a")``) are supported.
    out_key : NestedKey
        Key the concatenated tensor is written to.
    dim : int, default=-1
        Dimension along which the tensors are concatenated.
    unsqueeze_if_oor : bool, default=False
        If True, unsqueeze any input tensor for which ``dim`` is out of
        range along ``dim`` before concatenation. This is useful to mix
        per-row scalar features of shape ``[N]`` (pass ``dim=1`` so they
        become ``[N, 1]``) with already 2-D features of shape ``[N, k]``.

    Examples
    --------
    Concatenate two feature entries along the feature dimension:

    >>> import torch as th
    >>> from tensordict import TensorDict
    >>> data = TensorDict(
    ...     {
    ...         "feature": TensorDict(
    ...             {"a": th.randn(5, 3), "b": th.randn(5, 2)}, batch_size=[5]
    ...         )
    ...     },
    ...     batch_size=[5],
    ... )
    >>> cat = CatTensorDictTensors(
    ...     in_keys=[("feature", "a"), ("feature", "b")],
    ...     out_key="features",
    ... )
    >>> cat(data)["features"].shape
    torch.Size([5, 5])

    Mix a per-row scalar entry of shape ``[N]`` with a 2-D entry of shape
    ``[N, k]`` by unsqueezing the former to ``[N, 1]`` (``dim=1`` is out of
    range for a 1-D tensor):

    >>> data = TensorDict(
    ...     {
    ...         "feature": TensorDict(
    ...             {"a": th.randn(5), "b": th.randn(5, 2)}, batch_size=[5]
    ...         )
    ...     },
    ...     batch_size=[5],
    ... )
    >>> cat = CatTensorDictTensors(
    ...     in_keys=[("feature", "a"), ("feature", "b")],
    ...     out_key="features",
    ...     dim=1,
    ...     unsqueeze_if_oor=True,
    ... )
    >>> cat(data)["features"].shape
    torch.Size([5, 3])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_key: NestedKey,
        dim: int = -1,
        unsqueeze_if_oor: bool = False,
    ) -> None:
        super().__init__(
            module=Cat(dim=dim, unsqueeze_if_oor=unsqueeze_if_oor),
            in_keys=list(in_keys),
            out_keys=[out_key],
        )
