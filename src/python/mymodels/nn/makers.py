from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional, TypeVar

import torch as th
import torchinfo as thinfo

ModuleT = TypeVar("ModuleT", bound=th.nn.Module)


def make_lazy_nnet(
    in_features: int | None,
    out_features: int,
    layers: Sequence[th.nn.Module] | OrderedDict[str, th.nn.Module] | th.nn.Sequential,
    input_unflatten_shape: Optional[tuple[int, ...]] = None,
    verbose: Optional[int] = None,
) -> th.nn.Sequential:
    """Make a neural network with a lazily initialized output layer.

    The given ``layers`` are assembled into a sequential network, optionally
    preceded by an unflatten layer and always followed by a flatten layer and
    a lazily initialized linear output layer. If ``in_features`` is given
    (or can be derived from ``input_unflatten_shape``), the network is
    sanity-checked by running a ``torchinfo`` summary against an input of
    size ``in_features``, which also initializes the lazy output layer.

    Parameters
    ----------
    in_features : int, optional
        Input feature size of the network. If ``input_unflatten_shape`` is
        given, it must equal its product; if None, it is derived from
        ``input_unflatten_shape``. If None and no ``input_unflatten_shape``
        is given, the sanity check is skipped and the network is returned
        with its lazy layers uninitialized (they initialize on the first
        forward pass).
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
        if in_features is not None:
            assert in_features == math.prod(input_unflatten_shape)
        else:
            in_features = math.prod(input_unflatten_shape)
        nnet.insert(0, th.nn.Unflatten(dim=1, unflattened_size=input_unflatten_shape))
    nnet.extend([th.nn.Flatten(), th.nn.LazyLinear(out_features=out_features)])
    # sanity check
    if in_features is not None:
        _summary = thinfo.summary(
            nnet, input_size=(in_features,), batch_dim=0, verbose=verbose
        )
        if verbose is not None and verbose != 0:
            lgr = logging.getLogger(make_lazy_nnet.__name__)
            lgr.addHandler(logging.StreamHandler())
            lgr.setLevel(logging.INFO)
            lgr.info(_summary)
    return nnet
