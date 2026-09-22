from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import tensordict as thd
import torch as th


class CatTensorDictTensors(th.nn.Module):
    """Concatenate multiple TensorDict entries into a single tensor.

    Parameters
    ----------
    in_keys : sequence of NestedKey
        Keys of the tensors to concatenate, in order. Nested keys (e.g.
        ``("feature", "a")``) are supported.
    dim : int, default=-1
        Dimension along which the tensors are concatenated.
    unsqueeze_if_oor : bool, default=False
        If True, unsqueeze any input tensor for which ``dim`` is out of
        range along ``dim`` before concatenation. This is useful to mix
        per-row scalar features of shape ``[N]`` (pass ``dim=1`` so they
        become ``[N, 1]``) with already 2-D features of shape ``[N, k]``.
    out_key : NestedKey, optional
        If None (default), the concatenated tensor itself is returned. If
        given, the concatenated tensor is instead written to a TensorDict
        under this key and that TensorDict is returned.
    inplace : bool, default=True
        Only used when ``out_key`` is given, with the same semantics as
        ``tensordict.nn.TensorDictModule``: if True, the concatenated tensor
        is written into the input TensorDict, which is modified and
        returned; if False, the input is left untouched and a new TensorDict
        containing only the ``out_key`` entry is returned.

    Notes
    -----
    The ``in_keys``/``out_keys`` attributes are exposed so that
    ``tensordict.nn.TensorDictSequential`` treats instances as
    TensorDictModule-compatible (its compatibility check only requires those
    two attributes); with ``out_key=None``, ``out_keys`` is empty and the
    module returns a plain tensor, which only makes sense outside a
    TensorDict pipeline.

    Examples
    --------
    Concatenate two feature entries along the feature dimension, returning
    the concatenated tensor (default):

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
    >>> cat = CatTensorDictTensors(in_keys=[("feature", "a"), ("feature", "b")])
    >>> cat(data).shape
    torch.Size([5, 5])

    Write the concatenated tensor back into the TensorDict instead:

    >>> cat = CatTensorDictTensors(
    ...     in_keys=[("feature", "a"), ("feature", "b")],
    ...     out_key="features",
    ... )
    >>> cat(data)["features"].shape
    torch.Size([5, 5])

    With ``inplace=False``, the input TensorDict is left untouched and a new
    TensorDict holding only the output entry is returned:

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
    ...     inplace=False,
    ... )
    >>> out = cat(data)
    >>> list(out.keys())
    ['features']
    >>> "features" in data
    False

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
    ...     dim=1,
    ...     unsqueeze_if_oor=True,
    ... )
    >>> cat(data).shape
    torch.Size([5, 3])
    """

    def __init__(
        self,
        in_keys: Sequence[thd.NestedKey],
        dim: int = -1,
        unsqueeze_if_oor: bool = False,
        out_key: thd.NestedKey | None = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.in_keys = list(in_keys)
        self.out_keys = [] if out_key is None else [out_key]
        self.inplace = inplace
        self.cat = _Cat(dim=dim, unsqueeze_if_oor=unsqueeze_if_oor)

    def forward(self, data: thd.TensorDict) -> thd.TensorDict | th.Tensor:
        features = self.cat(*(data[key] for key in self.in_keys))
        if not self.out_keys:
            return features
        if self.inplace:
            data[self.out_keys[0]] = features
            return data
        tensordict_out = thd.TensorDict()
        tensordict_out[self.out_keys[0]] = features
        return tensordict_out


class CatTensorDictTensorsWithMissingIndicator(th.nn.Module):
    """Concatenate TensorDict entries, zero-filling missing ones and
    appending a missingness indicator.

    Two input contracts, selected by ``missing_repr`` (required, so the
    choice is always intentional):

    * ``"nan"`` — per-key entries. Features whose key is absent from the
      input TensorDict are treated as missing: their block is filled with
      zeros (using the shape declared in ``in_keys_and_shape``) and the
      corresponding block of the indicator is set to zeros. A present entry
      may still have individual missing values marked as NaN (e.g. when
      instances within a batch differ in which features are observed); NaN
      values are zero-filled as well.
    * ``"mask"`` — a single dense values entry under ``values_key``
      (already zero-filled by the caller) plus a bool/0-1 observed-feature
      mask under ``mask_key``, both with trailing width equal to the total
      declared block width. The forward is a single dtype cast and a single
      concatenation — the memory-efficient path for large batches. Missing
      keys play no role here; ``in_keys_and_shape`` only documents the
      feature order the dense tensors must follow.

    The indicator holds ones where a value is observed and zeros where it
    is missing. If the plain concatenation has shape ``[N, D]``, this
    module's output has shape ``[N, 2 * D]``: the concatenated features
    followed by the indicator along ``dim``.

    Parameters
    ----------
    in_keys_and_shape : dict of NestedKey to sequence of int
        Keys of the tensors to concatenate, in insertion order, mapped to the shape
        of the entry's non-batch dimensions (e.g. ``(3,)`` for an entry of
        shape ``[N, 3]``). In ``"nan"`` mode the declared shape is used to
        zero-fill the entry when its key is absent from the input
        TensorDict; in ``"mask"`` mode it only fixes the expected block
        layout. Nested keys (e.g. ``("feature", "a")``) are supported.
    missing_repr : {"nan", "mask"}
        How missing features are represented in the input TensorDict (see
        above). Required — no default is configured.
    values_key : NestedKey, default="features"
        ``"mask"`` mode only: key of the dense, already zero-filled values
        tensor.
    mask_key : NestedKey, default="mask"
        ``"mask"`` mode only: key of the bool/0-1 observed-feature mask.
    dim : int, default=-1
        Dimension along which the tensors are concatenated.
    unsqueeze_if_oor : bool, default=False
        ``"nan"`` mode only. If True, unsqueeze any input tensor for which
        ``dim`` is out of range along ``dim`` before concatenation. This is
        useful to mix per-row scalar features of shape ``[N]`` (pass
        ``dim=1`` so they become ``[N, 1]``) with already 2-D features of
        shape ``[N, k]``.
    out_key : NestedKey, optional
        If None (default), the output tensor itself is returned. If given,
        the output tensor is instead written to a TensorDict under this
        key and that TensorDict is returned.
    inplace : bool, default=True
        Only used when ``out_key`` is given, with the same semantics as
        ``tensordict.nn.TensorDictModule``: if True, the output tensor is
        written into the input TensorDict, which is modified and returned;
        if False, the input is left untouched and a new TensorDict
        containing only the ``out_key`` entry is returned.

    Notes
    -----
    To use an instance inside a ``tensordict.nn.TensorDictSequential``, set
    ``out_key`` and wrap it with ``tensordict.nn.WrapModule``, passing ``in_keys``
    and ``out_keys`` explicitly so the sequential can track keys::

        WrapModule(
            cat,
            in_keys=list(cat.in_keys_and_shape),
            out_keys=cat.out_keys,
        )

    In ``"mask"`` mode, pass ``in_keys=[cat.values_key, cat.mask_key]``
    instead. With ``out_key=None``, ``out_keys`` is empty and the module
    returns a plain tensor, which only makes sense outside a TensorDict
    pipeline.

    In ``"nan"`` mode the zero-filled entries and the indicator take the
    device and dtype of the first present input tensor, defaulting to the
    TensorDict's device and ``torch.float32`` when every feature is
    missing. Entries with a non-floating-point dtype cannot hold NaN and
    are always treated as fully observed.

    Examples
    --------
    Concatenate two feature entries where the second is missing, returning
    the features followed by the missingness indicator:

    >>> import torch as th
    >>> from tensordict import TensorDict
    >>> data = TensorDict(
    ...     {"feature": TensorDict({"a": th.ones(5, 3)}, batch_size=[5])},
    ...     batch_size=[5],
    ... )
    >>> cat = CatTensorDictTensorsWithMissingIndicator(
    ...     in_keys_and_shape={("feature", "a"): (3,), ("feature", "b"): (2,)},
    ...     missing_repr="nan",
    ... )
    >>> out = cat(data)
    >>> out.shape
    torch.Size([5, 10])
    >>> out[0]
    tensor([1., 1., 1., 0., 0., 1., 1., 1., 0., 0.])

    Write the output back into the TensorDict instead:

    >>> cat = CatTensorDictTensorsWithMissingIndicator(
    ...     in_keys_and_shape={("feature", "a"): (3,), ("feature", "b"): (2,)},
    ...     missing_repr="nan",
    ...     out_key="features",
    ... )
    >>> cat(data)["features"].shape
    torch.Size([5, 10])

    Present entries may still contain NaN values for instances whose
    feature is missing; those are zero-filled and flagged element-wise in
    the indicator:

    >>> data = TensorDict(
    ...     {
    ...         "feature": TensorDict(
    ...             {
    ...                 "a": th.tensor([[1.0, th.nan], [3.0, 4.0]]),
    ...                 "b": th.tensor([[5.0], [th.nan]]),
    ...             },
    ...             batch_size=[2],
    ...         )
    ...     },
    ...     batch_size=[2],
    ... )
    >>> cat = CatTensorDictTensorsWithMissingIndicator(
    ...     in_keys_and_shape={("feature", "a"): (2,), ("feature", "b"): (1,)},
    ...     missing_repr="nan",
    ... )
    >>> cat(data)
    tensor([[1., 0., 5., 1., 0., 1.],
            [3., 4., 0., 1., 1., 0.]])

    In ``"mask"`` mode the same result is produced from a dense,
    already zero-filled values tensor and an explicit mask:

    >>> data = TensorDict(
    ...     {
    ...         "features": th.tensor([[1.0, 0.0, 5.0], [3.0, 4.0, 0.0]]),
    ...         "mask": th.tensor([[True, False, True], [True, True, False]]),
    ...     },
    ...     batch_size=[2],
    ... )
    >>> cat = CatTensorDictTensorsWithMissingIndicator(
    ...     in_keys_and_shape={("feature", "a"): (2,), ("feature", "b"): (1,)},
    ...     missing_repr="mask",
    ... )
    >>> cat(data)
    tensor([[1., 0., 5., 1., 0., 1.],
            [3., 4., 0., 1., 1., 0.]])
    """

    in_keys_and_shape: dict[thd.NestedKey, tuple[int, ...]]
    missing_repr: Literal["nan", "mask"]
    values_key: thd.NestedKey
    mask_key: thd.NestedKey
    out_keys: list[thd.NestedKey]
    inplace: bool
    cat: _Cat

    def __init__(
        self,
        in_keys_and_shape: dict[thd.NestedKey, Sequence[int]],
        missing_repr: Literal["nan", "mask"],
        values_key: thd.NestedKey = "features",
        mask_key: thd.NestedKey = "mask",
        dim: int = -1,
        unsqueeze_if_oor: bool = False,
        out_key: thd.NestedKey | None = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.in_keys_and_shape = {
            key: tuple(shape) for key, shape in in_keys_and_shape.items()
        }
        self.missing_repr = missing_repr
        self.values_key = values_key
        self.mask_key = mask_key
        self.out_keys = [] if out_key is None else [out_key]
        self.inplace = inplace
        self.cat = _Cat(dim=dim, unsqueeze_if_oor=unsqueeze_if_oor)

    def forward(self, data: thd.TensorDict) -> thd.TensorDict | th.Tensor:
        match self.missing_repr:
            case "nan":
                return self._forward_nan_mode(data)
            case "mask":
                return self._forward_mask_mode(data)
        raise ValueError(
            f"missing_repr has an unsupported literal: {self.missing_repr!r}"
        )

    def _block_widths(self, batch_rank: int) -> list[int]:
        """Compute the block width of each declared entry along the cat dim.

        Parameters
        ----------
        batch_rank : int
            Number of batch dimensions of the input TensorDict.

        Returns
        -------
        list of int
            One width per entry of ``in_keys_and_shape``, in the same order:
            1 for entries that ``unsqueeze_if_oor`` would unsqueeze along
            ``dim``, otherwise the trailing axis of the declared shape.
        """
        _widths: list[int] = []
        for _shape in self.in_keys_and_shape.values():
            _ndim: int = batch_rank + len(_shape)
            _unsqueeze: bool = self.cat.unsqueeze_if_oor and (
                self.cat.dim >= _ndim or self.cat.dim < -_ndim
            )
            _widths.append(1 if _unsqueeze else (_shape[-1] if _shape else 1))
        return _widths

    def _write_output(
        self, data: thd.TensorDict, out: th.Tensor
    ) -> thd.TensorDict | th.Tensor:
        """Write the concatenated output per the out_key/inplace configuration.

        Parameters
        ----------
        data : tensordict.TensorDict
            The forward's input TensorDict; written to when ``out_key`` is
            set and ``inplace`` is True.
        out : torch.Tensor
            The concatenated output tensor.

        Returns
        -------
        tensordict.TensorDict or torch.Tensor
            ``out`` itself when ``out_key`` is None; otherwise ``data`` with
            ``out`` written under ``out_key`` (``inplace=True``) or a new
            TensorDict holding only that entry (``inplace=False``).
        """
        if not self.out_keys:
            return out
        if self.inplace:
            data[self.out_keys[0]] = out
            return data
        tensordict_out = thd.TensorDict()
        tensordict_out[self.out_keys[0]] = out
        return tensordict_out

    def _forward_nan_mode(self, data: thd.TensorDict) -> thd.TensorDict | th.Tensor:
        """Forward for ``missing_repr="nan"``: per-key entries, NaN-detected.

        Absent keys contribute zero blocks to both the values and the
        indicator (using the declared shapes); present entries are
        zero-filled at NaN positions and flagged element-wise in the
        indicator.

        Parameters
        ----------
        data : tensordict.TensorDict
            Input TensorDict carrying some subset of the
            ``in_keys_and_shape`` keys.

        Returns
        -------
        tensordict.TensorDict or torch.Tensor
            The concatenated output of shape ``[..., 2 * D]`` (see
            ``_write_output``).
        """
        reference: th.Tensor | None = next(
            (data[key] for key in self.in_keys_and_shape if key in data), None
        )
        device = reference.device if reference is not None else data.device
        dtype = reference.dtype if reference is not None else th.float32
        values, indicators = [], []
        for _key, _shape in self.in_keys_and_shape.items():
            if _key not in data:
                values.append(
                    th.zeros((*data.batch_size, *_shape), device=device, dtype=dtype)
                )
                indicators.append(th.zeros_like(values[-1]))
                continue
            values.append(th.nan_to_num(data[_key], nan=0.0))
            indicators.append((~th.isnan(data[_key])).to(values[-1].dtype))
        out = th.cat((self.cat(*values), self.cat(*indicators)), dim=self.cat.dim)
        if not self.out_keys:
            return out
        if self.inplace:
            data[self.out_keys[0]] = out
            return data
        tensordict_out = thd.TensorDict()
        tensordict_out[self.out_keys[0]] = out
        return tensordict_out

    def _forward_mask_mode(self, data: thd.TensorDict) -> thd.TensorDict | th.Tensor:
        """Forward for ``missing_repr="mask"``: dense values plus an explicit mask.

        Reads the already zero-filled dense values under ``values_key`` and
        the observed-feature mask under ``mask_key``, and concatenates them
        — one dtype cast and one concatenation, with no per-key temporaries.

        Parameters
        ----------
        data : tensordict.TensorDict
            Input TensorDict carrying the ``values_key`` and ``mask_key``
            entries, both with trailing width equal to the total declared
            block width.

        Returns
        -------
        tensordict.TensorDict or torch.Tensor
            The concatenated output of shape ``[..., 2 * D]`` (see
            ``_write_output``).

        Raises
        ------
        ValueError
            If either entry's trailing width differs from the total declared
            block width.
        """
        # (..., n_feats); dense values, already zero-filled by the caller,
        # and the bool/0-1 observed-feature mask
        values: th.Tensor = data[self.values_key]
        masks: th.Tensor = data[self.mask_key]
        n_feats: int = sum(self._block_widths(len(data.batch_size)))
        if values.shape[-1] != n_feats or masks.shape[-1] != n_feats:
            raise ValueError(
                f"mask-mode entries must have trailing width {n_feats} "
                f"(from in_keys_and_shape), got {values.shape[-1]} for "
                f"`{self.values_key}` and {masks.shape[-1]} for `{self.mask_key}`"
            )
        # (..., 2 * n_feats); one cast + one concatenation
        out: th.Tensor = th.cat((values, masks.to(dtype=values.dtype)), dim=self.cat.dim)
        return self._write_output(data, out)


class _Cat(th.nn.Module):
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
    >>> cat = _Cat(dim=-1, unsqueeze_if_oor=False)
    >>> cat(th.randn(5, 3), th.randn(5, 2)).shape
    torch.Size([5, 5])

    Mix a per-row scalar tensor of shape ``[N]`` with a 2-D tensor by
    unsqueezing the former to ``[N, 1]`` (``dim=1`` is out of range for a
    1-D tensor):

    >>> cat = _Cat(dim=1, unsqueeze_if_oor=True)
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
