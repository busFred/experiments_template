# Repository instructions

This repository is a template for Python experiments.

## Required conventions

### Package import
Before creating files, Python packages, experiment directories, datasets, models, libraries, or third-party integrations, read [this document](docs/coding/package-import.md) for general naming guidlines.

### Docstrings and comments
I adopt numpy style docstring. Every function, including ones that are protected, needs a complete docstring, i.e., description of the function or class, input arguments, and return types. It's fine for a one-time use or api internal helper function to have a very short one sentence description, but a docstring that contains only the description but without complete I/O description is unacceptable. Function with no return (i.e., function that returns only none) does not need the return block, e.g.,
```python
def good_no_return_func(input_arg: Any) -> None:
    """A function that has no return

    Parameters
    -------
    input_arg: Any
        ...
    """
    ...

def bad_no_return_func(input_arg: Any) -> None:
    """A function that has no return

    Parameters
    -------
    input_arg: Any
        ...
    
    Returns
    -------
    None
    """
    ...
```

If a function has dict-like object(s) (e.g, thd.TensorDict) as either input or output, make sure to document all of the key in the following style:
```python
def function_returning_dict_like_object(input_d: thd.TensorDict) -> dict[str, th.Tensor]:
    """
    Build a batch.
    
    Parameters
    -------
    input_d: thd.TensorDict
        Batch dictionary with keys:
        - `"feature"` (`th.Tensor`): covariates, shape ``(n, n_covs)``
        - `"label"`(`th.Tensor`): targets, shape ``(n,)``

    Returns
    -------
    dict[str, torch.Tensor]
        Batch dictionary with keys:
        - `"feature"` (`th.Tensor`): covariates, shape ``(n, n_covs)``
        - `"label"`(`th.Tensor`): targets, shape ``(n,)``
    """
```

When writing interactive python (ipython) script, the docstring for the file and the first import block should not be separated; also, avoid using markdown cells:
```python
# %%
from __future__ import annotations

import concurrent.futures as conc_futures

# %%
a_list = list()
```

Comments should always exist in a line prior not at the end of line, e.g.,:
```python
# goood
# max summed line-graph edges per batch (~3.3 GiB peak GAT memory per 8M on a 12 GB card)
LG_EDGE_BUDGET: int = 8_000_000
# bad
BATCH_MAX_COUNT: int = 2048  # max instances per batch
```

### Naming
When writing a class, all instance attributes should be listed with type decoration in a `cpp` style before constructor; for insatnce:
```python
class OneClassDefinition:
    a: str
    b: float
    
    def __init__(self, a: str, b: float) -> None:
        self.a = a
        self.b = b
```

When dealing with variables, python has function scope instead of block scope unlike `cpp` or `java`. To improve readability and unintentional access, prefix vairables intended only for local blocks with `_`, e.g., `_idx`, `_item`, `_local_var`.
```python
def fucn():
    var1: int = [1, 2, 3, 4, 5]
    for _idx, _item in enumerate(var1):
        _local_var: int = _item + 279
```

If a tensor has more than one instance, then the correspondign variable should be named in plural. in addition, if it is a batch of data, then there should have a `b` in front of the variable indicating it comes from a batch. For instance:
```python
def innocent_function(
    data: thd.TensorDict,  # (n_instances, )
    bsz: int,
) -> thd.TensorDict:
    xs: th.Tensor = data["x"]
    ys: th.tensor = data["y"]
    for _bidxs in th.split(th.arange(len(data)), bsz):
        _bxs: th.Tensor = xs[_bidxs]
        _bys: th.Tensor = ys[_bidxs]
        # say something requires nested loop over batched data
        for _bbidxs in th.split(th.arange(len(_bxs)), bsz//2):
            _bbxs: th.Tensor = _bxs[_bbidxs]
        # if looping over instances
        for _x in _bxs:
            pass
    return xs, ys
```
However, similar rules does not apply to keys of `dict`-like object, e.g.,:
```python
xs, ys = innocent_function(data=raw_data, bsz=bsz)
a_dict = {"x": xs, "y": ys}
```

### Indexing
When adding (a) new axis in `th.Tensor` or anything similar (e.g., `np.ndarray`, `cp.ndarray`), use `a_tensor[:, None]` to add an axis instead of `a_tensor.unsqueeze(1)`. The square bracket approach makes interpreting `ndim` of a tensor much easier.

### Use of path
Explore `mylib` for generic utilities and `mydatasets` for data loaders; in particular, `mylib.get_project_root_dir()` and `mydatasets.get_datasets_files_root_dir()` should be used to access the project and the dataset root directory, respectively, in python scripts. Use `os.path` instead of `pathlib.Path`.

### Machine Learning
* Use `torch` primarily instead of `numpy` unless absolutely necessary such as using `scikit-learn`; most of the `numpy` operations have a `torch` counterpart.
* Use `plf = pl.Fabric(accelerator="auto")` (or whatever accelerator that is suitable for the task) and `plf.device` to handle devices instead of handling device manually with `device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')`.
* Use `th.min`, `th.max`, `th.mean`, `th.argmax`, `th.sum`, etc., instead of `tensor.min()`; this resembles how mathematical operator were written in pen and paper and is easier to understand.
* Annotate the shape of tensors in comment; if multiple tensors share the same shape, annotate them once in the beginning. This is especially important for developers to keep track of shapes and avoid runtime surprises; for inline comment, shape annotation should be in its single line:
    ```python
    # (bsz, n_covs)
    # description
    features: th.Tensor = ...
    ```

## Response style

Follow the below guidelines when responding in chat:
* Be direct and concise.
* Prefer bullets over paragraphs.
* Use tables when they improve clarity or comparison.
* Keep unavoidable paragraphs brief.
* Do not restate the request or add unnecessary background.
* Include only information needed to act or decide.
* Optimize for low cognitive load and minimal context-window usage.
