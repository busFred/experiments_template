# Package Import Abbreviations

Coding convention for `import ... as ...` aliases, as used in the skyscapes-playground repository.

## General rule

- The alias starts with a short abbreviation of the **top-level** package,
  formed by dropping vowels where needed (`torch` → `th`, `torchvision` → `thv`,
  `torchmetrics` → `thm`, `transformers` → `tfmrs`, `tensordict` → `thd`).
- Submodules are appended to that prefix with underscores:
  `torch.utils.data` → `th_data`, `torchvision.transforms.v2` → `thv_tfmsv2`.
- E.g., `sklearn.<submodule>` would be imported as `skl_<submodule>`.
- Well-established community aliases (`np`, `plt`) are kept as-is.

## Prefer dotting into the root module over `from ... import ...`

If a symbol is reachable through the root package, do **not** import it
directly — dot into it via the aliased root instead. Only import a specific
symbol or submodule when it is **not** accessible through the root module.

```python
# BAD
from torch import nn

class MyModel(nn.Module): ...

# GOOD
import torch as th

class MyModel(th.nn.Module): ...
```

```python
# BAD
from torch.utils.data import Dataset
from torchvision.transforms import v2

# GOOD
import torch.utils.data as th_data
import torchvision.transforms.v2 as thv_tfmsv2

class MyDataset(th_data.Dataset): ...
thv_tfmsv2.Compose([...])
```

## Table

| Import | Alias |
|---|---|
| `hydra` | `hd` |
| `omegaconf` | `omgcf` |
| `torch` | `th` |
| `torch.utils.data` | `th_data` |
| `torchvision` | `thv` |
| `torchvision.io` | `thvio` |
| `torchvision.transforms.v2` | `thv_tfmsv2` |
| `torchvision.tv_tensors` | `thv_tv_tensors` |
| `torchmetrics` | `thm` |
| `torchinfo` | `thinfo` |
| `tensordict` | `thd` |
| `lightning` | `pl` |
| `lightning.fabric.loggers` | `plf_loggers` |
| `kornia` | `kn` |
| `transformers` | `tfmrs` |
| `numpy` | `np` |
| `tqdm.auto` | `tqdm` |
| `sklearn.<submodule>` | `skl_<submodule>` |


## Example

```python
import lightning as pl
import lightning.fabric.loggers as plf_loggers
import matplotlib.pyplot as plt
import numpy as np
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchmetrics as thm
import torchvision.transforms.v2 as thv_tfmsv2
import tqdm.auto as tqdm
```
