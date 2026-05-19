from __future__ import annotations

import csv
import os
from typing import Literal

from . import common


def load_sample_dataset(dataset_type: Literal["train", "test", "val"]):
    dataset_p: str = os.path.join(
        common.get_datasets_files_root_dir(), "sample", f"{dataset_type}.csv"
    )
    dataset: list[dict[str, str]] = list()
    with open(dataset_p, "r") as f:
        reader = csv.DictReader(f, ["model", "plate"])
        dataset = [r for r in reader]
    return dataset
