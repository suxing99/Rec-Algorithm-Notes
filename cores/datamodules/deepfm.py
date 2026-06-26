"""DeepFM DataModule（复用 Wide & Deep 的 demo 数据与连续特征处理）。"""

from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as pl
from datasets import Dataset, Value
from torch.utils.data import DataLoader

from cores.datamodules.wide_and_deep import (
    CONTINUOUS_FEATURES,
    compute_continuous_stats,
    get_feature_stats,
    load_or_create_demo_dataset,
    normalize_continuous,
)

__all__ = ["CONTINUOUS_FEATURES", "DeepFMDataModule"]


def preprocess_dataset(
    ds: Dataset,
    continuous_stats: Dict[str, Dict[str, float]],
) -> Dataset:
    def _map(example: Dict) -> Dict:
        return {
            "user_id": int(example["user_id"]),
            "item_id": int(example["item_id"]),
            "category_id": int(example["category_id"]),
            "continuous": normalize_continuous(example, continuous_stats),
            "label": float(example["label"]),
        }

    ds = ds.map(_map, desc="Preprocessing DeepFM dataset")
    return ds.cast_column("label", Value("float32"))


class DeepFMDataModule(pl.LightningDataModule):
    """DeepFM 训练/验证 DataModule。"""

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 256,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.feature_stats: Dict[str, int] = {}
        self.continuous_stats: Dict[str, Dict[str, float]] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        ds = load_or_create_demo_dataset(self.data_path)
        self.feature_stats = get_feature_stats(ds)

        split = ds.train_test_split(test_size=self.val_ratio, seed=self.seed, shuffle=True)
        train_raw = split["train"]
        val_raw = split["test"]

        self.continuous_stats = compute_continuous_stats(train_raw)

        self.train_dataset = preprocess_dataset(train_raw, self.continuous_stats)
        self.val_dataset = preprocess_dataset(val_raw, self.continuous_stats)

        columns = ["user_id", "item_id", "category_id", "continuous", "label"]
        self.train_dataset = self.train_dataset.with_format("torch", columns=columns)
        self.val_dataset = self.val_dataset.with_format("torch", columns=columns)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
