"""DIN 数据集与 demo 数据生成（基于 Hugging Face datasets）。"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
from datasets import Dataset, Value, load_dataset
from torch.utils.data import DataLoader


def max_hist_item(seq: str) -> int:
    return max(int(x) for x in str(seq).split("|") if x)


def pad_history(seq: str, max_seq_len: int) -> List[int]:
    history = [int(x) for x in str(seq).split("|") if x]
    history = history[-max_seq_len:]
    padded = [0] * max_seq_len
    for i, item_id in enumerate(history):
        padded[i] = item_id
    return padded


def get_feature_stats(ds: Dataset) -> Dict[str, int]:
    hist_max = max(max_hist_item(seq) for seq in ds["hist_item_ids"])
    return {
        "num_users": int(max(ds["user_id"])),
        "num_items": int(max(max(ds["item_id"]), hist_max)),
        "num_categories": int(max(ds["category_id"])),
    }


def generate_demo_data(
    output_path: Path,
    num_samples: int = 5000,
    num_users: int = 500,
    num_items: int = 200,
    num_categories: int = 10,
    max_hist_len: int = 15,
    seed: int = 42,
) -> None:
    """
    生成带简单信号的 synthetic CTR 数据并写入 CSV。

    若候选 item 类目与用户最近点击类目一致，点击概率更高。
    """
    rng = np.random.default_rng(seed)
    item_categories = rng.integers(1, num_categories + 1, size=num_items + 1)
    item_categories[0] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "item_id", "category_id", "hist_item_ids", "label"],
        )
        writer.writeheader()

        for _ in range(num_samples):
            user_id = int(rng.integers(1, num_users + 1))
            item_id = int(rng.integers(1, num_items + 1))
            category_id = int(item_categories[item_id])

            hist_len = int(rng.integers(1, max_hist_len + 1))
            if rng.random() < 0.6:
                hist_items = rng.integers(1, num_items + 1, size=hist_len).tolist()
            else:
                same_cat_items = np.where(item_categories[1:] == category_id)[0] + 1
                if len(same_cat_items) == 0:
                    hist_items = rng.integers(1, num_items + 1, size=hist_len).tolist()
                else:
                    hist_items = rng.choice(same_cat_items, size=hist_len).tolist()

            hist_categories = [item_categories[i] for i in hist_items]
            match_ratio = np.mean(np.array(hist_categories) == category_id)
            click_prob = 0.15 + 0.55 * match_ratio
            label = int(rng.random() < click_prob)

            writer.writerow(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "category_id": category_id,
                    "hist_item_ids": "|".join(map(str, hist_items)),
                    "label": label,
                }
            )


def load_or_create_demo_dataset(data_path: Path) -> Dataset:
    if not data_path.exists():
        generate_demo_data(data_path)
    return load_dataset("csv", data_files=str(data_path), split="train")


def preprocess_dataset(ds: Dataset, max_seq_len: int) -> Dataset:
    ds = ds.map(
        lambda example: {
            "user_id": example["user_id"],
            "item_id": example["item_id"],
            "category_id": example["category_id"],
            "hist_item_ids": pad_history(example["hist_item_ids"], max_seq_len),
            "label": float(example["label"]),
        },
        desc="Preprocessing DIN dataset",
    )
    return ds.cast_column("label", Value("float32"))


class DINDataModule(pl.LightningDataModule):
    """DIN 训练/验证 DataModule。"""

    def __init__(
        self,
        data_path: Path,
        max_seq_len: int = 20,
        batch_size: int = 256,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.feature_stats: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        ds = load_or_create_demo_dataset(self.data_path)
        self.feature_stats = get_feature_stats(ds)
        ds = preprocess_dataset(ds, max_seq_len=self.max_seq_len)

        split = ds.train_test_split(test_size=self.val_ratio, seed=self.seed, shuffle=True)
        columns = ["user_id", "item_id", "category_id", "hist_item_ids", "label"]
        self.train_dataset = split["train"].with_format("torch", columns=columns)
        self.val_dataset = split["test"].with_format("torch", columns=columns)

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
