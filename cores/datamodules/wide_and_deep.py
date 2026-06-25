"""Wide & Deep DataModule 与 demo 数据生成（含连续数值特征）。"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
from datasets import Dataset, Value, load_dataset
from torch.utils.data import DataLoader

CONTINUOUS_FEATURES = ("price", "user_age", "item_popularity")


def cross_hash(user_id: int, item_id: int, hash_size: int) -> int:
    """user×item 交叉特征哈希桶（Wide 分支记忆能力）。"""
    return int((user_id * 2654435761 + item_id * 40503) % hash_size)


def get_feature_stats(ds: Dataset) -> Dict[str, int]:
    return {
        "num_users": int(max(ds["user_id"])),
        "num_items": int(max(ds["item_id"])),
        "num_categories": int(max(ds["category_id"])),
        "num_continuous": len(CONTINUOUS_FEATURES),
    }


def compute_continuous_stats(ds: Dataset) -> Dict[str, Dict[str, float]]:
    """计算连续特征的均值与标准差，用于 z-score 标准化。"""
    stats = {}
    for feat in CONTINUOUS_FEATURES:
        values = np.array(ds[feat], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-8:
            std = 1.0
        stats[feat] = {"mean": mean, "std": std}
    return stats


def normalize_continuous(
    example: Dict,
    stats: Dict[str, Dict[str, float]],
) -> List[float]:
    return [
        (float(example[feat]) - stats[feat]["mean"]) / stats[feat]["std"]
        for feat in CONTINUOUS_FEATURES
    ]


def generate_demo_data(
    output_path: Path,
    num_samples: int = 5000,
    num_users: int = 500,
    num_items: int = 200,
    num_categories: int = 10,
    seed: int = 42,
) -> None:
    """
    生成带类别特征与连续数值特征的 synthetic CTR 数据。

    连续特征:
        - price: 商品价格（对数正态分布）
        - user_age: 用户年龄
        - item_popularity: 商品历史点击率

    点击信号:
        - user×item 交叉亲和度
        - 价格与用户年龄匹配度
        - 商品热度
    """
    rng = np.random.default_rng(seed)
    item_categories = rng.integers(1, num_categories + 1, size=num_items + 1)
    item_categories[0] = 0
    item_prices = rng.lognormal(mean=3.0, sigma=0.6, size=num_items + 1)
    item_prices[0] = 0.0
    item_popularity = rng.beta(2, 5, size=num_items + 1)
    item_popularity[0] = 0.0

    user_ages = rng.integers(18, 61, size=num_users + 1)
    user_ages[0] = 0
    # 每个用户对若干 item 有更高亲和度（供 Wide 交叉特征记忆）
    user_affinity: Dict[int, set] = {}
    for user_id in range(1, num_users + 1):
        affinity_items = rng.choice(
            np.arange(1, num_items + 1),
            size=min(5, num_items),
            replace=False,
        )
        user_affinity[user_id] = set(affinity_items.tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "user_id",
            "item_id",
            "category_id",
            "price",
            "user_age",
            "item_popularity",
            "label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_samples):
            user_id = int(rng.integers(1, num_users + 1))
            item_id = int(rng.integers(1, num_items + 1))
            category_id = int(item_categories[item_id])
            price = float(item_prices[item_id])
            user_age = int(user_ages[user_id])
            popularity = float(item_popularity[item_id])

            affinity_bonus = 0.35 if item_id in user_affinity[user_id] else 0.0
            price_match = 0.2 * (1.0 - abs(price - 20.0) / 50.0)
            age_factor = 0.1 * (user_age - 18) / 42.0
            click_prob = 0.1 + affinity_bonus + price_match + 0.25 * popularity + age_factor
            click_prob = float(np.clip(click_prob, 0.05, 0.95))
            label = int(rng.random() < click_prob)

            writer.writerow(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "category_id": category_id,
                    "price": round(price, 4),
                    "user_age": user_age,
                    "item_popularity": round(popularity, 4),
                    "label": label,
                }
            )


def load_or_create_demo_dataset(data_path: Path) -> Dataset:
    if not data_path.exists():
        generate_demo_data(data_path)
    return load_dataset("csv", data_files=str(data_path), split="train")


def preprocess_dataset(
    ds: Dataset,
    continuous_stats: Dict[str, Dict[str, float]],
    cross_hash_size: int,
) -> Dataset:
    def _map(example: Dict) -> Dict:
        user_id = int(example["user_id"])
        item_id = int(example["item_id"])
        return {
            "user_id": user_id,
            "item_id": item_id,
            "category_id": int(example["category_id"]),
            "continuous": normalize_continuous(example, continuous_stats),
            "cross_hash": cross_hash(user_id, item_id, cross_hash_size),
            "label": float(example["label"]),
        }

    ds = ds.map(_map, desc="Preprocessing Wide & Deep dataset")
    return ds.cast_column("label", Value("float32"))


class WideAndDeepDataModule(pl.LightningDataModule):
    """Wide & Deep 训练/验证 DataModule。"""

    def __init__(
        self,
        data_path: Path,
        cross_hash_size: int = 1_000_000,
        batch_size: int = 256,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.cross_hash_size = cross_hash_size
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

        # 仅用训练集统计量做标准化，避免数据泄漏
        self.continuous_stats = compute_continuous_stats(train_raw)

        self.train_dataset = preprocess_dataset(
            train_raw, self.continuous_stats, self.cross_hash_size
        )
        self.val_dataset = preprocess_dataset(
            val_raw, self.continuous_stats, self.cross_hash_size
        )

        columns = ["user_id", "item_id", "category_id", "continuous", "cross_hash", "label"]
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
