"""OneTrans DataModule：精排 parquet 特征加载与预处理。"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import lightning.pytorch as pl
import numpy as np
from datasets import Dataset, Value, load_dataset
from torch.utils.data import DataLoader

from cores.datamodules.two_tower import (
    load_parquet_source,
    load_parquet_sources,
    log_stage,
    resolve_date_partition_dirs,
)
from cores.models.onetrans import (
    CANDIDATE_GAME_FIELD,
    NS_CAT_FIELDS,
    NS_MULTI_CAT_FIELDS,
    NUM_FEAT_FIELDS,
    SEQ_FIELDS,
)

def _normalize_game_id_column(ds: Dataset) -> Dataset:
    """兼容 gameid / game_id 两种列名。"""
    if CANDIDATE_GAME_FIELD in ds.column_names:
        return ds
    if "gameid" in ds.column_names:
        return ds.rename_column("gameid", CANDIDATE_GAME_FIELD)
    raise ValueError(f"数据缺少候选游戏 ID 列，需要 `{CANDIDATE_GAME_FIELD}` 或 `gameid`")


GameSequence = Union[str, Sequence[int], np.ndarray]


def parse_int_sequence(seq: GameSequence) -> List[int]:
    if isinstance(seq, (list, tuple, np.ndarray)):
        return [int(x) for x in seq]
    return [int(x) for x in str(seq).split("|") if x]


def pad_int_sequence(seq: GameSequence, max_len: int) -> List[int]:
    history = [int(x) for x in parse_int_sequence(seq) if int(x) != 0][-max_len:]
    padded = [0] * max_len
    for i, val in enumerate(history):
        padded[i] = val
    return padded


def pad_float_array(arr: Sequence[float], dim: int) -> List[float]:
    values = [float(x) for x in arr]
    if len(values) < dim:
        values = values + [0.0] * (dim - len(values))
    return values[:dim]


def _scalar_max(table, name: str) -> int:
    import pyarrow.compute as pc

    col = table.column(name)
    if len(col) == 0:
        return 0
    value = pc.max(col)
    return int(value.as_py()) if value.is_valid else 0


def _list_max(table, name: str) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc

    col = table.column(name)
    if len(col) == 0:
        return 0
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        flat = pc.list_flatten(col)
        if len(flat) == 0:
            return 0
        value = pc.max(flat)
        return int(value.as_py()) if value.is_valid else 0
    return max((int(x) for seq in col.to_pylist() for x in seq), default=0)


def _array_dim(table, name: str) -> int:
    col = table.column(name)
    if len(col) == 0:
        return 0
    first = col[0].as_py()
    return len(first) if first is not None else 0


def get_feature_stats(train_ds: Dataset, val_ds: Dataset) -> Dict[str, int]:
    """从训练/验证集统计词表与连续特征维度。"""
    tables = [train_ds.data.table, val_ds.data.table]
    stats: Dict[str, int] = {}

    for field in NS_CAT_FIELDS:
        stats[f"num_{field}"] = max(_scalar_max(t, field) for t in tables)

    for field in NS_MULTI_CAT_FIELDS:
        stats[f"num_{field}"] = max(_list_max(t, field) for t in tables)

    game_max = 0
    recharge_max = 0
    for t in tables:
        if CANDIDATE_GAME_FIELD in t.column_names:
            game_max = max(game_max, _scalar_max(t, CANDIDATE_GAME_FIELD))
        if "channel_game" in t.column_names:
            game_max = max(game_max, _scalar_max(t, "channel_game"))
        for field, seq_type in SEQ_FIELDS:
            col_max = _list_max(t, field)
            if seq_type == "game":
                game_max = max(game_max, col_max)
            else:
                recharge_max = max(recharge_max, col_max)
    stats["num_games"] = game_max
    stats["num_recharge_types"] = recharge_max

    stats["user_num_dim"] = max(_array_dim(t, "user_num_feats") for t in tables)
    stats["num_feats_dim"] = max(_array_dim(t, "num_feats") for t in tables)
    stats["cross_num_dim"] = max(_array_dim(t, "cross_num_feats") for t in tables)
    return stats


def _preprocess_example(example: Dict, max_seq_len: int, feature_dims: Dict[str, int]) -> Dict:
    result: Dict = {
        "recharge_30d": float(example["recharge_30d"]),
        "versionv": example["versionv"],
    }
    result[CANDIDATE_GAME_FIELD] = int(example[CANDIDATE_GAME_FIELD])

    for field in NS_CAT_FIELDS:
        result[field] = int(example[field])

    for field in NS_MULTI_CAT_FIELDS:
        seq = parse_int_sequence(example[field])
        padded = [0] * 10
        for i, val in enumerate(seq[:10]):
            padded[i] = val
        result[field] = padded

    for field, _ in SEQ_FIELDS:
        result[field] = pad_int_sequence(example[field], max_seq_len)

    for field in NUM_FEAT_FIELDS:
        dim_key = {
            "user_num_feats": "user_num_dim",
            "num_feats": "num_feats_dim",
            "cross_num_feats": "cross_num_dim",
        }[field]
        result[field] = pad_float_array(example[field], feature_dims[dim_key])

    return result


def preprocess_dataset(
    ds: Dataset,
    max_seq_len: int,
    feature_dims: Dict[str, int],
    num_proc: int = 1,
    label: str = "",
) -> Dataset:
    prefix = f"{label} " if label else ""
    map_kwargs = {
        "function": _preprocess_example,
        "fn_kwargs": {"max_seq_len": max_seq_len, "feature_dims": feature_dims},
        "desc": f"Preprocessing OneTrans dataset {prefix}".strip(),
        "load_from_cache_file": True,
    }
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    log_stage(f"  预处理{prefix}({len(ds)} 条, num_proc={num_proc})...")
    started = time.perf_counter()
    ds = ds.map(**map_kwargs)
    ds = ds.cast_column("recharge_30d", Value("float32"))
    elapsed = time.perf_counter() - started
    log_stage(f"  预处理{prefix}完成 ({elapsed:.1f}s)")
    return ds


class OneTransDataModule(pl.LightningDataModule):
    """OneTrans 训练/验证 DataModule。"""

    def __init__(
        self,
        train_path: Optional[Path] = None,
        val_path: Optional[Path] = None,
        data_root: Optional[Path] = None,
        train_end_date: Optional[str] = None,
        train_days: Optional[int] = None,
        train_subdir: str = "rank/train",
        val_end_date: Optional[str] = None,
        val_days: Optional[int] = None,
        val_subdir: str = "rank/val",
        max_seq_len: int = 30,
        batch_size: int = 256,
        num_workers: int = 0,
        preprocess_num_proc: Optional[int] = None,
    ):
        super().__init__()
        self.train_path = Path(train_path) if train_path is not None else None
        self.val_path = Path(val_path) if val_path is not None else None
        self.data_root = Path(data_root) if data_root is not None else None
        self.train_end_date = train_end_date
        self.train_days = train_days
        self.train_subdir = train_subdir
        self.val_end_date = val_end_date
        self.val_days = val_days
        self.val_subdir = val_subdir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_num_proc = (
            preprocess_num_proc if preprocess_num_proc is not None else max(1, num_workers)
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.feature_stats: Dict[str, int] = {}

    @property
    def _columns(self) -> List[str]:
        cols = list(NS_CAT_FIELDS)
        cols.extend(NS_MULTI_CAT_FIELDS)
        cols.extend(field for field, _ in SEQ_FIELDS)
        cols.extend(NUM_FEAT_FIELDS)
        cols.append(CANDIDATE_GAME_FIELD)
        cols.append("recharge_30d")
        cols.append("versionv")
        return cols

    def _load_train_dataset(self) -> Dataset:
        if self.train_end_date is not None and self.train_days is not None:
            if self.data_root is None:
                raise ValueError("使用 train_end_date/train_days 时需配置 data_root")
            train_dirs = resolve_date_partition_dirs(
                self.data_root,
                self.train_end_date,
                self.train_days,
                self.train_subdir,
            )
            log_stage(
                f"      按日期加载训练集: 截止 {self.train_end_date}, "
                f"共 {self.train_days} 天, subdir={self.train_subdir}"
            )
            for train_dir in train_dirs:
                log_stage(f"        - {train_dir}")
            return _normalize_game_id_column(load_parquet_sources(train_dirs))

        if self.train_path is None:
            raise ValueError(
                "需配置 train_path，或 (data_root + train_end_date + train_days)"
            )
        return _normalize_game_id_column(load_parquet_source(self.train_path))

    def _load_val_dataset(self) -> Dataset:
        if self.val_end_date is not None and self.val_days is not None:
            if self.data_root is None:
                raise ValueError("使用 val_end_date/val_days 时需配置 data_root")
            val_dirs = resolve_date_partition_dirs(
                self.data_root,
                self.val_end_date,
                self.val_days,
                self.val_subdir,
            )
            log_stage(
                f"      按日期加载验证集: 截止 {self.val_end_date}, "
                f"共 {self.val_days} 天, subdir={self.val_subdir}"
            )
            for val_dir in val_dirs:
                log_stage(f"        - {val_dir}")
            return _normalize_game_id_column(load_parquet_sources(val_dirs))

        if self.val_path is None:
            raise ValueError(
                "需配置 val_path，或 (data_root + val_end_date + val_days)"
            )
        return _normalize_game_id_column(load_parquet_source(self.val_path))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        setup_started = time.perf_counter()

        log_stage("[1/4] 加载训练集 parquet ...")
        train_ds = self._load_train_dataset()
        log_stage(f"      训练集: {len(train_ds)} 条")

        log_stage("[2/4] 加载验证集 parquet ...")
        val_ds = self._load_val_dataset()
        log_stage(f"      验证集: {len(val_ds)} 条")

        log_stage("[3/4] 统计特征词表 ...")
        self.feature_stats = get_feature_stats(train_ds, val_ds)
        log_stage(f"      词表: {self.feature_stats}")

        feature_dims = {
            "user_num_dim": self.feature_stats["user_num_dim"],
            "num_feats_dim": self.feature_stats["num_feats_dim"],
            "cross_num_dim": self.feature_stats["cross_num_dim"],
        }

        log_stage(f"[4/4] 预处理数据集 (max_seq_len={self.max_seq_len}) ...")
        train_ds = preprocess_dataset(
            train_ds,
            max_seq_len=self.max_seq_len,
            feature_dims=feature_dims,
            num_proc=self.preprocess_num_proc,
            label="(训练集)",
        )
        val_ds = preprocess_dataset(
            val_ds,
            max_seq_len=self.max_seq_len,
            feature_dims=feature_dims,
            num_proc=self.preprocess_num_proc,
            label="(验证集)",
        )

        self.train_dataset = train_ds.with_format("torch", columns=self._columns)
        self.val_dataset = val_ds.with_format("torch", columns=self._columns)

        elapsed = time.perf_counter() - setup_started
        log_stage(f"数据准备完成 ({elapsed:.1f}s)")

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
