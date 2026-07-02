"""双塔召回 DataModule。"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import lightning.pytorch as pl
import numpy as np
from datasets import Dataset, Value, load_dataset
from torch.utils.data import DataLoader

GameSequence = Union[str, Sequence[int]]


def log_stage(message: str, *, newline_before: bool = False) -> None:
    """打印阶段日志；newline_before=True 时先换行，避免与 tqdm 进度条粘在同一行。"""
    if newline_before:
        print(f"\n{message}", flush=True)
    else:
        print(message, flush=True)


def parse_game_sequence(seq: GameSequence) -> List[int]:
    """解析游戏历史序列，兼容 parquet 中的 list 与 CSV 中的 '|' 分隔字符串。"""
    if isinstance(seq, (list, tuple, np.ndarray)):
        return [int(x) for x in seq if int(x) != 0]
    return [int(x) for x in str(seq).split("|") if x]


def max_seq_game(seq: GameSequence) -> int:
    games = parse_game_sequence(seq)
    return max(games) if games else 0


def pad_game_sequence(seq: GameSequence, max_seq_len: int) -> List[int]:
    history = parse_game_sequence(seq)[-max_seq_len:]
    padded = [0] * max_seq_len
    for i, game_id in enumerate(history):
        padded[i] = game_id
    return padded


def _scalar_column_max(table, name: str) -> int:
    import pyarrow.compute as pc

    col = table.column(name)
    if len(col) == 0:
        return 0
    value = pc.max(col)
    return int(value.as_py()) if value.is_valid else 0


def _list_column_max(table, name: str) -> int:
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
    return max(max_seq_game(seq) for seq in col.to_pylist())


def _get_feature_stats_arrow(ds: Dataset) -> Dict[str, int]:
    table = ds.data.table
    seq_max = _list_column_max(table, "gameidx_seq")
    stats = {
        "num_games": int(
            max(
                _scalar_column_max(table, "gameidx"),
                _scalar_column_max(table, "last_gameidx"),
                seq_max,
            )
        ),
        "num_models": _scalar_column_max(table, "modelx"),
        "num_versions": _scalar_column_max(table, "versionx"),
        "num_cities": _scalar_column_max(table, "cityx"),
        "num_datatrace": _scalar_column_max(table, "datatraceidx"),
    }
    stats["num_weeks"] = _scalar_column_max(table, "week") if "week" in table.column_names else 0
    stats["num_hours"] = _scalar_column_max(table, "hour") if "hour" in table.column_names else 0
    return stats


def _get_feature_stats_slow(ds: Dataset) -> Dict[str, int]:
    seq_max = max(max_seq_game(seq) for seq in ds["gameidx_seq"])
    stats = {
        "num_games": int(max(max(ds["gameidx"]), max(ds["last_gameidx"]), seq_max)),
        "num_models": int(max(ds["modelx"])),
        "num_versions": int(max(ds["versionx"])),
        "num_cities": int(max(ds["cityx"])),
        "num_datatrace": int(max(ds["datatraceidx"])),
    }
    if "week" in ds.column_names:
        stats["num_weeks"] = int(max(ds["week"]))
    else:
        stats["num_weeks"] = 0
    if "hour" in ds.column_names:
        stats["num_hours"] = int(max(ds["hour"]))
    else:
        stats["num_hours"] = 0
    return stats


def get_feature_stats(ds: Dataset, label: str = "") -> Dict[str, int]:
    prefix = f"{label} " if label else ""
    log_stage(f"  统计特征词表{prefix}({len(ds)} 条)...")
    started = time.perf_counter()
    try:
        stats = _get_feature_stats_arrow(ds)
        method = "PyArrow"
    except Exception:
        stats = _get_feature_stats_slow(ds)
        method = "Python"
    elapsed = time.perf_counter() - started
    log_stage(f"  特征词表{prefix}完成 ({method}, {elapsed:.1f}s): {stats}")
    return stats


def merge_feature_stats(*stats_list: Dict[str, int]) -> Dict[str, int]:
    """合并多份 feature_stats，取各特征最大值（避免验证集 id 越界）。"""
    merged: Dict[str, int] = {}
    for stats in stats_list:
        for key, value in stats.items():
            merged[key] = max(merged.get(key, 0), value)
    return merged


def _list_parquet_files(directory: Path) -> List[Path]:
    """列出目录中的 parquet 数据文件（兼容 Spark part-*.snappy.parquet）。"""
    return sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.name.endswith(".parquet") and not f.name.startswith("_")
    )


def resolve_date_partition_dirs(
    data_root: Path,
    end_date: str,
    days: int,
    subdir: str = "rank/train",
) -> List[Path]:
    """根据截止日期与天数，解析按 datekey 分区的数据目录（新到旧）。"""
    if days < 1:
        raise ValueError(f"days 必须 >= 1, 当前: {days}")

    end = datetime.strptime(str(end_date), "%Y%m%d").date()
    dirs: List[Path] = []
    for offset in range(days):
        datekey = (end - timedelta(days=offset)).strftime("%Y%m%d")
        dir_path = Path(data_root) / datekey / subdir
        if not dir_path.is_dir():
            raise FileNotFoundError(f"训练数据目录不存在: {dir_path}")
        dirs.append(dir_path)
    return dirs


def load_parquet_sources(paths: Sequence[Path]) -> Dataset:
    """加载多个 parquet 文件或目录，合并为一个 Dataset。"""
    if not paths:
        raise ValueError("至少需要一个数据路径")

    all_files: List[str] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据路径不存在: {path}")
        if path.is_dir():
            parquet_files = _list_parquet_files(path)
            if not parquet_files:
                log_stage(f"  跳过无 parquet 文件的目录: {path}")
                continue
            all_files.extend(str(f) for f in parquet_files)
        elif path.name.endswith(".parquet"):
            all_files.append(str(path))
        else:
            raise ValueError(f"不支持的 parquet 路径: {path}")

    if not all_files:
        raise FileNotFoundError("未找到任何 parquet 文件")
    log_stage(f"  共加载 {len(all_files)} 个 parquet 文件")
    return load_dataset("parquet", data_files=all_files, split="train")


def load_parquet_source(path: Path) -> Dataset:
    """加载单个 parquet 文件，或 Spark 输出目录下的全部 part-*.parquet。"""
    return load_parquet_sources([Path(path)])


def _preprocess_example(example: Dict, max_seq_len: int, has_temporal: bool) -> Dict:
    result = {
        "gameidx_seq": pad_game_sequence(example["gameidx_seq"], max_seq_len),
        "last_gameidx": example["last_gameidx"],
        "gameidx": example["gameidx"],
        "modelx": example["modelx"],
        "versionx": example["versionx"],
        "p_item": float(example["p_item"]),
        "cityx": example["cityx"],
        "datatraceidx": example["datatraceidx"],
    }
    if has_temporal:
        result["week"] = example["week"]
        result["hour"] = example["hour"]
    else:
        result["week"] = 0
        result["hour"] = 0
    return result


def preprocess_dataset(
    ds: Dataset,
    max_seq_len: int,
    num_proc: int = 1,
    label: str = "",
) -> Dataset:
    has_temporal = "week" in ds.column_names and "hour" in ds.column_names
    prefix = f"{label} " if label else ""
    map_kwargs = {
        "function": _preprocess_example,
        "fn_kwargs": {"max_seq_len": max_seq_len, "has_temporal": has_temporal},
        "desc": f"Preprocessing Two-Tower dataset {prefix}".strip(),
        "load_from_cache_file": True,
    }
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    log_stage(f"  预处理{prefix}({len(ds)} 条, num_proc={num_proc})...")
    started = time.perf_counter()
    ds = ds.map(**map_kwargs)
    ds = ds.cast_column("p_item", Value("float32"))
    elapsed = time.perf_counter() - started
    log_stage(f"  预处理{prefix}完成 ({elapsed:.1f}s)")
    return ds


class TwoTowerDataModule(pl.LightningDataModule):
    """双塔召回训练/验证 DataModule（需预先划分好的训练集与验证集目录）。"""

    def __init__(
        self,
        train_path: Path,
        val_path: Path,
        max_seq_len: int = 20,
        batch_size: int = 256,
        num_workers: int = 0,
        preprocess_num_proc: Optional[int] = None,
    ):
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_num_proc = (
            preprocess_num_proc if preprocess_num_proc is not None else max(1, num_workers)
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.feature_stats: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        columns = [
            "gameidx_seq",
            "last_gameidx",
            "gameidx",
            "modelx",
            "versionx",
            "p_item",
            "cityx",
            "datatraceidx",
            "week",
            "hour",
        ]

        setup_started = time.perf_counter()
        stage_times: List[tuple[str, float]] = []

        stage_start = time.perf_counter()
        log_stage("[1/5] 加载训练集 parquet ...")
        train_ds = load_parquet_source(self.train_path)
        stage_times.append(("加载训练集", time.perf_counter() - stage_start))
        log_stage(f"      训练集: {len(train_ds)} 条 ({stage_times[-1][1]:.1f}s)")

        stage_start = time.perf_counter()
        log_stage("[2/5] 加载验证集 parquet ...")
        val_ds = load_parquet_source(self.val_path)
        stage_times.append(("加载验证集", time.perf_counter() - stage_start))
        log_stage(f"      验证集: {len(val_ds)} 条 ({stage_times[-1][1]:.1f}s)")

        stage_start = time.perf_counter()
        log_stage("[3/5] 统计特征词表 ...")
        self.feature_stats = merge_feature_stats(
            get_feature_stats(train_ds, label="(训练集)"),
            get_feature_stats(val_ds, label="(验证集)"),
        )
        stage_times.append(("统计特征词表", time.perf_counter() - stage_start))
        log_stage(f"      合并词表: {self.feature_stats} ({stage_times[-1][1]:.1f}s)")

        stage_start = time.perf_counter()
        log_stage(f"[4/5] 预处理训练集 (num_proc={self.preprocess_num_proc}) ...")
        train_ds = preprocess_dataset(
            train_ds,
            max_seq_len=self.max_seq_len,
            num_proc=self.preprocess_num_proc,
            label="(训练集)",
        )
        stage_times.append(("预处理训练集", time.perf_counter() - stage_start))

        stage_start = time.perf_counter()
        log_stage(f"[5/5] 预处理验证集 (num_proc={self.preprocess_num_proc}) ...")
        val_ds = preprocess_dataset(
            val_ds,
            max_seq_len=self.max_seq_len,
            num_proc=self.preprocess_num_proc,
            label="(验证集)",
        )
        stage_times.append(("预处理验证集", time.perf_counter() - stage_start))

        self.train_dataset = train_ds.with_format("torch", columns=columns)
        self.val_dataset = val_ds.with_format("torch", columns=columns)

        total_elapsed = time.perf_counter() - setup_started
        log_stage("数据准备完成，各阶段耗时:")
        for name, elapsed in stage_times:
            log_stage(f"  - {name}: {elapsed:.1f}s")
        log_stage(f"  合计: {total_elapsed:.1f}s")
        log_stage("开始构建模型。")

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
