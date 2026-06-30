"""导出游戏物品塔向量，供 TensorFlow Embedding Projector 可视化。

用法:
    python examples/TwoTowerRecall/export_item_vectors.py \\
        --checkpoint lightning_logs/two_tower/version_3/checkpoints/xxx.ckpt \\
        --train-dir /path/to/trainData \\
        --output-dir examples/TwoTowerRecall/projector

打开 https://projector.tensorflow.org/
  Step 1: 上传 vectors.tsv
  Step 2: 上传 metadata.tsv（含 gameName，可在右侧 Label by gameName）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_GAME_NAMES = DATA_DIR / "gameidx-gameName.tsv"
sys.path.insert(0, str(ROOT))

from cores.datamodules.two_tower import load_parquet_source, log_stage
from cores.models.two_tower import TwoTowerRecall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出游戏向量到 Embedding Projector 格式")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning checkpoint 路径")
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="训练数据目录（Spark part parquet），用于读取 gameidx 与 p_item",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "projector",
        help="输出 vectors.tsv 与 metadata.tsv 的目录",
    )
    parser.add_argument(
        "--game-names",
        type=Path,
        default=DEFAULT_GAME_NAMES,
        help="gameidx 与游戏名映射 TSV（默认 data/gameidx-gameName.tsv）",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_game_catalog(data_path: Path) -> pd.DataFrame:
    """从训练数据聚合每个游戏的 p_item（取最大值）。"""
    ds = load_parquet_source(data_path)
    df = ds.to_pandas()
    catalog = (
        df.groupby("gameidx", as_index=False)["p_item"]
        .max()
        .sort_values("gameidx")
        .reset_index(drop=True)
    )
    return catalog


def load_game_names(names_path: Path) -> pd.DataFrame:
    """加载 gameidx -> gameName 映射。"""
    if not names_path.exists():
        raise FileNotFoundError(f"游戏名文件不存在: {names_path}")
    names = pd.read_csv(names_path, sep="\t", dtype={"gameidx": np.int64, "gameName": str})
    names = names.drop_duplicates(subset="gameidx", keep="first")
    names["gameName"] = names["gameName"].fillna("").str.strip()
    return names


def build_projector_metadata(catalog: pd.DataFrame, names_path: Path) -> pd.DataFrame:
    """合并游戏名，生成 Projector 友好的 metadata（gameName 放首列便于 Label）。"""
    names = load_game_names(names_path)
    meta = catalog.merge(names, on="gameidx", how="left")
    missing = meta["gameName"].isna() | (meta["gameName"] == "")
    meta.loc[missing, "gameName"] = meta.loc[missing, "gameidx"].map(lambda x: f"unknown_{x}")
    meta["label"] = meta["gameidx"].astype(str) + ":" + meta["gameName"]
    meta["p_item"] = meta["p_item"].map(lambda x: f"{x:.6g}")
    return meta[["gameName", "gameidx", "p_item", "label"]]


@torch.no_grad()
def encode_all_items(
    model: TwoTowerRecall,
    gameidx: np.ndarray,
    p_item: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    model.to(device)
    vectors = []
    for start in range(0, len(gameidx), batch_size):
        end = start + batch_size
        batch = {
            "gameidx": torch.tensor(gameidx[start:end], dtype=torch.long, device=device),
            "p_item": torch.tensor(p_item[start:end], dtype=torch.float32, device=device),
        }
        vec = model.encode_item(batch)
        vectors.append(vec.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def write_projector_tsv(vectors: np.ndarray, metadata: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "vectors.tsv"
    metadata_path = output_dir / "metadata.tsv"

    np.savetxt(vectors_path, vectors, delimiter="\t", fmt="%.8f")

    metadata.to_csv(metadata_path, sep="\t", index=False)

    log_stage(f"已保存 {len(metadata)} 个游戏向量")
    log_stage(f"  vectors:  {vectors_path.resolve()}")
    log_stage(f"  metadata: {metadata_path.resolve()}")
    log_stage("打开 https://projector.tensorflow.org/ 上传上述两个文件")
    log_stage("  建议: Label by → gameName，Color by → p_item，搜索可用 gameidx 或 label")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    device = resolve_device(args.device)
    log_stage(f"加载 checkpoint: {args.checkpoint}")
    log_stage(f"设备: {device}")

    model = TwoTowerRecall.load_from_checkpoint(args.checkpoint, map_location=device)

    log_stage(f"读取游戏目录: {args.train_dir}")
    catalog = load_game_catalog(args.train_dir)
    log_stage(f"共 {len(catalog)} 个游戏")

    log_stage(f"合并游戏名: {args.game_names}")
    metadata = build_projector_metadata(catalog, args.game_names)
    matched = metadata["gameName"].str.startswith("unknown_").sum()
    if matched:
        log_stage(f"  {len(metadata) - matched} 个已匹配名称，{matched} 个无名称（unknown_<id>）")

    vectors = encode_all_items(
        model,
        catalog["gameidx"].to_numpy(),
        catalog["p_item"].to_numpy(dtype=np.float32),
        batch_size=args.batch_size,
        device=device,
    )
    write_projector_tsv(vectors, metadata, args.output_dir)


if __name__ == "__main__":
    main()
