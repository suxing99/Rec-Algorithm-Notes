"""DIN 训练示例脚本（PyTorch Lightning）。

修改同目录下的 din.yaml 后执行:
    python examples/DIN/run_din.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "din.yaml"
sys.path.insert(0, str(ROOT))

from cores.datamodules.din import DINDataModule
from cores.models.din import DIN


def _resolve_path(path: str | None, base: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(config_path: Path = CONFIG_PATH) -> SimpleNamespace:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data = raw.get("data", {})
    model = raw.get("model", {})
    train = raw.get("train", {})
    trainer = raw.get("trainer", {})
    logging_cfg = raw.get("logging", {})

    data_path = _resolve_path(data.get("data_path"), ROOT)
    if data_path is None:
        raise ValueError("配置 data.data_path 为必填项")

    mlp_dims = model.get("mlp_dims", [128, 64])
    if not isinstance(mlp_dims, (list, tuple)) or not mlp_dims:
        raise ValueError("配置 model.mlp_dims 必须为非空列表，如 [128, 64]")

    return SimpleNamespace(
        data_path=data_path,
        val_ratio=float(data.get("val_ratio", 0.2)),
        seed=int(data.get("seed", 42)),
        embed_dim=int(model.get("embed_dim", 16)),
        max_seq_len=int(model.get("max_seq_len", 20)),
        mlp_dims=tuple(int(d) for d in mlp_dims),
        dropout=float(model.get("dropout", 0.2)),
        lr=float(model.get("lr", 1e-3)),
        batch_size=int(train.get("batch_size", 256)),
        max_epochs=int(train.get("max_epochs", 20)),
        num_workers=int(train.get("num_workers", 0)),
        log_every_n_steps=int(trainer.get("log_every_n_steps", 10)),
        early_stop_patience=int(trainer.get("early_stop_patience", 3)),
        log_save_dir=str(logging_cfg.get("save_dir", "lightning_logs")),
        log_name=str(logging_cfg.get("name", "din")),
        config_path=config_path.resolve(),
    )


def main() -> None:
    cfg = load_config()
    print(f"DIN 训练启动，配置: {cfg.config_path}")

    datamodule = DINDataModule(
        data_path=cfg.data_path,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    datamodule.setup()

    stats = datamodule.feature_stats
    model = DIN(
        num_users=stats["num_users"],
        num_items=stats["num_items"],
        num_categories=stats["num_categories"],
        embed_dim=cfg.embed_dim,
        max_seq_len=cfg.max_seq_len,
        mlp_dims=cfg.mlp_dims,
        dropout=cfg.dropout,
        lr=cfg.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        filename="din-{epoch:02d}-{val_auc:.4f}",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=cfg.early_stop_patience,
    )

    logger = TensorBoardLogger(save_dir=cfg.log_save_dir, name=cfg.log_name)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.log_every_n_steps,
    )

    print(f"数据路径: {cfg.data_path}")
    print(f"训练样本: {len(datamodule.train_dataset)}, 验证样本: {len(datamodule.val_dataset)}")
    print(
        f"用户数: {stats['num_users']}, "
        f"物品数: {stats['num_items']}, "
        f"类目数: {stats['num_categories']}"
    )
    print("-" * 60)

    trainer.fit(model, datamodule=datamodule)

    print("-" * 60)
    print(f"DIN 训练完成。最优 checkpoint: {checkpoint_callback.best_model_path}")
    log_dir = Path(logger.save_dir).resolve()
    print(f"查看 loss 曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
