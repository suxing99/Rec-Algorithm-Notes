"""OneTrans LTV30 训练示例脚本（PyTorch Lightning）。

修改同目录下的 onetrans.yaml 后执行:
    python examples/OneTrans/run_onetrans.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "onetrans.yaml"
sys.path.insert(0, str(ROOT))

from cores.datamodules.onetrans import OneTransDataModule, log_stage
from cores.models.onetrans import OneTrans


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

    train_path = _resolve_path(data.get("train_path"), ROOT)
    val_path = _resolve_path(data.get("val_path"), ROOT)
    if train_path is None or val_path is None:
        raise ValueError("配置 data.train_path 与 data.val_path 为必填项")

    return SimpleNamespace(
        train_path=train_path,
        val_path=val_path,
        embed_dim=int(model.get("embed_dim", 16)),
        d_model=int(model.get("d_model", 64)),
        num_heads=int(model.get("num_heads", 4)),
        num_layers=int(model.get("num_layers", 4)),
        ns_num_tokens=int(model.get("ns_num_tokens", 8)),
        max_seq_len=int(model.get("max_seq_len", 30)),
        ffn_ratio=float(model.get("ffn_ratio", 4.0)),
        dropout=float(model.get("dropout", 0.1)),
        min_s_keep=int(model.get("min_s_keep", 4)),
        target_log1p=bool(model.get("target_log1p", True)),
        lr=float(model.get("lr", 1e-3)),
        batch_size=int(train.get("batch_size", 128)),
        max_epochs=int(train.get("max_epochs", 30)),
        num_workers=int(train.get("num_workers", 0)),
        log_every_n_steps=int(trainer.get("log_every_n_steps", 10)),
        early_stop_patience=int(trainer.get("early_stop_patience", 5)),
        log_save_dir=str(logging_cfg.get("save_dir", "lightning_logs")),
        log_name=str(logging_cfg.get("name", "onetrans")),
        config_path=config_path.resolve(),
    )


def main() -> None:
    cfg = load_config()
    log_stage("=" * 60)
    log_stage(f"OneTrans LTV30 训练启动，配置: {cfg.config_path}")

    datamodule = OneTransDataModule(
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    datamodule.setup()

    model = OneTrans(
        feature_stats=datamodule.feature_stats,
        embed_dim=cfg.embed_dim,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ns_num_tokens=cfg.ns_num_tokens,
        max_seq_len=cfg.max_seq_len,
        ffn_ratio=cfg.ffn_ratio,
        dropout=cfg.dropout,
        min_s_keep=cfg.min_s_keep,
        target_log1p=cfg.target_log1p,
        lr=cfg.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        filename="onetrans-{epoch:02d}-{val_mae:.4f}",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_mae",
        mode="min",
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

    log_stage(f"训练样本: {len(datamodule.train_dataset)}, 验证样本: {len(datamodule.val_dataset)}")
    log_stage(f"特征词表: {datamodule.feature_stats}")
    log_stage("-" * 60)

    trainer.fit(model, datamodule=datamodule)

    log_stage("-" * 60)
    log_stage(f"OneTrans 训练完成。最优 checkpoint: {checkpoint_callback.best_model_path}")
    log_dir = Path(logger.save_dir).resolve()
    log_stage(f"查看 loss 曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
