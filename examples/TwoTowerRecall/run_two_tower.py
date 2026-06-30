"""双塔召回训练示例脚本（PyTorch Lightning）。

修改同目录下的 two_tower.yaml 后执行:
    python examples/TwoTowerRecall/run_two_tower.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "two_tower.yaml"
sys.path.insert(0, str(ROOT))

from cores.datamodules.two_tower import TwoTowerDataModule, log_stage
from cores.models.two_tower import PRIMARY_RECALL_K, RECALL_AT_K, TwoTowerRecall


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

    train_dir = _resolve_path(data.get("train_dir"), ROOT)
    val_dir = _resolve_path(data.get("val_dir"), ROOT)
    if train_dir is None or val_dir is None:
        raise ValueError("配置 data.train_dir 与 data.val_dir 为必填项")

    preprocess_num_proc = train.get("preprocess_num_proc")
    num_workers = int(train.get("num_workers", 0))

    return SimpleNamespace(
        train_dir=train_dir,
        val_dir=val_dir,
        embed_dim=int(model.get("embed_dim", 128)),
        output_dim=int(model.get("output_dim", 128)),
        max_seq_len=int(model.get("max_seq_len", 50)),
        lr=float(model.get("lr", 1e-3)),
        batch_size=int(train.get("batch_size", 512)),
        max_epochs=int(train.get("max_epochs", 20)),
        num_workers=num_workers,
        preprocess_num_proc=preprocess_num_proc if preprocess_num_proc is not None else None,
        log_every_n_steps=int(trainer.get("log_every_n_steps", 10)),
        early_stop_patience=int(trainer.get("early_stop_patience", 3)),
        log_save_dir=str(logging_cfg.get("save_dir", "lightning_logs")),
        log_name=str(logging_cfg.get("name", "two_tower")),
        config_path=config_path.resolve(),
    )


def main() -> None:
    cfg = load_config()
    log_stage("=" * 60)
    log_stage(f"双塔召回训练启动，配置: {cfg.config_path}")

    datamodule = TwoTowerDataModule(
        train_path=cfg.train_dir,
        val_path=cfg.val_dir,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        preprocess_num_proc=cfg.preprocess_num_proc,
    )
    datamodule.setup()

    stats = datamodule.feature_stats
    model = TwoTowerRecall(
        num_games=stats["num_games"],
        num_models=stats["num_models"],
        num_versions=stats["num_versions"],
        num_cities=stats["num_cities"],
        num_datatrace=stats["num_datatrace"],
        num_weeks=stats["num_weeks"],
        num_hours=stats["num_hours"],
        embed_dim=cfg.embed_dim,
        max_seq_len=cfg.max_seq_len,
        output_dim=cfg.output_dim,
        lr=cfg.lr,
    )

    primary_metric = f"val_recall@{PRIMARY_RECALL_K}"

    checkpoint_callback = ModelCheckpoint(
        monitor=primary_metric,
        mode="max",
        filename=f"two-tower-{{epoch:02d}}-{{val_recall@{PRIMARY_RECALL_K}:.4f}}",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor=primary_metric,
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

    print(f"训练目录: {datamodule.train_path}")
    print(f"验证目录: {datamodule.val_path}")
    print(f"训练样本: {len(datamodule.train_dataset)}, 验证样本: {len(datamodule.val_dataset)}")
    print(
        f"游戏数: {stats['num_games']}, "
        f"手机型号数: {stats['num_models']}, "
        f"版本数: {stats['num_versions']}, "
        f"城市数: {stats['num_cities']}, "
        f"资源位数: {stats['num_datatrace']}, "
        f"星期数: {stats['num_weeks']}, "
        f"小时数: {stats['num_hours']}"
    )
    print("-" * 60)

    trainer.fit(model, datamodule=datamodule)

    print("-" * 60)
    best_path = checkpoint_callback.best_model_path
    print(f"双塔召回训练完成。最优 checkpoint: {best_path}")

    if best_path:
        val_results = trainer.validate(
            model, datamodule=datamodule, ckpt_path=best_path, verbose=False
        )
        metrics = val_results[0] if val_results else {}
        print("最优模型验证集指标:")
        if "val_loss" in metrics:
            print(f"  val_loss: {metrics['val_loss']:.4f}")
        for k in RECALL_AT_K:
            key = f"val_recall@{k}"
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

    log_dir = Path(logger.save_dir).resolve()
    print(f"查看 loss 曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
