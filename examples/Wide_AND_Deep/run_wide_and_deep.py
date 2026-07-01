"""Wide & Deep 训练示例脚本（PyTorch Lightning）。"""

import sys
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cores.datamodules.wide_and_deep import CONTINUOUS_FEATURES, WideAndDeepDataModule
from cores.models.wide_and_deep import WideAndDeep


def main():
    data_path = Path(__file__).parent / "data" / "wide_and_deep_demo.csv"
    cross_hash_size = 1_000_000
    batch_size = 256
    max_epochs = 30
    embed_dim = 16
    lr = 1e-3

    datamodule = WideAndDeepDataModule(
        data_path=data_path,
        cross_hash_size=cross_hash_size,
        batch_size=batch_size,
    )
    datamodule.setup()

    stats = datamodule.feature_stats
    model = WideAndDeep(
        num_users=stats["num_users"],
        num_items=stats["num_items"],
        num_categories=stats["num_categories"],
        num_continuous=stats["num_continuous"],
        cross_hash_size=cross_hash_size,
        embed_dim=embed_dim,
        lr=lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        filename="wide-and-deep-{epoch:02d}-{val_auc:.4f}",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
    )

    logger = TensorBoardLogger(save_dir="lightning_logs", name="wide_and_deep")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )

    print(f"数据路径: {data_path}")
    print(f"训练样本: {len(datamodule.train_dataset)}, 验证样本: {len(datamodule.val_dataset)}")
    print(
        f"用户数: {stats['num_users']}, "
        f"物品数: {stats['num_items']}, "
        f"类目数: {stats['num_categories']}"
    )
    print(f"连续特征: {', '.join(CONTINUOUS_FEATURES)}")
    print("-" * 60)

    trainer.fit(model, datamodule=datamodule)

    print("-" * 60)
    print(f"Wide & Deep 训练完成。最优 checkpoint: {checkpoint_callback.best_model_path}")
    log_dir = Path(logger.save_dir).resolve()
    print(f"查看 loss 曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
