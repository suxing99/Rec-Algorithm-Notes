"""DIN 训练示例脚本（PyTorch Lightning）。"""

import sys
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cores.models.din import DIN
from cores.datamodules.din import DINDataModule


def main():
    data_path = Path(__file__).parent / "data" / "din_demo.csv"
    max_seq_len = 20
    batch_size = 256
    max_epochs = 20
    embed_dim = 16
    lr = 1e-3

    datamodule = DINDataModule(
        data_path=data_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )
    datamodule.setup()

    stats = datamodule.feature_stats
    model = DIN(
        num_users=stats["num_users"],
        num_items=stats["num_items"],
        num_categories=stats["num_categories"],
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        lr=lr,
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
        patience=3,
    )

    logger = TensorBoardLogger(save_dir="lightning_logs", name="din")

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
    print("-" * 60)

    trainer.fit(model, datamodule=datamodule)

    print("-" * 60)
    print(f"DIN 训练完成。最优 checkpoint: {checkpoint_callback.best_model_path}")
    log_dir = Path(logger.save_dir).resolve()
    print(f"查看 loss 曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
