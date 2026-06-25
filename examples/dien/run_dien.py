"""DIEN 训练示例脚本（PyTorch Lightning）。"""

import sys
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cores.datamodules.din import DINDataModule
from cores.models.dien import DIEN


def main():
    # 与 DIN 示例共用同一份 demo 数据，便于公平对比
    data_path = Path(__file__).resolve().parents[1] / "DIN" / "data" / "din_demo.csv"
    max_seq_len = 20
    batch_size = 256
    max_epochs = 50
    embed_dim = 16
    hidden_dim = 16
    aux_loss_weight = 0.1
    evolve_attn_mode = "din"  # "din" | "dien"
    lr = 1e-3

    datamodule = DINDataModule(
        data_path=data_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )
    datamodule.setup()

    stats = datamodule.feature_stats
    model = DIEN(
        num_users=stats["num_users"],
        num_items=stats["num_items"],
        num_categories=stats["num_categories"],
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        aux_loss_weight=aux_loss_weight,
        evolve_attn_mode=evolve_attn_mode,
        lr=lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        filename="dien-{epoch:02d}-{val_auc:.4f}",
        save_top_k=1,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
    )

    logger = TensorBoardLogger(save_dir="lightning_logs", name="dien")

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
    print(f"演化层 attention: {evolve_attn_mode}")
    print("-" * 60)

    trainer.fit(model, datamodule=datamodule)

    print("-" * 60)
    print(f"DIEN 训练完成。最优 checkpoint: {checkpoint_callback.best_model_path}")
    log_dir = Path(logger.save_dir).resolve()
    print(f"查看 loss 曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
