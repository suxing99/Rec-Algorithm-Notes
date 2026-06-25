"""在同一数据上对比 DIN vs DIEN，排查 DIEN 效果偏差。"""

import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cores.datamodules.din import DINDataModule
from cores.models.dien import DIEN
from cores.models.din import DIN

SEED = 42
DATA_PATH = ROOT / "examples" / "DIN" / "data" / "din_demo.csv"


def train_model(model, datamodule, name: str, max_epochs: int = 20) -> float:
    early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=5)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[early_stop],
    )
    pl.seed_everything(SEED, workers=True)
    trainer.fit(model, datamodule=datamodule)
    val_auc = trainer.callback_metrics.get("val_auc", torch.tensor(0.0))
    return float(val_auc)


def main():
    configs = [
        ("DIN", lambda s: DIN(
            num_users=s["num_users"], num_items=s["num_items"],
            num_categories=s["num_categories"], embed_dim=16, max_seq_len=20, lr=1e-3,
        )),
        ("DIEN (aux=0.1)", lambda s: DIEN(
            num_users=s["num_users"], num_items=s["num_items"],
            num_categories=s["num_categories"], embed_dim=16, hidden_dim=16,
            max_seq_len=20, aux_loss_weight=0.1, lr=1e-3,
        )),
        ("DIEN (aux=0.0)", lambda s: DIEN(
            num_users=s["num_users"], num_items=s["num_items"],
            num_categories=s["num_categories"], embed_dim=16, hidden_dim=16,
            max_seq_len=20, aux_loss_weight=0.0, lr=1e-3,
        )),
    ]

    dm = DINDataModule(data_path=DATA_PATH, max_seq_len=20, batch_size=256, seed=SEED)
    dm.setup()
    stats = dm.feature_stats

    print(f"数据: {DATA_PATH}")
    print(f"训练/验证: {len(dm.train_dataset)}/{len(dm.val_dataset)}")
    print("-" * 50)

    results = []
    for name, factory in configs:
        model = factory(stats)
        auc = train_model(model, dm, name)
        results.append((name, auc))
        print(f"{name:20s}  val_auc = {auc:.4f}")

    print("-" * 50)
    best = max(results, key=lambda x: x[1])
    print(f"最优: {best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    main()
