"""Deep Interest Network (DIN) 模型。"""

from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC

from cores.layers.activations import Dice
from cores.layers.attention import LocalActivationUnit


class DIN(pl.LightningModule):
    """
    Deep Interest Network for Click-Through Rate Prediction.
    (https://arxiv.org/abs/1706.06978)

    特征:
        - user_id: 用户 ID
        - item_id: 候选 item ID（与历史行为共享 embedding）
        - category_id: 候选 item 类目
        - hist_item_ids: 用户历史点击 item 序列（0 为 padding）
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        embed_dim: int = 8,
        max_seq_len: int = 20,
        mlp_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["mlp_dims"])
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)

        self.attention = LocalActivationUnit(embed_dim)
        self.mlp = self._build_mlp(embed_dim * 4, mlp_dims, dropout)
        self.val_auc = BinaryAUROC()

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden),
                    Dice(hidden),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        category_ids = batch["category_id"]
        hist_item_ids = batch["hist_item_ids"]

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        category_emb = self.category_embedding(category_ids)

        hist_emb = self.item_embedding(hist_item_ids)
        mask = (hist_item_ids != 0).float()
        interest, _ = self.attention(item_emb, hist_emb, mask)

        features = torch.cat([user_emb, item_emb, category_emb, interest], dim=-1)
        logits = self.mlp(features).squeeze(-1)
        return torch.sigmoid(logits)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self(batch)
        loss = F.binary_cross_entropy(preds, batch["label"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds = self(batch)
        loss = F.binary_cross_entropy(preds, batch["label"])
        self.val_auc.update(preds, batch["label"].int())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.val_auc.reset()

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.hparams.lr)
