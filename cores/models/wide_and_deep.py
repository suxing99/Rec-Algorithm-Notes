"""Wide & Deep Learning for Recommender Systems 模型。"""

from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC

from cores.layers.activations import Dice


class WideAndDeep(pl.LightningModule):
    """
    Wide & Deep Learning for Recommender Systems.
    (https://arxiv.org/abs/1606.07792)

    Wide 分支（记忆）: 类别特征线性项 + user×item 交叉特征 + 连续特征线性项
    Deep 分支（泛化）: 类别 embedding 与连续特征拼接后过 MLP

    特征:
        - user_id, item_id, category_id: 类别特征
        - continuous: 连续数值特征，shape (batch, num_continuous)
        - cross_hash: user×item 交叉哈希桶，由 DataModule 预计算
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        num_continuous: int,
        cross_hash_size: int = 1_000_000,
        embed_dim: int = 8,
        deep_mlp_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["deep_mlp_dims"])
        self.num_continuous = num_continuous

        # Wide: 每个稀疏特征一个标量权重，等价于带偏置的线性模型
        self.user_wide = nn.Embedding(num_users + 1, 1, padding_idx=0)
        self.item_wide = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.category_wide = nn.Embedding(num_categories + 1, 1, padding_idx=0)
        self.cross_wide = nn.Embedding(cross_hash_size, 1, padding_idx=0)
        self.continuous_wide = nn.Linear(num_continuous, 1, bias=False)
        self.wide_bias = nn.Parameter(torch.zeros(1))

        # Deep: embedding + 连续特征 → MLP
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        deep_input_dim = embed_dim * 3 + num_continuous
        self.deep_mlp = self._build_mlp(deep_input_dim, deep_mlp_dims, dropout)

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
        continuous = batch["continuous"]
        cross_hash = batch["cross_hash"]

        wide_logit = (
            self.user_wide(user_ids).squeeze(-1)
            + self.item_wide(item_ids).squeeze(-1)
            + self.category_wide(category_ids).squeeze(-1)
            + self.cross_wide(cross_hash).squeeze(-1)
            + self.continuous_wide(continuous).squeeze(-1)
            + self.wide_bias
        )

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        category_emb = self.category_embedding(category_ids)
        deep_input = torch.cat([user_emb, item_emb, category_emb, continuous], dim=-1)
        deep_logit = self.deep_mlp(deep_input).squeeze(-1)

        logits = wide_logit + deep_logit
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
