"""DeepFM 模型。"""

from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC

from cores.layers.activations import Dice
from cores.layers.fm import fm_second_order


class DeepFM(pl.LightningModule):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.
    (https://arxiv.org/abs/1703.04247)

    FM 分支: 一阶线性项 + 二阶特征交叉（共享 embedding）
    Deep 分支: 各域 embedding 与连续特征拼接后过 MLP

    特征:
        - user_id, item_id, category_id: 类别特征
        - continuous: 连续数值特征，shape (batch, num_continuous)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        num_continuous: int,
        embed_dim: int = 8,
        deep_mlp_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["deep_mlp_dims"])
        self.num_continuous = num_continuous

        # FM 一阶
        self.user_first = nn.Embedding(num_users + 1, 1, padding_idx=0)
        self.item_first = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.category_first = nn.Embedding(num_categories + 1, 1, padding_idx=0)
        self.continuous_first = nn.Linear(num_continuous, 1, bias=False)
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # FM 二阶与 Deep 共享 embedding
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

    def _field_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeds = torch.stack(
            [
                self.user_embedding(batch["user_id"]),
                self.item_embedding(batch["item_id"]),
                self.category_embedding(batch["category_id"]),
            ],
            dim=1,
        )
        return embeds

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        continuous = batch["continuous"]

        fm_first = (
            self.fm_bias
            + self.user_first(batch["user_id"]).squeeze(-1)
            + self.item_first(batch["item_id"]).squeeze(-1)
            + self.category_first(batch["category_id"]).squeeze(-1)
            + self.continuous_first(continuous).squeeze(-1)
        )

        field_embeds = self._field_embeddings(batch)
        fm_second = fm_second_order(field_embeds)

        deep_input = torch.cat(
            [
                field_embeds[:, 0],
                field_embeds[:, 1],
                field_embeds[:, 2],
                continuous,
            ],
            dim=-1,
        )
        deep_logit = self.deep_mlp(deep_input).squeeze(-1)

        logits = fm_first + fm_second + deep_logit
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
