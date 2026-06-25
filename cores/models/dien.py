"""Deep Interest Evolution Network (DIEN) 模型。"""

from typing import Any, Dict, Literal, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC

from cores.layers.activations import Dice
from cores.layers.attention import LocalActivationUnit
from cores.layers.gru import AUGRU

EvolveAttnMode = Literal["din", "dien"]


class DIEN(pl.LightningModule):
    """
    Deep Interest Evolution Network for Click-Through Rate Prediction.
    (https://arxiv.org/abs/1809.03672)

    特征:
        - user_id: 用户 ID
        - item_id: 候选 item ID（与历史行为共享 embedding）
        - category_id: 候选 item 类目
        - hist_item_ids: 用户历史点击 item 序列（0 为 padding）

    evolve_attn_mode:
        - "din": DIN 局部激活单元（LocalActivationUnit），默认
        - "dien": 论文原始双线性 attention + softmax
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        embed_dim: int = 8,
        hidden_dim: int = 8,
        max_seq_len: int = 20,
        mlp_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        aux_loss_weight: float = 0.1,
        evolve_attn_mode: EvolveAttnMode = "din",
        lr: float = 1e-3,
    ):
        super().__init__()
        if evolve_attn_mode not in ("din", "dien"):
            raise ValueError(f"evolve_attn_mode 必须是 'din' 或 'dien'，收到: {evolve_attn_mode}")

        self.save_hyperparameters(ignore=["mlp_dims"])
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.aux_loss_weight = aux_loss_weight
        self.evolve_attn_mode = evolve_attn_mode

        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)

        self.interest_extractor = nn.GRU(
            embed_dim, hidden_dim, batch_first=True
        )
        self.aux_proj = nn.Linear(hidden_dim, embed_dim)

        if evolve_attn_mode == "din":
            self.state_proj = nn.Linear(hidden_dim, embed_dim)
            self.local_attention = LocalActivationUnit(embed_dim)
        else:
            self.evolve_attention = nn.Linear(hidden_dim, embed_dim)

        self.interest_evolver = AUGRU(hidden_dim, hidden_dim)

        self.mlp = self._build_mlp(embed_dim * 3 + hidden_dim, mlp_dims, dropout)
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

    def _compute_evolve_attention(
        self,
        interest_states: torch.Tensor,
        ad_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算兴趣演化层 attention 权重，供 AUGRU 门控使用。"""
        if self.evolve_attn_mode == "din":
            return self._compute_evolve_attention_din(interest_states, ad_emb, mask)
        return self._compute_evolve_attention_dien(interest_states, ad_emb, mask)

    def _compute_evolve_attention_din(
        self,
        interest_states: torch.Tensor,
        ad_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """DIN 局部激活：MLP([ad, h, ad-h, ad*h]) + sigmoid。"""
        keys = self.state_proj(interest_states)
        _, weights = self.local_attention(ad_emb, keys, mask)
        return weights

    def _compute_evolve_attention_dien(
        self,
        interest_states: torch.Tensor,
        ad_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """论文原始方式：双线性打分 + softmax。"""
        att_proj = self.evolve_attention(interest_states)
        scores = (att_proj * ad_emb.unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(
            mask.bool(),
            weights,
            torch.zeros_like(weights),
        )
        return weights

    def _auxiliary_loss(
        self,
        interest_states: torch.Tensor,
        hist_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        用第 t+1 步真实点击行为监督第 t 步兴趣状态。
        """
        next_items = hist_item_ids[:, 1:]
        curr_states = interest_states[:, :-1, :]
        valid = (hist_item_ids[:, :-1] != 0) & (next_items != 0)

        if valid.sum() == 0:
            return interest_states.new_tensor(0.0)

        pos_emb = self.item_embedding(next_items.clamp(min=0))
        neg_items = torch.randint(
            1,
            self.hparams.num_items + 1,
            next_items.shape,
            device=hist_item_ids.device,
        )
        neg_emb = self.item_embedding(neg_items)

        curr_states_proj = self.aux_proj(curr_states)
        pos_score = (curr_states_proj * pos_emb).sum(dim=-1)
        neg_score = (curr_states_proj * neg_emb).sum(dim=-1)

        loss = (
            -torch.log(torch.sigmoid(pos_score) + 1e-8)
            - torch.log(1.0 - torch.sigmoid(neg_score) + 1e-8)
        )
        return (loss * valid.float()).sum() / valid.float().sum()

    def _last_valid_state(
        self,
        states: torch.Tensor,
        hist_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """取每个样本最后一个有效行为对应的兴趣状态。"""
        seq_lens = (hist_item_ids != 0).sum(dim=1).long()
        last_idx = (seq_lens - 1).clamp(min=0)
        batch_idx = torch.arange(states.size(0), device=states.device)
        return states[batch_idx, last_idx]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        category_ids = batch["category_id"]
        hist_item_ids = batch["hist_item_ids"]

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        category_emb = self.category_embedding(category_ids)

        hist_emb = self.item_embedding(hist_item_ids)
        mask = (hist_item_ids != 0).float()

        interest_states, _ = self.interest_extractor(hist_emb)
        interest_states = interest_states * mask.unsqueeze(-1)

        evolve_weights = self._compute_evolve_attention(
            interest_states, item_emb, mask
        )
        evolved_states, _ = self.interest_evolver(
            interest_states, evolve_weights, mask=mask
        )
        evolved_interest = self._last_valid_state(evolved_states, hist_item_ids)

        features = torch.cat(
            [user_emb, item_emb, category_emb, evolved_interest], dim=-1
        )
        logits = self.mlp(features).squeeze(-1)
        preds = torch.sigmoid(logits)

        if return_aux:
            aux_loss = self._auxiliary_loss(interest_states, hist_item_ids)
            return preds, aux_loss
        return preds

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        preds, aux_loss = self(batch, return_aux=True)
        target_loss = F.binary_cross_entropy(preds, batch["label"])
        loss = target_loss + self.aux_loss_weight * aux_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_target_loss", target_loss, on_step=False, on_epoch=True)
        self.log("train_aux_loss", aux_loss, on_step=False, on_epoch=True)
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
