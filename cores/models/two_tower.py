"""双塔召回模型（Two-Tower Recall）。"""

from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

RECALL_AT_K = (1, 10, 50, 100, 200)
PRIMARY_RECALL_K = 200


class TwoTowerRecall(pl.LightningModule):
    """
    双塔召回模型：用户塔与物品塔分别编码为向量，内积衡量匹配度。

    训练方式：数据仅含正样本，每个 batch 内其余样本作为负例，
    使用 softmax 交叉熵（对角线为正样本，即 in-batch negative sampling）。

    打分公式（用于损失与召回）:
        y_pre = u · v / τ - log(p_item)

    其中 p_item 为物品被采样到的概率（与热度成正比）。batch 内负样本来自
    训练数据分布，热门游戏更常出现在 batch 中、也更频繁地被当作其他用户的
    负样本；若不做校正，模型会对热门物品积累过多负向梯度。减去 log(p_item)
    是 sampled softmax 对提议分布 Q(item)∝p_item 的标准无偏校正，使梯度
    逼近全量 softmax。推理时同一公式可得到去热门偏置的排序分。

    用户塔特征:
        - gameidx_seq: 下载过的游戏 id 序列（0 为 padding）
        - last_gameidx: 最后一次下载的游戏 id
        - modelx: 手机型号
        - versionx: app 版本
        - cityx: 城市
        - datatraceidx: 资源位 id
        - week: 星期（1-7），embedding 维度 8
        - hour: 小时（0-23），embedding 维度 8

    物品塔特征:
        - gameidx: 目标游戏 id（与序列共享 embedding）
        - p_item: 目标游戏占全部目标游戏的比例，取值 [0, 1]
    """

    def __init__(
        self,
        num_games: int,
        num_models: int,
        num_versions: int,
        num_cities: int,
        num_datatrace: int,
        num_weeks: int,
        num_hours: int,
        embed_dim: int = 128,
        max_seq_len: int = 20,
        tower_hidden_dims: Tuple[int, ...] = (128,),
        output_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        temperature: float = 0.07,
        p_item_eps: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tower_hidden_dims"])
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.p_item_eps = p_item_eps

        # 游戏 id 在用户序列、最近游戏与物品塔之间共享
        self.game_embedding = nn.Embedding(num_games + 1, embed_dim, padding_idx=0)
        self.model_embedding = nn.Embedding(num_models + 1, embed_dim, padding_idx=0)
        self.version_embedding = nn.Embedding(num_versions + 1, embed_dim, padding_idx=0)
        self.city_embedding = nn.Embedding(num_cities + 1, embed_dim, padding_idx=0)
        self.datatrace_embedding = nn.Embedding(num_datatrace + 1, embed_dim, padding_idx=0)
        temporal_embed_dim = 8
        self.week_embedding = nn.Embedding(num_weeks + 1, temporal_embed_dim, padding_idx=0)
        self.hour_embedding = nn.Embedding(num_hours + 1, temporal_embed_dim, padding_idx=0)

        user_input_dim = embed_dim * 6 + temporal_embed_dim * 2
        item_input_dim = embed_dim + 1  # game + p_item

        self.user_tower = self._build_tower(user_input_dim, tower_hidden_dims, output_dim, dropout)
        self.item_tower = self._build_tower(item_input_dim, tower_hidden_dims, output_dim, dropout)

    @staticmethod
    def _build_tower(
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = []
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def _pool_sequence(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """对游戏序列做 masked mean pooling。"""
        emb = self.game_embedding(seq_ids)
        mask = (seq_ids != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count

    def encode_user(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_emb = self._pool_sequence(batch["gameidx_seq"])
        last_emb = self.game_embedding(batch["last_gameidx"])
        model_emb = self.model_embedding(batch["modelx"])
        version_emb = self.version_embedding(batch["versionx"])
        city_emb = self.city_embedding(batch["cityx"])
        datatrace_emb = self.datatrace_embedding(batch["datatraceidx"])
        week_emb = self.week_embedding(batch["week"])
        hour_emb = self.hour_embedding(batch["hour"])

        features = torch.cat(
            [seq_emb, last_emb, model_emb, version_emb, city_emb, datatrace_emb, week_emb, hour_emb],
            dim=-1,
        )
        vec = self.user_tower(features)
        return F.normalize(vec, p=2, dim=-1)

    def encode_item(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        game_emb = self.game_embedding(batch["gameidx"])
        p_item = batch["p_item"].unsqueeze(-1)
        features = torch.cat([game_emb, p_item], dim=-1)
        vec = self.item_tower(features)
        return F.normalize(vec, p=2, dim=-1)

    def _log_p_item(self, p_item: torch.Tensor) -> torch.Tensor:
        """log(p_item)，对 0 做下限截断避免 -inf。"""
        return torch.log(p_item.clamp(min=self.p_item_eps))

    def compute_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 batch 内 user-item 全交叉打分矩阵，shape (B, B)。

        logits[i, j] = user_i · item_j / τ - log(p_item_j)
        """
        user_vec = self.encode_user(batch)
        item_vec = self.encode_item(batch)
        similarity = user_vec @ item_vec.T / self.temperature
        popularity_bias = self._log_p_item(batch["p_item"]).unsqueeze(0)
        return similarity - popularity_bias

    def _in_batch_loss(self, logits: torch.Tensor) -> torch.Tensor:
        labels = torch.eye(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _batch_recall_at_k(logits: torch.Tensor, k: int = 1) -> torch.Tensor:
        targets = torch.arange(logits.size(0), device=logits.device)
        _, topk = logits.topk(min(k, logits.size(1)), dim=1)
        hits = (topk == targets.unsqueeze(1)).any(dim=1).float().mean()
        return hits

    def _log_recalls(
        self,
        logits: torch.Tensor,
        prefix: str,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for k in RECALL_AT_K:
            recall = self._batch_recall_at_k(logits, k=k)
            metric = f"{prefix}_recall@{k}"
            show_bar = k == PRIMARY_RECALL_K
            self.log(metric, recall, prog_bar=show_bar, on_step=on_step, on_epoch=on_epoch)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回配对正样本分数 (B,)，即相似度矩阵对角线经 sigmoid。"""
        logits = self.compute_logits(batch)
        return torch.sigmoid(logits.diag())

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self.compute_logits(batch)
        loss = self._in_batch_loss(logits)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self._log_recalls(logits, prefix="train", on_step=True, on_epoch=False)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        logits = self.compute_logits(batch)
        loss = self._in_batch_loss(logits)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_recalls(logits, prefix="val", on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def recall_top_k(
        self,
        user_batch: Dict[str, torch.Tensor],
        item_gameidx: torch.Tensor,
        item_p_item: torch.Tensor,
        k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对一批用户从候选物品中召回 Top-K。

        Args:
            user_batch: 用户侧特征 batch
            item_gameidx: (num_candidates,) 候选游戏 id
            item_p_item: (num_candidates,) 候选游戏占比
            k: 召回数量

        Returns:
            scores: (batch, k) 匹配分数
            indices: (batch, k) 候选集中的下标
        """
        self.eval()
        user_vec = self.encode_user(user_batch)
        item_batch = {
            "gameidx": item_gameidx,
            "p_item": item_p_item,
        }
        item_vec = self.encode_item(item_batch)
        similarity = user_vec @ item_vec.T / self.temperature
        popularity_bias = self._log_p_item(item_p_item).unsqueeze(0)
        scores = similarity - popularity_bias
        top_scores, top_indices = torch.topk(scores, k=min(k, scores.size(1)), dim=1)
        return top_scores, top_indices
