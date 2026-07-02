"""OneTrans 模型：统一序列与非序列特征的 Transformer 精排骨干。"""

from typing import Any, Dict, List, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from cores.datamodules.two_tower import log_stage
from cores.layers.onetrans import OneTransStack, RMSNorm
from cores.utils.metrics import group_ranking_metrics


# 与特征表对齐：上线特征，排除仅训练使用的 pos
NS_CAT_FIELDS = [
    "model",
    "channel",
    "version",
    "network",
    "platform",
    "channel_game",
    "page_num",
    "hour",
    "week",
    "is_coin_stock",
    "top_scene",
    "moduleid",
    "scene",
    "open_server",
    "welfare",
    "is_free_id",
    "free_money_id",
    "discount_id",
    "moneys_all_bucket",
]

NS_MULTI_CAT_FIELDS = [
    "kind_id",
    "tag_id",
    "top2_tag_id",
    "welfare_tag_id",
    "feature_tag_id",
]

# Timestamp-agnostic 合并：按意图强度从高到低排列（强意图在前，因果注意力下弱意图可见强意图）
SEQ_FIELDS: List[Tuple[str, str]] = [
    ("global_recharge_types", "recharge_type"),
    ("recharge_game", "game"),
    ("play_game", "game"),
    ("finish_down", "game"),
    ("discount_game", "game"),
    ("entry_detail", "game"),
]

NUM_FEAT_FIELDS = ["user_num_feats", "num_feats", "cross_num_feats"]

# 候选游戏 ID（与行为序列中的 game id 共享 embedding）
CANDIDATE_GAME_FIELD = "game_id"

# 与 game_id 共享 game_embedding 的类别字段
SHARED_GAME_CAT_FIELDS = frozenset({"channel_game"})

# 支持单独配置 embedding 维度的离散特征
EMBEDDING_FIELDS = (
    [f for f in NS_CAT_FIELDS if f not in SHARED_GAME_CAT_FIELDS]
    + list(NS_MULTI_CAT_FIELDS)
    + [CANDIDATE_GAME_FIELD, "recharge_type", "seq_type"]
)


def normalize_embed_dims(
    embed_dim: int,
    embed_dims: Dict[str, int] | None = None,
) -> Dict[str, int]:
    """解析 embed_dims 覆盖项，未配置字段回退到 embed_dim。"""
    if not embed_dims:
        return {}
    raw = {field: int(dim) for field, dim in embed_dims.items()}
    if "channel_game" in raw:
        if CANDIDATE_GAME_FIELD in raw and raw["channel_game"] != raw[CANDIDATE_GAME_FIELD]:
            raise ValueError("channel_game 与 game_id 共享 embedding，维度需一致")
        raw.setdefault(CANDIDATE_GAME_FIELD, raw["channel_game"])
        del raw["channel_game"]
    unknown = set(raw) - set(EMBEDDING_FIELDS)
    if unknown:
        raise ValueError(f"embed_dims 含未知字段: {sorted(unknown)}")
    return raw


class OneTrans(pl.LightningModule):
    """
    OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer.

    论文: https://arxiv.org/abs/2510.26104

    本实现面向 LTV30（recharge_30d）回归任务，包含：
    - 序列 / 非序列统一 tokenizer
    - 混合参数 OneTrans Block（S-token 共享，NS-token 独立）
    - 金字塔式 S-token 裁剪
    - Auto-Split 非序列 tokenizer
  """

    def __init__(
        self,
        feature_stats: Dict[str, int],
        embed_dim: int = 16,
        embed_dims: Dict[str, int] | None = None,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        ns_num_tokens: int = 8,
        max_seq_len: int = 30,
        num_seqs: int = len(SEQ_FIELDS),
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        min_s_keep: int = 4,
        lr: float = 1e-3,
        target_log1p: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_stats"])
        self.embed_dim = embed_dim
        self.embed_dims = normalize_embed_dims(embed_dim, embed_dims)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_seqs = num_seqs
        self.ns_num_tokens = ns_num_tokens
        self.target_log1p = target_log1p
        self.lr = lr

        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._val_group_ids: List[str] = []

        self.user_num_dim = feature_stats["user_num_dim"]
        self.num_feats_dim = feature_stats["num_feats_dim"]
        self.cross_num_dim = feature_stats["cross_num_dim"]

        # 类别 embedding（channel_game 与 game_id 共享 game_embedding）
        self.cat_embeddings = nn.ModuleDict()
        for field in NS_CAT_FIELDS:
            if field in SHARED_GAME_CAT_FIELDS:
                continue
            vocab = feature_stats[f"num_{field}"] + 1
            self.cat_embeddings[field] = nn.Embedding(
                vocab, self._field_dim(field), padding_idx=0
            )

        self.multi_cat_embeddings = nn.ModuleDict()
        for field in NS_MULTI_CAT_FIELDS:
            vocab = feature_stats[f"num_{field}"] + 1
            self.multi_cat_embeddings[field] = nn.Embedding(
                vocab, self._field_dim(field), padding_idx=0
            )

        self.game_embedding = nn.Embedding(
            feature_stats["num_games"] + 1,
            self._field_dim(CANDIDATE_GAME_FIELD),
            padding_idx=0,
        )
        self.recharge_type_embedding = nn.Embedding(
            feature_stats["num_recharge_types"] + 1,
            self._field_dim("recharge_type"),
            padding_idx=0,
        )
        self.seq_type_embedding = nn.Embedding(
            num_seqs, self._field_dim("seq_type")
        )

        # 序列事件投影：game / recharge_type 事件维度可能不同
        self.seq_event_projs = nn.ModuleDict()
        for event_type in ("game", "recharge_type"):
            event_field = CANDIDATE_GAME_FIELD if event_type == "game" else "recharge_type"
            proj_in_dim = self._field_dim(event_field) + self._field_dim("seq_type")
            self.seq_event_projs[event_type] = nn.Sequential(
                nn.Linear(proj_in_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        self.sep_tokens = nn.Parameter(torch.randn(num_seqs - 1, d_model) * 0.02)

        # Auto-Split NS tokenizer（含候选 game_id embedding）
        ns_input_dim = (
            sum(
                self._field_dim(CANDIDATE_GAME_FIELD)
                if field in SHARED_GAME_CAT_FIELDS
                else self._field_dim(field)
                for field in NS_CAT_FIELDS
            )
            + sum(self._field_dim(field) for field in NS_MULTI_CAT_FIELDS)
            + self._field_dim(CANDIDATE_GAME_FIELD)
            + self.user_num_dim
            + self.num_feats_dim
            + self.cross_num_dim
        )
        self.ns_tokenizer = nn.Sequential(
            nn.Linear(ns_input_dim, d_model * ns_num_tokens),
            nn.GELU(),
            nn.Linear(d_model * ns_num_tokens, d_model * ns_num_tokens),
        )

        self.input_norm = RMSNorm(d_model)
        self.backbone = OneTransStack(
            num_layers=num_layers,
            dim=d_model,
            num_heads=num_heads,
            ns_num_tokens=ns_num_tokens,
            min_s_keep=min_s_keep,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )

        # 任务塔：聚合 NS-token 输出做回归
        self.task_head = nn.Sequential(
            nn.Linear(d_model * ns_num_tokens, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _field_dim(self, field: str) -> int:
        return self.embed_dims.get(field, self.embed_dim)

    def _embed_multi_cat(self, field: str, values: torch.Tensor) -> torch.Tensor:
        """对多值类别做 masked mean pooling。values: (B, max_len)"""
        emb = self.multi_cat_embeddings[field](values)
        mask = (values != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count

    def _tokenize_ns(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for field in NS_CAT_FIELDS:
            if field in SHARED_GAME_CAT_FIELDS:
                parts.append(self.game_embedding(batch[field]))
            else:
                parts.append(self.cat_embeddings[field](batch[field]))
        for field in NS_MULTI_CAT_FIELDS:
            parts.append(self._embed_multi_cat(field, batch[field]))
        parts.append(self.game_embedding(batch[CANDIDATE_GAME_FIELD]))

        parts.append(batch["user_num_feats"].float())
        parts.append(batch["num_feats"].float())
        parts.append(batch["cross_num_feats"].float())

        flat = torch.cat(parts, dim=-1)
        tokens = self.ns_tokenizer(flat).view(-1, self.ns_num_tokens, self.d_model)
        return tokens

    def _tokenize_seq(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将多行为序列合并为 S-token 序列（序列间插入 [SEP]）。"""
        batch_size = batch[NS_CAT_FIELDS[0]].size(0)
        seq_chunks: List[torch.Tensor] = []

        for seq_idx, (field, seq_type) in enumerate(SEQ_FIELDS):
            seq_ids = batch[field]
            if seq_type == "game":
                event_emb = self.game_embedding(seq_ids)
            else:
                event_emb = self.recharge_type_embedding(seq_ids)

            type_emb = self.seq_type_embedding(
                torch.full((batch_size, seq_ids.size(1)), seq_idx, device=seq_ids.device)
            )
            event_input = torch.cat([event_emb, type_emb], dim=-1)
            s_tokens = self.seq_event_projs[seq_type](event_input)

            mask = (seq_ids != 0).float().unsqueeze(-1)
            s_tokens = s_tokens * mask

            seq_chunks.append(s_tokens)
            if seq_idx < self.num_seqs - 1:
                sep = self.sep_tokens[seq_idx].unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
                seq_chunks.append(sep)

        return torch.cat(seq_chunks, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        s_tokens = self._tokenize_seq(batch)
        ns_tokens = self._tokenize_ns(batch)

        x = torch.cat([s_tokens, ns_tokens], dim=1)
        x = self.input_norm(x)

        num_s = s_tokens.size(1)
        x = self.backbone(x, num_s)

        ns_out = x[:, -self.ns_num_tokens :, :].reshape(x.size(0), -1)
        return self.task_head(ns_out).squeeze(-1)

    def _transform_target(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_log1p:
            return torch.log1p(y.clamp(min=0.0))
        return y

    def _inverse_transform_pred(self, pred: torch.Tensor) -> torch.Tensor:
        if self.target_log1p:
            return torch.expm1(pred).clamp(min=0.0)
        return pred

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        y = batch["recharge_30d"].float()
        pred = self(batch)
        target = self._transform_target(y)
        loss = F.huber_loss(pred, target, delta=1.0)

        with torch.no_grad():
            pred_ltv = self._inverse_transform_pred(pred)
            mae = (pred_ltv - y).abs().mean()
            rmse = ((pred_ltv - y) ** 2).mean().sqrt()

        batch_size = y.size(0)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_mae", mae, prog_bar=(stage == "val"), batch_size=batch_size)
        self.log(f"{stage}_rmse", rmse, batch_size=batch_size)
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_labels.clear()
        self._val_group_ids.clear()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y = batch["recharge_30d"].float()
        pred = self(batch)
        target = self._transform_target(y)
        loss = F.huber_loss(pred, target, delta=1.0)

        with torch.no_grad():
            pred_ltv = self._inverse_transform_pred(pred)
            mae = (pred_ltv - y).abs().mean()
            rmse = ((pred_ltv - y) ** 2).mean().sqrt()

        batch_size = y.size(0)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val_mae", mae, prog_bar=True, batch_size=batch_size)
        self.log("val_rmse", rmse, batch_size=batch_size)

        self._val_preds.append(pred_ltv.detach().cpu())
        self._val_labels.append(y.detach().cpu())
        versionv = batch["versionv"]
        if isinstance(versionv, (list, tuple)):
            self._val_group_ids.extend(versionv)
        else:
            self._val_group_ids.append(versionv)

        return loss

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self._val_preds).numpy()
        labels = torch.cat(self._val_labels).numpy()
        metrics = group_ranking_metrics(
            self._val_group_ids,
            preds,
            labels,
            positive_threshold=0.0,
            ks=(1, 3),
        )

        num_groups = int(metrics["num_groups"])
        self.log("val_mrr", metrics["mrr"], prog_bar=True, on_epoch=True)
        self.log("val_hit@1", metrics["hit@1"], prog_bar=True, on_epoch=True)
        self.log("val_hit@3", metrics["hit@3"], prog_bar=True, on_epoch=True)

        if num_groups > 0:
            log_stage(
                f"验证排序指标 (有效推荐组 {num_groups}): "
                f"MRR={metrics['mrr']:.4f}, "
                f"Hit@1={metrics['hit@1']:.4f}, "
                f"Hit@3={metrics['hit@3']:.4f}"
            )
        else:
            log_stage("验证排序指标: 无有效推荐组（所有 versionv 组均无充值）")

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
