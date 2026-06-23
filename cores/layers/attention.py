"""DIN 局部激活单元（Local Activation Unit）。"""

from typing import Optional, Tuple

import torch
from torch import nn


class LocalActivationUnit(nn.Module):
    """
    根据候选 item 对历史行为序列做 attention 加权 pooling。

    attention 输入为 [query, key, query-key, query*key] 四路拼接。
    """

    def __init__(self, embed_dim: int, hidden_dims: Tuple[int, ...] = (80, 40)):
        super().__init__()
        layers = []
        input_dim = embed_dim * 4
        for hidden in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden), nn.PReLU()])
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, embed_dim) 候选 item embedding
            keys: (batch, seq_len, embed_dim) 历史行为 embedding
            mask: (batch, seq_len) 1 表示有效行为，0 表示 padding

        Returns:
            interest: (batch, embed_dim) 用户兴趣向量
            weights: (batch, seq_len) attention 权重
        """
        seq_len = keys.size(1)
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        att_input = torch.cat(
            [query_expanded, keys, query_expanded - keys, query_expanded * keys],
            dim=-1,
        )
        scores = self.mlp(att_input).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0.0)

        weights = torch.sigmoid(scores)
        if mask is not None:
            weights = weights * mask

        interest = (weights.unsqueeze(-1) * keys).sum(dim=1)
        return interest, weights
