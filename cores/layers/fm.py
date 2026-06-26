"""Factorization Machine 相关层。"""

from typing import Optional

import torch
from torch import nn


def fm_second_order(
    field_embeddings: torch.Tensor,
    field_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FM 二阶特征交叉的高效计算。

    公式（向量 v_i ∈ R^k，标量 x_i）::

        1/2 * Σ_f [ (Σ_i v_{i,f} x_i)^2 - Σ_i v_{i,f}^2 x_i^2 ]

    等价于论文中常见的向量写法::

        1/2 * sum_f( (∑ v_i x_i)^2 - Σ v_i^2 x_i^2 )_f

    其中对向量的平方为**逐元素**平方，最后对 embedding 维度 f 求和得到标量。
    稀疏类别特征每个域仅一个非零值，可取 x_i = 1，此时 field_embeddings 即为 v_i x_i。

    Args:
        field_embeddings: v_i 或 v_i x_i，shape (batch, num_fields, embed_dim)
        field_values: 可选特征值 x_i，shape (batch, num_fields) 或 (batch, num_fields, 1)

    Returns:
        shape (batch,) 的二阶得分
    """
    vx = field_embeddings
    if field_values is not None:
        if field_values.dim() == 2:
            field_values = field_values.unsqueeze(-1)
        vx = field_embeddings * field_values

    sum_vx = vx.sum(dim=1)  # ∑_i v_i x_i，shape (batch, embed_dim)
    sum_vx_sq = sum_vx.pow(2)  # (∑ v_i x_i)^2，逐元素
    vi_sq_xi_sq = vx.pow(2).sum(dim=1)  # ∑_i v_i^2 x_i^2，逐元素
    return 0.5 * (sum_vx_sq - vi_sq_xi_sq).sum(dim=1)


class FactorizationMachine(nn.Module):
    """
    FM 一阶 + 二阶项（不含 Deep 分支）。

    每个稀疏特征域各有一阶权重 embedding(1) 与二阶 embedding(k)；
    连续特征仅参与一阶线性项。
    """

    def __init__(
        self,
        field_sizes: tuple[int, ...],
        num_continuous: int,
        embed_dim: int,
    ):
        super().__init__()
        self.first_orders = nn.ModuleList(
            [nn.Embedding(size + 1, 1, padding_idx=0) for size in field_sizes]
        )
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(size + 1, embed_dim, padding_idx=0) for size in field_sizes]
        )
        self.continuous_first = nn.Linear(num_continuous, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        field_ids: tuple[torch.Tensor, ...],
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        first = self.bias
        for emb, ids in zip(self.first_orders, field_ids):
            first = first + emb(ids).squeeze(-1)
        first = first + self.continuous_first(continuous).squeeze(-1)

        embeds = [emb(ids) for emb, ids in zip(self.field_embeddings, field_ids)]
        second = fm_second_order(torch.stack(embeds, dim=1))
        return first + second
