"""OneTrans 核心层：RMSNorm、混合参数因果注意力与 FFN。"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class MixedLinear(nn.Module):
    """
    混合线性层：S-token 共享权重，NS-token 各自独立权重。

    前 num_s 个位置使用共享 Linear；后 ns_num_tokens 个位置使用逐 token 权重。
    """

    def __init__(self, in_dim: int, out_dim: int, ns_num_tokens: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ns_num_tokens = ns_num_tokens
        self.shared = nn.Linear(in_dim, out_dim, bias=bias)
        self.ns_weight = nn.Parameter(torch.empty(ns_num_tokens, out_dim, in_dim))
        self.ns_bias = nn.Parameter(torch.zeros(ns_num_tokens, out_dim)) if bias else None
        nn.init.xavier_uniform_(self.ns_weight)

    def forward(self, x: torch.Tensor, num_s: int) -> torch.Tensor:
        """x: (B, L, D_in)，前 num_s 为 S-token，后 ns_num_tokens 为 NS-token。"""
        if num_s > 0:
            s_out = self.shared(x[:, :num_s])
        else:
            s_out = x.new_zeros(x.size(0), 0, self.out_dim)

        if self.ns_num_tokens > 0:
            ns_x = x[:, num_s : num_s + self.ns_num_tokens]
            ns_out = torch.einsum("bnd,nod->bno", ns_x, self.ns_weight)
            if self.ns_bias is not None:
                ns_out = ns_out + self.ns_bias.unsqueeze(0)
        else:
            ns_out = x.new_zeros(x.size(0), 0, self.out_dim)

        return torch.cat([s_out, ns_out], dim=1)


class MixedFFN(nn.Module):
    """混合 FFN：S-token 共享，NS-token 逐 token 独立。"""

    def __init__(self, dim: int, ns_num_tokens: int, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.ns_num_tokens = ns_num_tokens
        self.fc1 = MixedLinear(dim, hidden, ns_num_tokens)
        self.fc2 = MixedLinear(hidden, dim, ns_num_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, num_s: int) -> torch.Tensor:
        h = F.gelu(self.fc1(x, num_s))
        h = self.dropout(h)
        return self.fc2(h, num_s)


class MixedCausalAttention(nn.Module):
    """混合参数多头因果注意力，支持金字塔式 S-token 查询裁剪。"""

    def __init__(self, dim: int, num_heads: int, ns_num_tokens: int, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ns_num_tokens = ns_num_tokens
        self.scale = self.head_dim**-0.5

        self.q_proj = MixedLinear(dim, dim, ns_num_tokens, bias=False)
        self.k_proj = MixedLinear(dim, dim, ns_num_tokens, bias=False)
        self.v_proj = MixedLinear(dim, dim, ns_num_tokens, bias=False)
        self.out_proj = MixedLinear(dim, dim, ns_num_tokens, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, l, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, self.dim)

    def forward(
        self,
        x: torch.Tensor,
        num_s: int,
        keep_s: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x: (B, L, D)，L = num_s + ns_num_tokens
            num_s: 当前 S-token 数量
            keep_s: 本层保留的 S-token 查询数（取尾部最近事件）

        Returns:
            out: (B, keep_s + ns_num_tokens, D)
            kept_positions: 输出 token 对应的原始位置索引
        """
        b, seq_len, _ = x.shape
        num_ns = self.ns_num_tokens
        keep_s = min(keep_s, num_s)
        s_start = num_s - keep_s
        q_positions = list(range(s_start, num_s)) + list(range(num_s, seq_len))

        k = self._split_heads(self.k_proj(x, num_s))
        v = self._split_heads(self.v_proj(x, num_s))

        q_tokens = x[:, q_positions, :]
        q_num_s = keep_s
        q = self._split_heads(self.q_proj(q_tokens, q_num_s))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        q_idx = torch.tensor(q_positions, device=x.device)
        k_idx = torch.arange(seq_len, device=x.device)
        causal = k_idx.unsqueeze(0) > q_idx.unsqueeze(1)
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out, q_num_s)
        return out, q_positions


class OneTransBlock(nn.Module):
    """OneTrans Block：Pre-norm + Mixed Causal Attention + Mixed FFN。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ns_num_tokens: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ns_num_tokens = ns_num_tokens
        self.norm1 = RMSNorm(dim)
        self.attn = MixedCausalAttention(dim, num_heads, ns_num_tokens, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = MixedFFN(dim, ns_num_tokens, ffn_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        num_s: int,
        keep_s: int,
    ) -> Tuple[torch.Tensor, int]:
        residual = x
        normed = self.norm1(x)
        attn_out, kept_positions = self.attn(normed, num_s, keep_s)

        q_len = attn_out.size(1)
        gathered_residual = residual[:, kept_positions, :]
        x = gathered_residual + self.dropout(attn_out)

        num_s_out = min(keep_s, num_s)
        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in, num_s_out)
        x = x + self.dropout(ffn_out)
        return x, num_s_out


class OneTransStack(nn.Module):
    """堆叠 OneTrans Block，逐层金字塔裁剪 S-token。"""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        ns_num_tokens: int,
        min_s_keep: int = 4,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ns_num_tokens = ns_num_tokens
        self.min_s_keep = min_s_keep
        self.blocks = nn.ModuleList(
            [
                OneTransBlock(dim, num_heads, ns_num_tokens, ffn_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def _keep_s_tokens(self, num_s: int, layer_idx: int) -> int:
        if num_s <= self.min_s_keep:
            return num_s
        ratio = (layer_idx + 1) / self.num_layers
        keep = int(num_s - ratio * (num_s - self.min_s_keep))
        return max(self.min_s_keep, min(num_s, keep))

    def forward(self, x: torch.Tensor, num_s: int) -> torch.Tensor:
        for layer_idx, block in enumerate(self.blocks):
            keep_s = self._keep_s_tokens(num_s, layer_idx)
            x, num_s = block(x, num_s, keep_s)
        return x
