"""激活函数模块。"""

import torch
from torch import nn


class Dice(nn.Module):
    """Data Adaptive Activation Function (DIN, Alibaba 2018)."""

    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(x)
        gate = torch.sigmoid(normalized)
        return self.alpha * (1.0 - gate) * x + gate * x
