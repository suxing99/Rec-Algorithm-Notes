"""DIEN 兴趣演化层：AUGRU（GRU with Attentional Update Gate）。"""

from typing import Optional, Tuple

import torch
from torch import nn


class AUGRU(nn.Module):
    """
    GRU with Attentional Update Gate.

    用 attention 分数缩放 update gate，弱化与候选 item 无关的历史兴趣对演化过程的干扰。
    参考: Deep Interest Evolution Network (Zhou et al., 2018)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_scores: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (batch, seq_len, input_size) 兴趣状态序列
            attention_scores: (batch, seq_len) 与候选 item 的相关性分数
            mask: (batch, seq_len) 1 表示有效步，0 表示 padding
            hidden: (batch, hidden_size) 初始隐状态

        Returns:
            outputs: (batch, seq_len, hidden_size) 各步演化后的兴趣状态
            hidden: (batch, hidden_size) 最后一步隐状态
        """
        batch_size, seq_len, _ = inputs.shape
        if hidden is None:
            hidden = inputs.new_zeros(batch_size, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            x = inputs[:, t]
            att = attention_scores[:, t].unsqueeze(-1)

            reset_gate = torch.sigmoid(self.W_ir(x) + self.W_hr(hidden))
            update_gate = torch.sigmoid(self.W_iz(x) + self.W_hz(hidden))
            new_hidden = torch.tanh(
                self.W_in(x) + reset_gate * self.W_hn(hidden)
            )

            update_gate = att * update_gate
            hidden = (1.0 - update_gate) * hidden + update_gate * new_hidden
            outputs.append(hidden.unsqueeze(1))

        outputs_tensor = torch.cat(outputs, dim=1)
        if mask is not None:
            outputs_tensor = outputs_tensor * mask.unsqueeze(-1)

        return outputs_tensor, hidden
