"""Temporal heads over cached per-frame features: MS-TCN++ and BiGRU.

Both take [B, T, D] float features and return per-stage logits
[num_stages, B, T, C]; single-stage heads return [1, B, T, C] so the training
loop treats every head identically.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.out(F.relu(self.conv(x))))


class DualDilatedLayer(nn.Module):
    """MS-TCN++ prediction-generation layer: parallel small+large dilation."""

    def __init__(self, channels: int, d_small: int, d_large: int):
        super().__init__()
        self.conv_s = nn.Conv1d(channels, channels, 3, padding=d_small, dilation=d_small)
        self.conv_l = nn.Conv1d(channels, channels, 3, padding=d_large, dilation=d_large)
        self.fuse = nn.Conv1d(2 * channels, channels, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.conv_s(x), self.conv_l(x)], dim=1)
        return x + self.dropout(self.fuse(F.relu(h)))


class MSTCNPP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, channels: int = 64,
                 num_layers: int = 10, num_refine_stages: int = 3):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.pg = nn.ModuleList(
            DualDilatedLayer(channels, 2**i, 2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        )
        self.pg_out = nn.Conv1d(channels, num_classes, 1)
        self.refine = nn.ModuleList()
        for _ in range(num_refine_stages):
            self.refine.append(
                nn.ModuleDict(
                    {
                        "inp": nn.Conv1d(num_classes, channels, 1),
                        "layers": nn.ModuleList(
                            DilatedResidualLayer(channels, 2**i) for i in range(num_layers)
                        ),
                        "out": nn.Conv1d(channels, num_classes, 1),
                    }
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        for layer in self.pg:
            h = layer(h)
        logits = self.pg_out(h)
        outs = [logits]
        for stage in self.refine:
            h = stage["inp"](F.softmax(logits, dim=1))
            for layer in stage["layers"]:
                h = layer(h)
            logits = stage["out"](h)
            outs.append(logits)
        return torch.stack([o.transpose(1, 2) for o in outs])  # [S, B, T, C]


class BiGRU(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, bidirectional=True,
                          batch_first=True, dropout=0.3)
        self.out = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(F.relu(self.proj(x)))
        return self.out(h).unsqueeze(0)  # [1, B, T, C]


def mstcn_loss(stage_logits: torch.Tensor, target: torch.Tensor,
               class_weights: torch.Tensor | None = None, tmse_weight: float = 0.15) -> torch.Tensor:
    """Sum over stages of CE + truncated-MSE smoothing (the MS-TCN recipe)."""
    loss = stage_logits.new_zeros(())
    for logits in stage_logits:  # [B, T, C]
        flat = logits.reshape(-1, logits.shape[-1])
        loss = loss + F.cross_entropy(flat, target.reshape(-1), weight=class_weights)
        log_p = F.log_softmax(logits, dim=-1)
        tmse = torch.clamp(
            (log_p[:, 1:] - log_p[:, :-1].detach()) ** 2, max=16.0
        ).mean()
        loss = loss + tmse_weight * tmse
    return loss


HEADS = {"mstcnpp": MSTCNPP, "bigru": BiGRU}
