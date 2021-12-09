import torch
from torch import nn

"""
from https://d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html
"""


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, dim, dropout=0.1, max_length=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = nn.parameter.Parameter(torch.zeros((1, max_length, dim)),False)
        X = torch.arange(max_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :]
        return self.dropout(X)