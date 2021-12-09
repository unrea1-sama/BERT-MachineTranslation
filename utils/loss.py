import torch
from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss,self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, y_pred, y, mask):
        return torch.mean(self.loss_fn(y_pred, y) * mask)
