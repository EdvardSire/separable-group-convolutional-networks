import torch
from torch import Tensor
import numpy as np

class RollingShutter(torch.nn.Module):
    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = std

    def forward(self, img: Tensor) -> Tensor:
        rows = img.shape[1]

        src = torch.arange(0, rows)

        dst = torch.clamp(torch.round(torch.randn(rows) * self.std + src).to(int), 0, rows - 1)

        img[:, src, :] = img[:, dst, :]
        return img
