
import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_dim=64, out_dim=7):  # 3D bbox + class
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.head(x.mean(dim=1))
