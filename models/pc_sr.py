
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DenseGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels, improved=True)
        self.conv2 = GCNConv(out_channels, out_channels, improved=True)
        self.dilation = dilation

    def forward(self, x, edge_index):
        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.conv2(x1, edge_index))
        return x2

class IDGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block1 = DenseGCNBlock(in_channels, hidden_channels, dilation=1)
        self.block2 = DenseGCNBlock(in_channels, hidden_channels, dilation=2)
        self.block3 = DenseGCNBlock(in_channels, hidden_channels, dilation=3)

    def forward(self, x, edge_index):
        f1 = self.block1(x, edge_index)
        f2 = self.block2(x, edge_index)
        f3 = self.block3(x, edge_index)
        return torch.cat([f1, f2, f3], dim=-1)

class PCSR(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=5):
        super().__init__()
        self.idgcn = IDGCN(in_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels*3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        features = self.idgcn(x, edge_index)
        out = self.mlp(features)
        return out
