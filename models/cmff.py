
import torch
import torch.nn as nn

class DeformableAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, feat_img, feat_radar):
        Q = self.to_q(feat_img)
        K = self.to_k(feat_radar)
        V = self.to_v(feat_radar)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1,2))/Q.size(-1)**0.5, dim=-1)
        out = torch.bmm(attn, V)
        return self.fc(out)
