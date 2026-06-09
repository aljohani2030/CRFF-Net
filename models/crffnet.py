
import torch
import torch.nn as nn
from models.pc_sr import PCSR
from models.camera_bev import CameraBranch
from models.cmff import DeformableAttention
from models.detector import DetectionHead

class CRFFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pc_sr = PCSR()
        self.camera_branch = CameraBranch()
        self.cmff = DeformableAttention(dim=64)
        self.detector = DetectionHead(in_dim=64)

    def forward(self, radar, edge_index, camera):
        radar_sr = self.pc_sr(radar, edge_index)
        camera_feat = self.camera_branch(camera)
        fused_feat = self.cmff(camera_feat, radar_sr)
        pred = self.detector(fused_feat)
        return pred
