
import torch
import torch.nn as nn
import torchvision.models as models

class CameraBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        feat_map = self.backbone(x)
        bev_feat = self.bev_transform(feat_map)
        return bev_feat

    def bev_transform(self, feat_map):
        # Placeholder for depth probability & height compression
        return feat_map
