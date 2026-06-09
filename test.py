
import torch
from models.crffnet import CRFFNet

model = CRFFNet().cuda()
model.eval()

# Placeholder for evaluation
radar = torch.rand(2,100,5).cuda()
edge_index = torch.randint(0,100,(2,500)).cuda()
camera = torch.rand(2,3,720,1280).cuda()
with torch.no_grad():
    pred = model(radar, edge_index, camera)
print(pred.shape)
