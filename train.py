
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.crffnet import CRFFNet

# Placeholder dataset class
class VoDDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        radar = torch.rand(100,5)
        edge_index = torch.randint(0,100,(2,500))
        camera = torch.rand(3,720,1280)
        target = torch.rand(7)
        return radar, edge_index, camera, target

dataloader = DataLoader(VoDDataset(), batch_size=2)

model = CRFFNet().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = torch.nn.MSELoss()

for epoch in range(2):
    for radar, edge_index, camera, target in dataloader:
        radar, camera, target = radar.cuda(), camera.cuda(), target.cuda()
        optimizer.zero_grad()
        pred = model(radar, edge_index, camera)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} done')
