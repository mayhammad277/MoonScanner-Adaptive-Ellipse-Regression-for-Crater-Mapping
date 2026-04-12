import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
from tqdm import tqdm

# --- 1. MODEL DEFINITION ---

class CraterBackbone(nn.Module):
    """A memory-efficient CNN backbone that reduces 640x640 to 160x160 (Stride 4)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # 640 -> 320
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 320 -> 160
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

class CraterHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        # Heatmap head
        self.hm = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        # Radius head
        self.rad = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        # Offset head
        self.off = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        return {'hm': self.hm(x), 'r': self.rad(x), 'off': self.off(x)}

class CraterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CraterBackbone()
        self.head = CraterHead()

    def forward(self, x):
        return self.head(self.backbone(x))

# --- 2. LOSS AND DATASET ---

class FastCraterDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.ids = [f.replace('_img.npy', '') for f in os.listdir(folder) if f.endswith('_img.npy')]

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, f"{self.ids[idx]}_img.npy"))
        gt = np.load(os.path.join(self.folder, f"{self.ids[idx]}_gt.npy"))
        # To Torch
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        gt = torch.from_numpy(gt).float()
        return img, gt

class CraterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        # target: [4, 160, 160] -> [HM, R, Ox, Oy]
        gt_hm = target[:, 0:1, :, :]
        # Focal Loss (Simplified version)
        pred_hm = torch.sigmoid(preds['hm'])
        hm_loss = -((1 - pred_hm)**2 * gt_hm * torch.log(pred_hm + 1e-8) + 
                    (pred_hm)**2 * (1 - gt_hm) * torch.log(1 - pred_hm + 1e-8)).mean()

        # Regression Loss (only at centers)
        mask = gt_hm == 1.0
        num = mask.float().sum() + 1e-4
        r_loss = self.l1(preds['r'][mask], target[:, 1:2, :, :][mask]) / num
        off_loss = self.l1(preds['off'][mask.repeat(1,2,1,1)], target[:, 2:4, :, :][mask.repeat(1,2,1,1)]) / num
        
        return hm_loss + 0.1 * r_loss + 0.1 * off_loss

# --- 3. MAIN TRAINING LOOP ---

def main():
    device = torch.device("cuda")
    model = CraterModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CraterLoss()
    scaler = GradScaler() # Mixed Precision
    
    dataset = FastCraterDataset("./processed_data")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for epoch in range(10):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for imgs, gts in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            
            optimizer.zero_grad()
            with autocast(): # 16-bit forward pass
                preds = model(imgs)
                loss = criterion(preds, gts)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "crater_model.pth")

if __name__ == "__main__":
    main()
