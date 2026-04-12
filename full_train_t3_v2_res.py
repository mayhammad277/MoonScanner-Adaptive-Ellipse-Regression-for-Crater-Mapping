import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
from tqdm import tqdm
from torchvision.models import swin_t, Swin_T_Weights

# --- 1. MODEL ARCHITECTURE ---
class SwinCraterV4(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pre-trained Swin-T
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        
        # Neck: Bridge Swin's 20x20 output to your 160x160 target
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Separate Heads
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        x = self.backbone(x) # Output: [B, 20, 20, 768]
        x = x.permute(0, 3, 1, 2).contiguous() # Move channels to dim 1
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. DATASET & LOSS (Kept from your V3 for compatibility) ---
class CraterDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.ids = [f.replace('_img.npy', '') for f in os.listdir(folder) if f.endswith('_img.npy')]
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, f"{self.ids[idx]}_img.npy"))
        gt = np.load(os.path.join(self.folder, f"{self.ids[idx]}_gt.npy"))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        return img, torch.from_numpy(gt).float()

class PeakFocusedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')
    def forward(self, preds, target):
        gt_hm = target[:, 0:1, :, :]
        pred_hm = torch.sigmoid(preds['hm']).clamp(1e-4, 1-1e-4)
        pos_mask = (gt_hm == 1.0).float()
        neg_mask = (gt_hm < 1.0).float()
        
        # HM Loss (Weighted higher for confidence)
        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, 2) * pos_mask
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, 2) * torch.pow(1 - gt_hm, 4) * neg_mask
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + 1e-4)
        
        # Regression
        reg_mask = (gt_hm == 1.0)
        num_reg = reg_mask.float().sum() + 1e-4
        r_loss = self.l1(preds['r'][reg_mask], target[:, 1:2, :, :][reg_mask]) / num_reg
        off_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 2:4, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        return (2.0 * hm_loss) + (0.1 * r_loss) + (0.1 * off_loss)

# --- 3. TRAIN LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterV4().to(device)
    
    # Differential Learning Rates
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5}, # Slow backbone
        {'params': model.neck.parameters(), 'lr': 1e-4},
        {'params': model.hm.parameters(), 'lr': 1e-4}
    ])
    
    loader = DataLoader(CraterDataset("./processed_data_aug"), batch_size=8, shuffle=True)
    scaler = GradScaler()
    criterion = PeakFocusedLoss()

    for epoch in range(30):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, gts in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())
            
        torch.save(model.state_dict(), "swin_v4_model.pth")

if __name__ == "__main__":
    train()
