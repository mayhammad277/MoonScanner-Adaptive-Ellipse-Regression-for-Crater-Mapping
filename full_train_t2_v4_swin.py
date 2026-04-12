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
class SwinCraterV3(nn.Module):
    def __init__(self):
        super().__init__()
        # Official Swin-T backbone (pre-trained)
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        
        # Neck: Upsample from Swin's 20x20 output to target 160x160
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # V3 Separate Heads
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        x = self.backbone(x) # Output: [B, 20, 20, 768]
        x = x.permute(0, 3, 1, 2).contiguous() # Fix dimension order: BHWC -> BCHW
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. PEAK-FOCUSED LOSS ---
class PeakFocusedLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:1, :, :]
        pred_hm = torch.sigmoid(preds['hm']).clamp(min=1e-4, max=1-1e-4)

        pos_mask = (gt_hm == 1.0).float()
        neg_mask = (gt_hm < 1.0).float()
        
        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hm, self.beta) * neg_mask
        
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + 1e-4)

        reg_mask = (gt_hm == 1.0)
        num_reg = reg_mask.float().sum() + 1e-4
        r_loss = self.l1(preds['r'][reg_mask], target[:, 1:2, :, :][reg_mask]) / num_reg
        off_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 2:4, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        
        return hm_loss + (0.1 * r_loss) + (0.1 * off_loss)

# --- 3. DATASET ---
class CraterDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.ids = [f.replace('_img.npy', '') for f in os.listdir(folder) if f.endswith('_img.npy')]
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, f"{self.ids[idx]}_img.npy"))
        gt = np.load(os.path.join(self.folder, f"{self.ids[idx]}_gt.npy"))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        gt = torch.from_numpy(gt).float()
        return img, gt

# --- 4. TRAIN LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterV3().to(device)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.neck.parameters(), 'lr': 1e-4},
        {'params': model.hm.parameters(), 'lr': 1e-4},
        {'params': model.rad.parameters(), 'lr': 1e-4},
        {'params': model.off.parameters(), 'lr': 1e-4}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = PeakFocusedLoss()
    scaler = GradScaler()
    
    loader = DataLoader(CraterDataset("./processed_data_aug"), batch_size=8, shuffle=True)

    for epoch in range(40):
        model.train()
        epoch_loss = 0
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
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        torch.save(model.state_dict(), "swin_v4_final.pth")

if __name__ == "__main__":
    train()
