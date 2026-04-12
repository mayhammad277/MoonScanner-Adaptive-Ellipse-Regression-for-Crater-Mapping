import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
from tqdm import tqdm

# --- 1. MODEL ARCHITECTURE (Your Custom Stride-4 CNN) ---
class CraterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), 
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        # Heads - NO activation at the very end to allow for stable gradients
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        feat = self.backbone(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. DATASET LOADER (With Normalization) ---
class CraterDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.ids = [f.replace('_img.npy', '') for f in os.listdir(folder) if f.endswith('_img.npy')]
        # CLAHE data works best with these standard stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, f"{self.ids[idx]}_img.npy"))
        gt = np.load(os.path.join(self.folder, f"{self.ids[idx]}_gt.npy"))
        
        # Normalize 0-255 uint8 to float tensor
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = (img - self.mean) / self.std # Apply ImageNet stats
        
        return img, torch.from_numpy(gt).float()

# --- 3. BOOSTED PEAK LOSS ---
class PeakFocusedLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:1, :, :]
        pred_hm = torch.sigmoid(preds['hm']) # Apply sigmoid for probability
        pred_hm = torch.clamp(pred_hm, min=1e-4, max=1-1e-4)

        pos_mask = (gt_hm == 1.0).float()
        neg_mask = (gt_hm < 1.0).float()
        
        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred_hm) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hm, self.beta) * neg_mask
        
        num_pos = pos_mask.sum()
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (num_pos + 1e-4)

        reg_mask = (gt_hm == 1.0)
        num_reg = reg_mask.float().sum() + 1e-4
        
        r_loss = self.l1(preds['r'][reg_mask], target[:, 1:2, :, :][reg_mask]) / num_reg
        off_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 2:4, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        
        # BOOST: Weighted Heatmap heavily to ensure confidence is high
        return (10.0 * hm_loss) + (0.1 * r_loss) + (0.1 * off_loss)

# --- 4. MAIN TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CraterModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    criterion = PeakFocusedLoss()
    scaler = GradScaler()
    
    # Ensure you are pointing to the V2 (CLAHE) processed data
    dataset = CraterDataset("./processed_data_aug_v2")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    for epoch in range(40):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/40")
        
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
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"custom_clahe_ep{epoch+1}.pth")

    torch.save(model.state_dict(), "custom_clahe_final.pth")

if __name__ == "__main__":
    train()
