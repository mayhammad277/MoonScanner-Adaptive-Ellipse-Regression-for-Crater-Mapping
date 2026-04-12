import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
from tqdm import tqdm
from torchvision.models import swin_t, Swin_T_Weights

# --- 1. ARCHITECTURE (Must match Inference) ---
class SwinCraterV4(nn.Module):
    def __init__(self):
        super().__init__()
        # Load weights for backbone, but heads start fresh
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Separate Heads - IMPORTANT: No ReLU at the end of Conv2d(64, 1, 1)
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        x = self.backbone(x) 
        x = x.permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. DATASET (With ImageNet Normalization) ---
class CraterDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.ids = [f.replace('_img.npy', '') for f in os.listdir(folder) if f.endswith('_img.npy')]
        # Stats must match Inference
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.folder, f"{self.ids[idx]}_img.npy"))
        gt = np.load(os.path.join(self.folder, f"{self.ids[idx]}_gt.npy"))
        
        # img is saved as uint8 (0-255), convert to float and normalize
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = (img - self.mean) / self.std
        return img, torch.from_numpy(gt).float()
class PeakFocusedLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha # Focal loss power to sharpen peaks
        self.beta = beta   # Penalty for background noise
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        # target shape: [Batch, 4, 160, 160]
        gt_hm = target[:, 0:1, :, :]
        pred_hm = torch.sigmoid(preds['hm']) # Predicted probability map
        
        # 1. THE FOCAL PENALTY:
        # This formula heavily penalizes the model for being "sort of sure" (red circles)
        # on pixels that should be background.
        pos_mask = (gt_hm == 1.0).float()
        neg_mask = (gt_hm < 1.0).float()
        
        # Penalize missing the center
        pos_loss = torch.log(pred_hm + 1e-6) * torch.pow(1 - pred_hm, self.alpha) * pos_mask
        
        # PENALIZE BACKGROUND NOISE: 
        # (1-gt_hm)^beta reduces loss for pixels near a crater so the model 
        # doesn't get confused by the "glow" of the Gaussian peak.
        neg_loss = torch.log(1 - pred_hm + 1e-6) * torch.pow(pred_hm, self.alpha) * torch.pow(1 - gt_hm, self.beta) * neg_mask
        
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + 1e-4)

        # 2. MASKED REGRESSION:
        reg_mask = (gt_hm == 1.0)
        num_reg = reg_mask.float().sum() + 1e-4
        r_loss = self.l1(preds['r'][reg_mask], target[:, 1:2, :, :][reg_mask]) / num_reg
        off_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 2:4, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        
        # 3. THE "GREEN CIRCLE" MULTIPLIER:
        # We increase this to 20.0 to force the optimizer to fix the heatmap first.
        return (20.0 * hm_loss) + (0.1 * r_loss) + (0.1 * off_loss)
# --- 3. HARD-PEAK LOSS ---
class HardPeakLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:1, :, :]
        pred_hm = torch.sigmoid(preds['hm']) # Squash to 0-1
        
        pos_mask = (gt_hm == 1.0).float()
        neg_mask = (gt_hm < 1.0).float()
        
        # Focal Loss variant
        pos_loss = torch.log(pred_hm + 1e-6) * torch.pow(1 - pred_hm, 2) * pos_mask
        neg_loss = torch.log(1 - pred_hm + 1e-6) * torch.pow(pred_hm, 2) * torch.pow(1 - gt_hm, 4) * neg_mask
        
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + 1e-4)

        # Regression only at peak centers
        reg_mask = (gt_hm == 1.0)
        num_reg = reg_mask.float().sum() + 1e-4
        r_loss = self.l1(preds['r'][reg_mask], target[:, 1:2, :, :][reg_mask]) / num_reg
        off_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 2:4, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        
        # THE BIG BOOST: 15.0x multiplier on Heatmap
        #return (15.0 * hm_loss) + (0.1 * r_loss) + (0.1 * off_loss)
        # Force the model to care 20x more about getting the heatmap peak right
        return (20.0 * hm_loss) + (0.1 * r_loss) + (0.1 * off_loss)
# --- 4. TRAINING LOOP ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterV4().to(device)
    
    # Differential LR: Freeze backbone partially to keep pre-trained edge detection
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-6}, 
        {'params': model.neck.parameters(), 'lr': 1e-4},
        {'params': model.hm.parameters(), 'lr': 1e-4},
        {'params': model.rad.parameters(), 'lr': 1e-4},
        {'params': model.off.parameters(), 'lr': 1e-4}
    ])
    
    loader = DataLoader(CraterDataset("./processed_data_aug_v2"), batch_size=8, shuffle=True)
    criterion = HardPeakLoss()
    scaler = GradScaler()

    for epoch in range(70): # 25 Epochs is usually the sweet spot
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
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Save every 5 epochs and the final
        #if (epoch + 1) % 5 == 0:
        #    torch.save(model.state_dict(), f"swin_boosted_ep{epoch+1}.pth")

    torch.save(model.state_dict(), "swin_boosted.pth")

if __name__ == "__main__":
    train()
