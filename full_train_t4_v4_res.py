import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm
from torchvision.models import swin_t, Swin_T_Weights

# --- 1. ARCHITECTURE (V4 Logic with Mahanti Heads) ---
class SwinCraterMahantiV4(nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Heads: 5 Class HM, 2 Axes (Maj/Min), 2 Offsets (dx/dy), 1 Rotation
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

# --- 2. DATASET (V4 Normalization + Hybrid Heatmaps) ---
class MahantiDatasetV4(Dataset):
    def __init__(self, csv_file, img_dir, heatmap_dir):
        self.df = pd.read_csv(csv_file).dropna(subset=['crater_classification'])
        self.img_dir = os.path.expanduser(img_dir)
        self.heatmap_dir = os.path.expanduser(heatmap_dir)
        self.unique_images = self.df['inputImage'].unique()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.scale = 160 / 2592

    def __len__(self): return len(self.unique_images)

    def __getitem__(self, idx):
        img_name = self.unique_images[idx]
        img_path = os.path.join(self.img_dir, img_name if img_name.endswith('.png') else img_name + ".png")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = (img - self.mean) / self.std

        hm_path = os.path.join(self.heatmap_dir, img_name.split('.')[0] + ".npy")
        existing_hm = np.load(hm_path) if os.path.exists(hm_path) else None

        gt = torch.zeros((10, 160, 160))
        annos = self.df[self.df['inputImage'] == img_name]
        for _, row in annos.iterrows():
            cls = int(row['crater_classification'])
            ctx, cty = row['ellipseCenterX(px)'] * self.scale, row['ellipseCenterY(px)'] * self.scale
            ix, iy = int(ctx), int(cty)

            if 0 <= ix < 160 and 0 <= iy < 160:
                if existing_hm is not None:
                    r = max(2, int(row['ellipseSemimajor(px)'] * self.scale))
                    y1, y2, x1, x2 = max(0, iy-r), min(160, iy+r+1), max(0, ix-r), min(160, ix+r+1)
                    gt[cls, y1:y2, x1:x2] = torch.max(gt[cls, y1:y2, x1:x2], torch.from_numpy(existing_hm[y1:y2, x1:x2]))
                else: gt[cls, iy, ix] = 1.0
                
                gt[5, iy, ix] = row['ellipseSemimajor(px)'] * self.scale
                gt[6, iy, ix] = row['ellipseSemiminor(px)'] * self.scale
                gt[7, iy, ix] = ctx - ix
                gt[8, iy, ix] = cty - iy
                gt[9, iy, ix] = np.deg2rad(row['ellipseRotation(deg)'])
        return img, gt

# --- 3. HARD-PEAK LOSS (V4 Multiplier Logic) ---
class HardPeakLossV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:5, :, :]
        pred_hm = torch.sigmoid(preds['hm'])
        pos_mask = (gt_hm == 1.0).float()
        neg_mask = (gt_hm < 1.0).float()
        
        pos_loss = torch.log(pred_hm + 1e-6) * torch.pow(1 - pred_hm, 2) * pos_mask
        neg_loss = torch.log(1 - pred_hm + 1e-6) * torch.pow(pred_hm, 2) * torch.pow(1 - gt_hm, 4) * neg_mask
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (pos_mask.any(dim=1).sum() + 1e-4)

        reg_mask = (gt_hm.max(dim=1, keepdim=True)[0] == 1.0)
        num_reg = reg_mask.float().sum() + 1e-4
        a_loss = self.l1(preds['axes'][reg_mask.repeat(1,2,1,1)], target[:, 5:7, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        o_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 7:9, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        r_loss = self.l1(preds['rot'][reg_mask], target[:, 9:10, :, :][reg_mask]) / num_reg
        
        return (20.0 * hm_loss) + (0.1 * a_loss) + (0.1 * o_loss) + (0.1 * r_loss)

# --- 4. EXECUTION ---
if __name__ == "__main__":
    device = torch.device("cuda")
    model = SwinCraterMahantiV4().to(device)
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-6}, 
        {'params': model.neck.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if any(h in n for h in ['hm', 'axes', 'off', 'rot'])], 'lr': 1e-4}
    ])
    loader = DataLoader(MahantiDatasetV4("train-tgt.csv", "~/train", "./processed_data_aug_v2"), batch_size=8, shuffle=True)
    criterion = HardPeakLossV4()
    scaler = GradScaler()

    for epoch in range(70):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for imgs, gts in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    torch.save(model.state_dict(), "swin_mahanti_v4.pth")
