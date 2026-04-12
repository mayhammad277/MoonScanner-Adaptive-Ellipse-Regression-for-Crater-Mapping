

import os, cv2, torch, numpy as np, pandas as pd, glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
from pathlib import Path

# --- 1. SETTINGS ---
IMG_ROOT = "./train"
CSV_PATH = "train-gt.csv"
SAVE_PREFIX = "swin_v4_ft"
BEST_MODEL = "swin_crater_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 32
LR = 8e-6

# --- 2. ARCHITECTURE (SwinV4) ---
class SwinCraterV4(nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        feat = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(feat)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

# --- 3. DATASET ---
class CraterDataset(Dataset):
    def __init__(self, csv_file, img_root):
        self.df = pd.read_csv(csv_file).dropna(subset=['crater_classification'])
        self.img_root = Path(img_root)
        self.unique_images = self.df['inputImage'].unique()
        self.scale = 160 / 2592
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.unique_images)

    def __getitem__(self, idx):
        img_rel_path = self.unique_images[idx]
        full_path = self.img_root / (img_rel_path if img_rel_path.endswith('.png') else img_rel_path + ".png")
        img_raw = cv2.imread(str(full_path))
        if img_raw is None: return self.__getitem__(np.random.randint(0, len(self)))

        img_res = cv2.resize(img_raw, (640, 640))
        img_t = (torch.from_numpy(img_res).permute(2,0,1).float() / 255.0 - self.mean) / self.std

        gt = torch.zeros((10, 160, 160))
        annos = self.df[self.df['inputImage'] == img_rel_path]
        
        for _, row in annos.iterrows():
            cls = int(row['crater_classification'])
            ctx, cty = row['ellipseCenterX(px)'] * self.scale, row['ellipseCenterY(px)'] * self.scale
            ix, iy = int(ctx), int(cty)

            if 0 <= ix < 160 and 0 <= iy < 160:
                ma = row['ellipseSemimajor(px)'] * self.scale
                mi = row['ellipseSemiminor(px)'] * self.scale
                sigma = max(1.0, ma / 4.0)
                y, x = np.ogrid[:160, :160]
                gaussian = np.exp(-((x - ctx)**2 + (y - cty)**2) / (2 * sigma**2))
                gt[cls] = torch.max(gt[cls], torch.from_numpy(gaussian).float())
                
                gt[5, iy, ix] = np.log1p(ma) 
                gt[6, iy, ix] = np.log1p(mi)
                gt[7, iy, ix] = ctx - ix
                gt[8, iy, ix] = cty - iy
                gt[9, iy, ix] = np.deg2rad(row['ellipseRotation(deg)'])
        return img_t, gt

# --- 4. LOSS (15/15 BALANCE) ---
class AggressiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum')
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:5, :, :]
        pred_hm = torch.sigmoid(preds['hm'])
        
        pos = (gt_hm > 0.8).float()
        neg = (gt_hm <= 0.8).float()
        hm_loss = - (torch.log(pred_hm + 1e-6) * (1-pred_hm)**2 * pos + 
                     torch.log(1-pred_hm + 1e-6) * pred_hm**2 * (1-gt_hm)**4 * neg).sum() / (pos.sum() + 1e-4)

        mask = (gt_hm.max(dim=1, keepdim=True)[0] > 0.8)
        num = mask.float().sum() + 1e-4
        
        a_loss = self.smooth_l1(preds['axes'][mask.repeat(1,2,1,1)], target[:, 5:7, :, :][mask.repeat(1,2,1,1)]) / num
        o_loss = self.l1(preds['off'][mask.repeat(1,2,1,1)], target[:, 7:9, :, :][mask.repeat(1,2,1,1)]) / num
        r_loss = self.l1(preds['rot'][mask], target[:, 9:10, :, :][mask]) / num
        
        return (10.0 * hm_loss) + (5.0 * a_loss) + (1.0 * o_loss) + (1.0 * r_loss)

# --- 5. TRAIN LOOP ---
if __name__ == "__main__":
    model = SwinCraterV4().to(DEVICE)
    
   
    if os.path.exists(BEST_MODEL):
        print(f"🔄 Starting fresh using base best model: {BEST_MODEL}")
        model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))

    train_loader = DataLoader(CraterDataset(CSV_PATH, IMG_ROOT), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = AggressiveLoss()
    scaler = GradScaler()

    for epoch in range( EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, gts in loop:
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
        
    torch.save(model.state_dict(), f"{SAVE_PREFIX}.pth")
        #torch.save(model.state_dict(), BEST_MODEL) # Always update 'best' too
