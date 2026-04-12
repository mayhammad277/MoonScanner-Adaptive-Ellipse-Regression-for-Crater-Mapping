import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast


# =========================
# MODEL
# =========================
class SwinCraterMahantiV5(nn.Module):
    def __init__(self):
        super().__init__()
        base = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base.features

        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.hm   = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off  = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot  = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0,3,1,2).contiguous()
        feat = self.neck(x)
        return {
            "hm": self.hm(feat),
            "axes": self.axes(feat),
            "off": self.off(feat),
            "rot": self.rot(feat)
        }


# =========================
# DATASET
# =========================
class LiveMahantiDataset(Dataset):
    def __init__(self, csv_file, img_root):
        self.df = pd.read_csv(csv_file).dropna(subset=['crater_classification'])
        self.root = Path(img_root)
        self.images = self.df['inputImage'].unique()

        self.ORIG = 2592
        self.INP  = 640
        self.HM   = 160

        self.s_img = self.INP / self.ORIG
        self.s_hm  = self.HM / self.INP

        self.mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    def __len__(self):
        return len(self.images) * 2

    def __getitem__(self, idx):
        flip = idx >= len(self.images)
        name = self.images[idx % len(self.images)]

        if not name.endswith(".png"):
            name = name + ".png"

        img = cv2.imread(str(self.root / name))
        if img is None:
            return self.__getitem__(np.random.randint(0, len(self)))

        if flip:
            img = cv2.flip(img, 1)

        img = cv2.resize(img, (self.INP, self.INP))
        img = torch.from_numpy(img).permute(2,0,1).float()/255.0
        img = (img - self.mean) / self.std

        gt = torch.zeros((10, self.HM, self.HM))
        rows = self.df[self.df['inputImage'] == name.replace(".png","")]

        for _, r in rows.iterrows():
            cls = int(r['crater_classification'])
            cx, cy = r['ellipseCenterX(px)'], r['ellipseCenterY(px)']
            rot = r['ellipseRotation(deg)']

            if flip:
                cx = self.ORIG - cx
                rot = 180 - rot
                if rot > 90: rot -= 180

            cx = cx * self.s_img * self.s_hm
            cy = cy * self.s_img * self.s_hm

            ix, iy = int(cx), int(cy)
            if not (0 <= ix < self.HM and 0 <= iy < self.HM):
                continue

            ma = r['ellipseSemimajor(px)'] * self.s_img * self.s_hm
            mi = r['ellipseSemiminor(px)'] * self.s_img * self.s_hm

            sigma = max(2.5, ma / 2.5)  # STRONGER GAUSSIAN (recall boost)
            y, x = np.ogrid[:self.HM, :self.HM]
            g = np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))
            gt[cls] = torch.max(gt[cls], torch.from_numpy(g).float())

            gt[5, iy, ix] = ma
            gt[6, iy, ix] = mi
            gt[7, iy, ix] = cx - ix
            gt[8, iy, ix] = cy - iy
            gt[9, iy, ix] = np.deg2rad(rot)

        return img, gt


# =========================
# LOSS (RECALL BIASED)
# =========================
class RecallPeakLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, p, t):
        gt = t[:,0:5]
        hm = torch.sigmoid(p['hm'])

        pos = (gt > 0.7).float()
        neg = (gt <= 0.7).float()

        pos_loss = torch.log(hm+1e-6) * torch.pow(1-hm,2) * pos
        neg_loss = torch.log(1-hm+1e-6) * torch.pow(hm,2) * torch.pow(1-gt,4) * neg

        hm_loss = -(pos_loss.sum() + 0.5*neg_loss.sum()) / (pos.sum()+1e-4)

        mask = (gt.max(dim=1,keepdim=True)[0] > 0.7)
        n = mask.float().sum()+1e-4

        a = self.l1(p['axes'][mask.repeat(1,2,1,1)], t[:,5:7][mask.repeat(1,2,1,1)]) / n
        o = self.l1(p['off'][mask.repeat(1,2,1,1)],  t[:,7:9][mask.repeat(1,2,1,1)]) / n

        return 18*hm_loss + 8*a + 2*o


# =========================
# TRAIN
# =========================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinCraterMahantiV5().to(device)
    ds = LiveMahantiDataset("train-gt.csv", "./train")

    tr_idx, va_idx = train_test_split(range(len(ds)), test_size=0.15, random_state=42)

    tr_loader = DataLoader(Subset(ds, tr_idx), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(Subset(ds, va_idx), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    crit = RecallPeakLoss()
    scaler = GradScaler()

    best = 1e9

    for e in range(60):
        model.train()
        for x,y in tqdm(tr_loader, desc=f"Epoch {e+1} Train"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        val = 0
        with torch.no_grad():
            for x,y in va_loader:
                x,y = x.to(device), y.to(device)
                val += crit(model(x), y).item()

        val /= len(va_loader)
        print(f"Epoch {e+1} Val: {val:.4f}")

        if val < best:
            best = val
            torch.save(model.state_dict(), "swin_crater_best.pth")
            print("⭐ saved")

