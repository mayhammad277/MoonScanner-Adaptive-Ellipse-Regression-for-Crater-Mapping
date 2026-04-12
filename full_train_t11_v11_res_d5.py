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

# =========================
# 1. MODEL
# =========================
class SwinCraterNet(nn.Module):
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
        f = self.neck(x)
        return {
            "hm": self.hm(f),
            "axes": self.axes(f),
            "off": self.off(f),
            "rot": self.rot(f)
        }

# =========================
# 2. DATASET
# =========================
class CraterDataset(Dataset):
    def __init__(self, csv_file, img_root):
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna(subset=["crater_classification"])
        self.img_root = Path(img_root)

        self.images = self.df["inputImage"].unique()

        self.ORIG = 2592
        self.INP = 640
        self.HM = 160

        self.scale_img = self.INP / self.ORIG
        self.scale_hm = self.HM / self.INP

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = self.img_root / f"{img_id}.png"

        img = cv2.imread(str(img_path))
        if img is None:
            return self.__getitem__(np.random.randint(0, len(self)))

        img = cv2.resize(img, (self.INP, self.INP))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = (img - self.mean) / self.std

        gt = torch.zeros((10, self.HM, self.HM))

        ann = self.df[self.df["inputImage"] == img_id]

        for _, r in ann.iterrows():
            cls = int(r["crater_classification"])
            cx = r["ellipseCenterX(px)"]
            cy = r["ellipseCenterY(px)"]
            ma = r["ellipseSemimajor(px)"]
            mi = r["ellipseSemiminor(px)"]
            rot = r["ellipseRotation(deg)"]

            cx = cx * self.scale_img * self.scale_hm
            cy = cy * self.scale_img * self.scale_hm

            ix, iy = int(cx), int(cy)
            if not (0 <= ix < self.HM and 0 <= iy < self.HM):
                continue

            ma = ma * self.scale_img * self.scale_hm
            mi = mi * self.scale_img * self.scale_hm

            # 🔥 HARD PEAK (CRITICAL)
            gt[cls, iy, ix] = 1.0

            gt[5, iy, ix] = ma
            gt[6, iy, ix] = mi
            gt[7, iy, ix] = cx - ix
            gt[8, iy, ix] = cy - iy

            # 🔥 NORMALIZED ROTATION [-1,1]
            gt[9, iy, ix] = rot / 90.0

        return img, gt

# =========================
# 3. LOSS
# =========================
class CraterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="sum")

    def forward(self, preds, gt):
        hm_pred = torch.sigmoid(preds["hm"])
        hm_gt = gt[:,0:5]

        pos = hm_gt == 1
        neg = hm_gt < 1

        pos_loss = torch.log(hm_pred + 1e-6) * torch.pow(1 - hm_pred, 2) * pos
        neg_loss = torch.log(1 - hm_pred + 1e-6) * torch.pow(hm_pred, 2) * neg

        hm_loss = -(pos_loss.sum() + neg_loss.sum()) / (pos.sum() + 1e-4)

        # 🔥 CORRECT MASK
        reg_mask = (hm_gt.sum(dim=1, keepdim=True) > 0)

        num = reg_mask.sum() + 1e-4

        a_loss = self.l1(preds["axes"][reg_mask.repeat(1,2,1,1)], gt[:,5:7][reg_mask.repeat(1,2,1,1)]) / num
        o_loss = self.l1(preds["off"][reg_mask.repeat(1,2,1,1)], gt[:,7:9][reg_mask.repeat(1,2,1,1)]) / num
        r_loss = self.l1(preds["rot"][reg_mask], gt[:,9:10][reg_mask]) / num

        # 🔥 REBALANCED
        return (10.0 * hm_loss) + (8.0 * a_loss) + (2.0 * o_loss) + (6.0 * r_loss)

# =========================
# 4. TRAIN
# =========================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", DEVICE)

    dataset = CraterDataset("train-gt.csv", "./train")
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.15, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True, num_workers=4)
    val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False, num_workers=4)

    model = SwinCraterNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = CraterLoss()

    best = 1e9

    for epoch in range(50):
        model.train()
        tl = 0
        for imgs, gts in tqdm(train_loader, desc=f"Train {epoch}"):
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, gts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for imgs, gts in tqdm(val_loader, desc=f"Val {epoch}"):
                imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
                preds = model(imgs)
                vl += criterion(preds, gts).item()

        print(f"\nEpoch {epoch} | Train {tl/len(train_loader):.4f} | Val {vl/len(val_loader):.4f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(), "swin_crater_best.pth")
            print("⭐ SAVED BEST")

    print("DONE")

