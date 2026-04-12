import os, torch, cv2, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
import torchvision.transforms as T
from tqdm import tqdm

class SwinCraterV6(nn.Module):
    def __init__(self, weights=Swin_T_Weights.DEFAULT):
        super().__init__()
        base = swin_t(weights=weights)
        self.backbone = base.features
        self.neck = nn.Sequential(nn.ConvTranspose2d(768, 256, 8, 8), nn.BatchNorm2d(256), nn.ReLU(True))
        self.hm = nn.Conv2d(256, 5, 1); self.axes = nn.Conv2d(256, 2, 1)
        self.off = nn.Conv2d(256, 2, 1); self.rot = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        f = self.neck(self.backbone(x).permute(0, 3, 1, 2).contiguous())
        return {"hm": self.hm(f), "axes": self.axes(f), "off": self.off(f), "rot": self.rot(f)}

class CraterDataset(Dataset):
    def __init__(self, csv, img_dir, augment=True):
        self.df = pd.read_csv(csv); self.img_dir = img_dir
        self.imgs = self.df['inputImage'].unique()
        self.augment = augment
        self.aug_pipe = T.Compose([
            T.ColorJitter(brightness=0.4, contrast=0.4)])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]; img_path = os.path.join(self.img_dir, f"{name}.png")
        img = cv2.imread(img_path)
        h, w = img.shape[:2]; target = np.zeros((10, 160, 160), dtype=np.float32)
        
        craters = self.df[self.df['inputImage'] == name]
        for _, r in craters.iterrows():
            # Skip invalid entries or rows with NaN coordinates
            if pd.isna(r['ellipseCenterX(px)']) or r['ellipseCenterX(px)'] == -1: continue
            
            gx, gy = r['ellipseCenterX(px)']*160/w, r['ellipseCenterY(px)']*160/h
            ix, iy = int(np.clip(gx, 0, 159)), int(np.clip(gy, 0, 159))
            
            # FIX: Handle NaN classification by defaulting to class 0
            if pd.isna(r['crater_classification']):
                cls_idx = 0
            else:
                cls_val = int(r['crater_classification'])
                cls_idx = max(0, min(4, cls_val if cls_val != -1 else 0))
            
            target[cls_idx, iy, ix] = 1.0
            target[5:7, iy, ix] = np.log(np.array([r['ellipseSemimajor(px)'], r['ellipseSemiminor(px)']]) + 1)
            target[7:9, iy, ix] = [gx-ix, gy-iy]
            target[9, iy, ix] = np.deg2rad(r['ellipseRotation(deg)'])
            
        img = torch.from_numpy(cv2.resize(img, (640, 640))).float().permute(2,0,1)/255.
        if self.augment: img = self.aug_pipe(img)
        return img, torch.from_numpy(target)

def train():
    dev = torch.device("cuda"); model = SwinCraterV6().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    def criterion(p, t):
        mask = (t[:, 5:6] > 0).float()
        l_hm = F.binary_cross_entropy_with_logits(p['hm'], t[:, :5])
        l_axes = (F.l1_loss(p['axes'], t[:, 5:7], reduction='none') * mask).sum() / (mask.sum() + 1e-4)
        l_off = (F.l1_loss(p['off'], t[:, 7:9], reduction='none') * mask).sum() / (mask.sum() + 1e-4)
        return l_hm + 2.0 * l_axes + 1.0 * l_off

    ds = CraterDataset("train-gt.csv", "train", augment=True)
    ld = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    for ep in range(1, 71):
        model.train(); tl = 0
        for x, y in tqdm(ld, desc=f"Epoch {ep}"):
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            out = model(x); loss = criterion(out, y)
            loss.backward(); opt.step(); tl += loss.item()
        torch.save(model.state_dict(), "swin_crater_best.pth")

if __name__ == "__main__": train()
