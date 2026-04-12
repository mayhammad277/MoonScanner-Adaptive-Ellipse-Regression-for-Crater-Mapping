import os, cv2, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# MODEL: 320x320 OUTPUT
# =========================
class SwinCrater320(nn.Module):
    def __init__(self):
        super().__init__()
        base = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base.features
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, 4, 4), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 4), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.hm = nn.Conv2d(128, 5, 1)
        self.axes = nn.Conv2d(128, 2, 1)
        self.off = nn.Conv2d(128, 2, 1)
        self.rot = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous()
        f = self.neck(x)
        return {"hm": self.hm(f), "axes": self.axes(f), "off": self.off(f), "rot": self.rot(f)}

# =========================
# DATASET: SIZE-ACCURACY FILTERED
# =========================
class CraterDataset320(Dataset):
    def __init__(self, df, img_root, img_list):
        self.df = df
        self.img_root = Path(img_root)
        self.images = img_list
        self.scale = 320 / 2592
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        p = self.img_root / (img_id if img_id.endswith(".png") else img_id + ".png")
        img_raw = cv2.imread(str(p))
        if img_raw is None: return self.__getitem__(0)
        img = cv2.resize(img_raw, (640, 640))
        img = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - self.mean) / self.std
        
        gt = torch.zeros((10, 320, 320))
        annos = self.df[self.df['inputImage'] == img_id]
        for _, r in annos.iterrows():
            ma_s = r['ellipseSemimajor(px)'] * self.scale
            if ma_s < 1.5: continue # Ignore tiny craters to focus on scorable ones
            ctx, cty = r['ellipseCenterX(px)'] * self.scale, r['ellipseCenterY(px)'] * self.scale
            ix, iy = int(np.clip(ctx, 0, 319)), int(np.clip(cty, 0, 319))
            sigma = max(1.0, ma_s / 6.0)
            y, x = np.ogrid[:320, :320]
            g = np.exp(-((x - ctx)**2 + (y - cty)**2) / (2 * sigma**2))
            gt[int(r['crater_classification'])] = torch.max(gt[int(r['crater_classification'])], torch.from_numpy(g).float())
            gt[5, iy, ix] = np.log(ma_s + 1.0) # Log-space axes
            gt[6, iy, ix] = np.log(r['ellipseSemiminor(px)'] * self.scale + 1.0)
            gt[7, iy, ix], gt[8, iy, ix] = ctx - ix, cty - iy
            gt[9, iy, ix] = np.deg2rad(r['ellipseRotation(deg)'])
        return img, gt

# =========================
# LOSS: SIZE REGRESSION PRIORITY
# =========================
class SizePriorityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='sum', delta=1.0)
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:5]
        reg_mask = (gt_hm.max(dim=1, keepdim=True)[0] > 0.95)
        num = reg_mask.sum() + 1e-4
        # PRIORITY: Massive weight on Axis Huber Loss
        a_loss = self.huber(preds['axes'][reg_mask.repeat(1,2,1,1)], target[:, 5:7][reg_mask.repeat(1,2,1,1)]) / num
        o_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 7:9][reg_mask.repeat(1,2,1,1)]) / num
        hm_loss = nn.functional.binary_cross_entropy_with_logits(preds['hm'], gt_hm)
        return (15.0 * hm_loss) + (40.0 * a_loss) + (10.0 * o_loss)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gt_csv", required=True)
    args = parser.parse_args()
    DEVICE = torch.device("cuda")
    df = pd.read_csv(args.gt_csv).dropna(subset=['crater_classification'])
    t_imgs, v_imgs = train_test_split(df['inputImage'].unique(), test_size=0.1, random_state=42)
    train_loader = DataLoader(CraterDataset320(df, args.data_dir, t_imgs), batch_size=32, shuffle=True)
    val_loader = DataLoader(CraterDataset320(df, args.data_dir, v_imgs), batch_size=32, shuffle=False)
    model = SwinCrater320().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion, scaler = SizePriorityLoss(), GradScaler()
    best_mae = float('inf')
    for ep in range(70):
        model.train()
        for imgs, gts in tqdm(train_loader):
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            with autocast(): loss = criterion(model(imgs), gts)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        model.eval(); err, count = 0, 0
        with torch.no_grad():
            for imgs, gts in val_loader:
                out = model(imgs.to(DEVICE))
                p_ax, g_ax = torch.exp(out['axes']) - 1.0, torch.exp(gts[:, 5:7].to(DEVICE)) - 1.0
                mask = (gts[:, 0:5].max(dim=1)[0] > 0.95).to(DEVICE)
                if mask.any():
                    err += torch.abs(p_ax[mask.unsqueeze(1).repeat(1,2,1,1)] - g_ax[mask.unsqueeze(1).repeat(1,2,1,1)]).mean().item()
                    count += 1
        mae = err / count if count > 0 else 99
        print(f"Ep {ep+1} Size MAE: {mae:.5f}")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "best_size_model_14.pth")
            print("⭐ New Size Accuracy Record!")
if __name__ == "__main__": main()
