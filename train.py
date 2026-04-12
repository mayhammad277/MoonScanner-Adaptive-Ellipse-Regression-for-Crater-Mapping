import os, cv2, torch, numpy as np, pandas as pd, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 1. ARCHITECTURE ---
class SwinCraterMahantiV4(nn.Module):
    def __init__(self):
        super().__init__()
        # Using Swin-T as per your training setup
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

# --- 2. DATASET ---
class LiveMahantiDataset(Dataset):
    def __init__(self, csv_file, img_root):
        self.df = pd.read_csv(csv_file).dropna(subset=['crater_classification'])
        self.img_root = Path(img_root)
        self.unique_images = self.df['inputImage'].unique()
        self.scale = 160 / 2592
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): 
        return len(self.unique_images) * 2 # Account for flips

    def __getitem__(self, idx):
        is_flip = idx >= len(self.unique_images)
        img_rel_path = self.unique_images[idx % len(self.unique_images)]
        
        full_path = self.img_root / (img_rel_path if img_rel_path.endswith('.png') else img_rel_path + ".png")
        img_raw = cv2.imread(str(full_path))
        if img_raw is None:
            return self.__getitem__(np.random.randint(0, len(self)))
            
        if is_flip:
            img_raw = cv2.flip(img_raw, 1)

        img_res = cv2.resize(img_raw, (640, 640))
        img_t = torch.from_numpy(img_res).permute(2,0,1).float() / 255.0
        img_t = (img_t - self.mean) / self.std

        gt = torch.zeros((10, 160, 160))
        annos = self.df[self.df['inputImage'] == img_rel_path]
        
        for _, row in annos.iterrows():
            cls = int(row['crater_classification'])
            cx_raw, rot_deg = row['ellipseCenterX(px)'], row['ellipseRotation(deg)']
            if is_flip:
                cx_raw = 2592 - cx_raw
                rot_deg = -rot_deg

            ctx, cty = cx_raw * self.scale, row['ellipseCenterY(px)'] * self.scale
            ix, iy = int(ctx), int(cty)

            if 0 <= ix < 160 and 0 <= iy < 160:
                ma_scaled = row['ellipseSemimajor(px)'] * self.scale
                sigma = max(1.5, ma_scaled / 3.0)
                y, x = np.ogrid[:160, :160]
                gaussian = np.exp(-((x - ctx)**2 + (y - cty)**2) / (2 * sigma**2))
                gt[cls] = torch.max(gt[cls], torch.from_numpy(gaussian).float())
                gt[5, iy, ix] = ma_scaled
                gt[6, iy, ix] = row['ellipseSemiminor(px)'] * self.scale
                gt[7, iy, ix] = ctx - ix
                gt[8, iy, ix] = cty - iy
                gt[9, iy, ix] = np.deg2rad(rot_deg)
        return img_t, gt

# --- 3. LOSS ---
class HardPeakLossV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:5, :, :]
        pred_hm = torch.sigmoid(preds['hm'])
        pos_mask = (gt_hm > 0.9).float()
        neg_mask = (gt_hm <= 0.9).float()
        
        pos_loss = torch.log(pred_hm + 1e-6) * torch.pow(1 - pred_hm, 2) * pos_mask
        neg_loss = torch.log(1 - pred_hm + 1e-6) * torch.pow(pred_hm, 2) * torch.pow(1 - gt_hm, 4) * neg_mask
        hm_loss = - (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + 1e-4)

        reg_mask = (gt_hm.max(dim=1, keepdim=True)[0] > 0.9)
        num_reg = reg_mask.float().sum() + 1e-4
        a_loss = self.l1(preds['axes'][reg_mask.repeat(1,2,1,1)], target[:, 5:7, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        o_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 7:9, :, :][reg_mask.repeat(1,2,1,1)]) / num_reg
        r_loss = self.l1(preds['rot'][reg_mask], target[:, 9:10, :, :][reg_mask]) / num_reg
        
        return (20.0 * hm_loss) + (0.1 * a_loss) + (0.1 * o_loss) + (0.1 * r_loss)

# --- 4. TRAIN SCRIPT ---
def train_model(train_dir, csv_file, model_save_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterMahantiV4().to(DEVICE)
    
    full_ds = LiveMahantiDataset(csv_file, train_dir)
    train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=0.15, random_state=42)
    
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=8, shuffle=False, num_workers=4)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = HardPeakLossV4()
    scaler = GradScaler()
    best_val_loss = float('inf')

    for epoch in range(70):
        model.train()
        for imgs, gts in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(imgs), gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs, gts = imgs.to(DEVICE), gts.to(DEVICE)
                loss = criterion(model(imgs), gts)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    # Standard competition format: train.sh calls python train.py [data_dir] [csv_file]
    TRAIN_DIR = sys.argv[1] if len(sys.argv) > 1 else "./train"
    CSV_PATH = sys.argv[2] if len(sys.argv) > 2 else "train-gt.csv"
    SAVE_PATH = "swin_crater_best.pth"
    
    train_model(TRAIN_DIR, CSV_PATH, SAVE_PATH)
