import torch
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torchvision.models import swin_t
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
# --- NMS LOGIC ---
def apply_nms(detections, iou_thresh=0.3):
    if not detections: return []
    # Sort by score (index 3)
    detections = sorted(detections, key=lambda x: x[3], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for item in detections:
            # Circle overlap check: distance between centers
            dist = np.sqrt((best[0]-item[0])**2 + (best[1]-item[1])**2)
            # If distance is too small relative to size, it's a duplicate
            if dist > (best[2] + item[2]) * iou_thresh:
                remaining.append(item)
        detections = remaining
    return keep

@torch.inference_mode()
def run_inference():
    device = torch.device("cpu")
    model = SwinCraterV3() # Use architecture from above
    model.load_state_dict(torch.load("/home/bora3i/crater_challenge/swin_v4_final.pth", map_location=device))
    model.eval()

    test_dir = "./test"
    scale = 2592 / 160
    results = []

    for img_name in tqdm(os.listdir(test_dir)):
        if not img_name.endswith('.png'): continue
        img = cv2.imread(os.path.join(test_dir, img_name))
        inp = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640))
        inp_t = torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0) / 255.0

        out = model(inp_t)
        hm = torch.sigmoid(out['hm']).squeeze().numpy()
        rad = out['r'].squeeze().numpy()
        off = out['off'].squeeze().numpy()

        # Find Peaks
        peaks = (hm == cv2.dilate(hm, np.ones((3,3)))) & (hm > 0.3)
        y, x = np.where(peaks)
        
        raw_dets = []
        for yj, xj in zip(y, x):
            cx = (xj + off[0, yj, xj]) * scale
            cy = (yj + off[1, yj, xj]) * scale
            r = rad[yj, xj] * scale
            raw_dets.append([cx, cy, r, hm[yj, xj]])

        # Apply NMS
        final_dets = apply_nms(raw_dets)
        for d in final_dets:
            results.append([img_name] + list(d))

    pd.DataFrame(results, columns=['image_id','cx','cy','radius','confidence']).to_csv("submission.csv", index=False)

if __name__ == "__main__":
    run_inference()
