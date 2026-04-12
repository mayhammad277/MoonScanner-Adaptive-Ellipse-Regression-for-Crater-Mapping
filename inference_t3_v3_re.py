import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. MODEL ARCHITECTURE ---
class SwinCraterV4(nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t(weights=None) 
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        x = self.backbone(x) 
        x = x.permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. ADVANCED NMS & TOOLS ---
def apply_circle_nms(detections, iou_thresh=0.3):
    """Prevents overlapping red clusters by checking area intersection."""
    if not detections: return []
    # Sort by confidence descending
    detections = sorted(detections, key=lambda x: x[3], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for item in detections:
            d = np.sqrt((best[0]-item[0])**2 + (best[1]-item[1])**2)
            r1, r2 = best[2], item[2]
            if d >= r1 + r2:
                remaining.append(item)
                continue
            overlap_dist = max(0, r1 + r2 - d)
            # Intersection ratio relative to smaller crater
            overlap_ratio = overlap_dist / (2 * min(r1, r2))
            if overlap_ratio < iou_thresh:
                remaining.append(item)
        detections = remaining
    return keep

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(3):
        img[:,:,i] = clahe.apply(img[:,:,i])
    return img

# --- 3. INFERENCE ENGINE ---
@torch.inference_mode()
def run_final_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterV4().to(device)
    
    model_path = "swin_boosted.pth"
    if not os.path.exists(model_path):
        print(f"❌ Weights {model_path} not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ImageNet stats for Swin
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    test_root = "/home/bora3i/crater_challenge/test"
    image_paths = [os.path.join(r, f) for r, _, files in os.walk(test_root) 
                   for f in files if f.lower().endswith(('.png', '.jpg'))]

    results = []
    scale = 2592 / 160

    for path in tqdm(image_paths):
        raw = cv2.imread(path)
        if raw is None: continue
        
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        enhanced = apply_clahe(rgb)
        
        inp = cv2.resize(enhanced, (640, 640))
        inp_t = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        inp_t = (inp_t.to(device) - mean) / std
        inp_t = inp_t.unsqueeze(0)

        out = model(inp_t)
        hm = torch.sigmoid(out['hm']).squeeze().cpu().numpy()
        rad = out['r'].squeeze().cpu().numpy()
        off = out['off'].squeeze().cpu().numpy()

        # Find Peaks
        hm_max = cv2.dilate(hm, np.ones((3, 3)))
        y, x = np.where((hm == hm_max) & (hm > 0.2))

        img_dets = []
        for yj, xj in zip(y, x):
            cx = (xj + off[0, yj, xj]) * scale
            cy = (yj + off[1, yj, xj]) * scale
            r = rad[yj, xj] * scale
            img_dets.append([cx, cy, r, hm[yj, xj]])

        # Apply Circle-NMS and Top-50 filter
        final_dets = apply_circle_nms(img_dets, iou_thresh=0.3)
        final_dets = sorted(final_dets, key=lambda x: x[3], reverse=True)[:50]
        
        rel_path = os.path.relpath(path, test_root)
        for d in final_dets:
            # Type casting to standard Python types avoids the Pandas/Numpy 2.0 error
            results.append({
                'image_id': str(rel_path),
                'cx': float(d[0]),
                'cy': float(d[1]),
                'radius': float(d[2]),
                'confidence': float(d[3])
            })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("submission_pro_final.csv", index=False)
        print(f"✅ Created submission with {len(df)} total detections.")
        print(f"📈 High confidence (>0.7): {len(df[df['confidence'] > 0.7])}")
    else:
        print("⚠️ No detections found.")

if __name__ == "__main__":
    run_final_inference()
