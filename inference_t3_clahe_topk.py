import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- Configuration ---
MAX_DETECTIONS_PER_IMAGE = 50  # Limit to the 50 best craters per image
CONF_THRESHOLD = 0.25         # Ignore anything below this confidence
NMS_IOU_THRESHOLD = 0.3       # Overlap threshold for NMS

class CraterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), 
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        feat = self.backbone(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(3):
        img[:,:,i] = clahe.apply(img[:,:,i])
    return img

def apply_nms(detections, threshold=0.3):
    if not detections: return []
    detections = sorted(detections, key=lambda x: x[3], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for item in detections:
            # Circle-based distance check
            dist = np.sqrt((best[0]-item[0])**2 + (best[1]-item[1])**2)
            if dist > (best[2] + item[2]) * threshold:
                remaining.append(item)
        detections = remaining
    return keep

@torch.inference_mode()
def run_topk_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CraterModel().to(device)
    
    model_path = "custom_clahe_final.pth"
    if not os.path.exists(model_path):
        print(f"❌ Weights {model_path} not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Normalization stats
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

        # Find Local Peaks
        hm_max = cv2.dilate(hm, np.ones((3, 3)))
        y, x = np.where((hm == hm_max) & (hm > CONF_THRESHOLD))

        img_dets = []
        for yj, xj in zip(y, x):
            cx = (xj + off[0, yj, xj]) * scale
            cy = (yj + off[1, yj, xj]) * scale
            r = rad[yj, xj] * scale
            img_dets.append([cx, cy, r, hm[yj, xj]])

        # 1. Apply NMS to remove duplicates
        final_dets = apply_nms(img_dets, threshold=NMS_IOU_THRESHOLD)
        
        # 2. Apply Top-K (Sort and slice)
        final_dets = sorted(final_dets, key=lambda x: x[3], reverse=True)[:MAX_DETECTIONS_PER_IMAGE]
        
        rel_path = os.path.relpath(path, test_root)
        for d in final_dets:
            results.append([rel_path, d[0], d[1], d[2], d[3]])

    df = pd.DataFrame(results, columns=['image_id', 'cx', 'cy', 'radius', 'confidence'])
    df.to_csv("submission_topk.csv", index=False)
    print(f"✅ Created submission with {len(df)} total detections.")

if __name__ == "__main__":
    run_topk_inference()
