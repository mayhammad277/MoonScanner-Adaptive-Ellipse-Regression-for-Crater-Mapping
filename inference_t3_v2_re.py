import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. MODEL ARCHITECTURE (Must match Boosted Training) ---
class SwinCraterV4(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize structure
        base_swin = swin_t(weights=None) 
        self.backbone = base_swin.features 
        
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Separate Heads - NO ReLU at the very end
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        x = self.backbone(x) 
        x = x.permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. NMS LOGIC ---
def apply_nms(detections, threshold=0.3):
    if not detections: return []
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[3], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        # Keep items if they are far enough away from the best detection
        remaining = []
        for item in detections:
            dist = np.sqrt((best[0]-item[0])**2 + (best[1]-item[1])**2)
            if dist > (best[2] + item[2]) * threshold:
                remaining.append(item)
        detections = remaining
    return keep

# --- 3. MAIN INFERENCE FUNCTION ---
@torch.inference_mode()
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = SwinCraterV4().to(device)
    model_path = "swin_boosted.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Normalization Constants (Swin Standard)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    test_root = "/home/bora3i/crater_challenge/test"
    output_csv = "submission_boosted.csv"
    
    # Recursive search for images
    image_paths = []
    for root, _, files in os.walk(test_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))

    print(f"🚀 Found {len(image_paths)} images. Starting processing...")

    results = []
    scale = 2592 / 160  # Scale 160 grid to 2592px original
    
    for path in tqdm(image_paths):
        raw = cv2.imread(path)
        if raw is None: continue
        
        # Preprocess
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (640, 640))
        inp_t = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        
        # Apply Normalization
        inp_t = (inp_t.to(device) - mean) / std
        inp_t = inp_t.unsqueeze(0)

        # Predict
        out = model(inp_t)
        hm = torch.sigmoid(out['hm']).squeeze().cpu().numpy()
        rad = out['r'].squeeze().cpu().numpy()
        off = out['off'].squeeze().cpu().numpy()

        # Local Peak Detection (Confidence > 0.3)
        hm_max = cv2.dilate(hm, np.ones((3, 3)))
        peaks = (hm == hm_max) & (hm > 0.3)
        y, x = np.where(peaks)

        img_detections = []
        for yj, xj in zip(y, x):
            # Back-project from grid to image pixels
            cx = (xj + off[0, yj, xj]) * scale
            cy = (yj + off[1, yj, xj]) * scale
            r = rad[yj, xj] * scale
            img_detections.append([cx, cy, r, hm[yj, xj]])

        # Apply NMS
        final_detections = apply_nms(img_detections)

        # Store results with relative path (to keep altitude folder info)
        rel_path = os.path.relpath(path, test_root)
        for d in final_detections:
            results.append([rel_path, d[0], d[1], d[2], d[3]])

    # Export to CSV
    df = pd.DataFrame(results, columns=['image_id', 'cx', 'cy', 'radius', 'confidence'])
    df.to_csv(output_csv, index=False)
    print(f"✅ Finished! Predictions saved to {output_csv}")
    print(f"Total craters found: {len(df)}")

if __name__ == "__main__":
    run_inference()
