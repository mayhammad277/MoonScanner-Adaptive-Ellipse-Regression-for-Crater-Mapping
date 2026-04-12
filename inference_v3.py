import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION ---
TEST_ROOT = "/home/bora3i/crater_challenge/test/"
MODEL_PATH = "crater_model_final.pth"
CSV_OUTPUT = "test_predictions_v3.csv"
VIS_DIR = "./final_visuals"

ORIGINAL_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUTPUT_RES = TRAIN_RES // STRIDE # 160
THRESHOLD = 0.35  # Confidence threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VIS_DIR, exist_ok=True)

# --- 2. MODEL ARCHITECTURE ---

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

# --- 3. CORRECTED POST-PROCESSING ---

def process_output(preds, threshold=0.35):
    # Apply sigmoid to heatmap to get probability 0.0 - 1.0
    hm = torch.sigmoid(preds['hm']).squeeze().cpu().numpy()  # Result: (160, 160)
    rad = preds['r'].squeeze().cpu().numpy()                # Result: (160, 160)
    off = preds['off'].squeeze().cpu().numpy()              # Result: (2, 160, 160)

    # Local Maxima search (3x3 window)
    hm_max = cv2.dilate(hm, np.ones((3,3)))
    peaks = (hm == hm_max) & (hm > threshold)
    y_idx, x_idx = np.where(peaks)
    
    scale = ORIGINAL_RES / OUTPUT_RES # 16.2
    detections = []

    for y, x in zip(y_idx, x_idx):
        confidence = hm[y, x]
        
        # FIX: rad is 2D (H, W), no channel index needed
        r_scaled = rad[y, x] 
        
        # off is 3D (Channel, H, W)
        fine_x = x + off[0, y, x]
        fine_y = y + off[1, y, x]
        
        # Scale back to original 2592x2592 resolution
        cx = fine_x * scale
        cy = fine_y * scale
        r = r_scaled * scale
        
        # Filter out negative or impossibly small radii
        if r > 2:
            detections.append([cx, cy, r, confidence])
    
    return detections

# --- 4. INFERENCE EXECUTION ---

def run_inference():
    model = CraterModel().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file {MODEL_PATH} not found!")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Walk through test folder
    img_list = []
    for root, _, files in os.walk(TEST_ROOT):
        for f in files:
            if f.endswith('.png'):
                img_list.append(os.path.join(root, f))

    print(f"🚀 Processing {len(img_list)} images on {DEVICE}...")
    
    all_results = []

    with torch.no_grad():
        for i, img_path in enumerate(tqdm(img_list)):
            raw_bgr = cv2.imread(img_path)
            if raw_bgr is None: continue
            
            # Preprocess
            img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            img_in = cv2.resize(img_rgb, (TRAIN_RES, TRAIN_RES))
            img_t = torch.from_numpy(img_in).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
            
            # Predict
            preds = model(img_t)
            detections = process_output(preds, THRESHOLD)
            
            # Store Results
            rel_path = os.path.relpath(img_path, TEST_ROOT)
            for d in detections:
                all_results.append([rel_path] + d)

            # Visualization (Save a few samples)
            if i < 15:
                vis_img = raw_bgr.copy()
                for cx, cy, r, conf in detections:
                    cv2.circle(vis_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                
                out_name = rel_path.replace(os.sep, "_")
                cv2.imwrite(os.path.join(VIS_DIR, f"pred_{out_name}"), vis_img)

    # Export
    df = pd.DataFrame(all_results, columns=['image_id', 'cx', 'cy', 'radius', 'confidence'])
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\n✅ Done! CSV saved to {CSV_OUTPUT}")

if __name__ == "__main__":
    run_inference()
