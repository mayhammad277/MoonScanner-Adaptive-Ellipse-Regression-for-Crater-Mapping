import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
TEST_ROOT = "/home/bora3i/crater_challenge/test/"
MODEL_PATH = "/home/bora3i/crater_challenge/swin_v4_final.pth"
CSV_OUTPUT = "test_predictions_v3_2.csv"
VIS_DIR = "./inference_results"

ORIGINAL_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUTPUT_RES = TRAIN_RES // STRIDE 
THRESHOLD = 0.30
NMS_THRESH = 0.5  # If centers are closer than 0.5 * radius, suppress one.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VIS_DIR, exist_ok=True)

# --- 1. NMS FUNCTION ---

def apply_nms(detections, dist_thresh=0.5):
    """
    detections: list of [cx, cy, r, conf]
    dist_thresh: percentage of radius to consider as overlap
    """
    if len(detections) == 0: return []
    
    # Sort by confidence descending
    detections = sorted(detections, key=lambda x: x[3], reverse=True)
    keep = []
    
    while len(detections) > 0:
        best = detections.pop(0)
        keep.append(best)
        
        remaining = []
        for feat in detections:
            # Calculate Euclidean distance between centers
            dist = np.sqrt((best[0] - feat[0])**2 + (best[1] - feat[1])**2)
            
            # If distance is greater than threshold, keep it for next iteration
            # Otherwise, it's suppressed (overlapping with 'best')
            if dist > (best[2] * dist_thresh):
                remaining.append(feat)
        detections = remaining
        
    return keep

# --- 2. MODEL DEFINITION ---

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

# --- 3. INFERENCE PROCESS ---

def process_output(preds, threshold=0.30):
    hm = torch.sigmoid(preds['hm']).squeeze().cpu().numpy()
    rad = preds['r'].squeeze().cpu().numpy()
    off = preds['off'].squeeze().cpu().numpy()

    # Peak detection
    hm_max = cv2.dilate(hm, np.ones((3,3)))
    peaks = (hm == hm_max) & (hm > threshold)
    y_idx, x_idx = np.where(peaks)
    
    scale = ORIGINAL_RES / OUTPUT_RES
    raw_detections = []

    for y, x in zip(y_idx, x_idx):
        conf = hm[y, x]
        fine_x, fine_y = x + off[0, y, x], y + off[1, y, x]
        cx, cy = fine_x * scale, fine_y * scale
        r = rad[y, x] * scale
        
        if r > 3:
            raw_detections.append([cx, cy, r, conf])
    
    # Apply NMS to clean up overlapping circles
    return apply_nms(raw_detections, NMS_THRESH)

def run_inference():
    model = CraterModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_images = []
    for root, _, files in os.walk(TEST_ROOT):
        for f in files:
            if f.endswith('.png'):
                test_images.append(os.path.join(root, f))

    results = []
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(test_images)):
            bgr = cv2.imread(img_path)
            if bgr is None: continue
            
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(rgb, (TRAIN_RES, TRAIN_RES))
            inp_t = torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
            
            preds = model(inp_t)
            dets = process_output(preds, THRESHOLD)
            
            rel_path = os.path.relpath(img_path, TEST_ROOT)
            for d in dets:
                results.append([rel_path] + d)

            if i < 10:
                vis = bgr.copy()
                for cx, cy, r, conf in dets:
                    cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(VIS_DIR, f"nms_{rel_path.replace(os.sep, '_')}"), vis)

    pd.DataFrame(results, columns=['image_id', 'cx', 'cy', 'radius', 'confidence']).to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Inference with NMS complete. Results: {CSV_OUTPUT}")

if __name__ == "__main__":
    run_inference()
