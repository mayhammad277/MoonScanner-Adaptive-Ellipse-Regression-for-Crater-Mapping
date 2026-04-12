import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION ---
TEST_ROOT =  "/home/bora3i/crater_challenge/test/"
MODEL_PATH = "crater_model_best.pth"
CSV_OUTPUT = "test_predictions_final.csv"
VIS_DIR = "./final_visuals"

ORIGINAL_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUTPUT_RES = TRAIN_RES // STRIDE # 160
THRESHOLD = 0.4  # Probability threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VIS_DIR, exist_ok=True)

# --- 2. EXACT MODEL ARCHITECTURE (Must match v3 train) ---

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

# --- 3. POST-PROCESSING LOGIC ---

def process_output(preds, threshold=0.4):
    hm = torch.sigmoid(preds['hm']).squeeze().cpu().numpy()
    rad = preds['r'].squeeze().cpu().numpy()
    off = preds['off'].squeeze().cpu().numpy()

    # Find peaks (3x3 Local Maxima)
    hm_max = cv2.dilate(hm, np.ones((3,3)))
    peaks = (hm == hm_max) & (hm > threshold)
    y_idx, x_idx = np.where(peaks)
    
    scale = ORIGINAL_RES / OUTPUT_RES # 2592 / 160 = 16.2
    detections = []

    for y, x in zip(y_idx, x_idx):
        confidence = hm[y, x]
        
        # Apply offsets from the 160x160 grid
        fine_x = x + off[0, y, x]
        fine_y = y + off[1, y, x]
        
        # Rescale to 2592x2592
        cx = fine_x * scale
        cy = fine_y * scale
        r = rad[0, y, x] * scale # radius is first channel of rad map
        
        detections.append([cx, cy, r, confidence])
    
    return detections

# --- 4. INFERENCE LOOP ---

def run_inference():
    # Load Model
    model = CraterModel().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file {MODEL_PATH} not found!")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Find all images in test-sample/altitudeXX/longitudeXX/
    img_list = []
    for root, _, files in os.walk(TEST_ROOT):
        for f in files:
            if f.endswith('.png'):
                img_list.append(os.path.join(root, f))

    print(f"Found {len(img_list)} images. Starting detection...")
    
    results_data = []

    with torch.no_grad():
        for i, img_path in enumerate(tqdm(img_list)):
            # 1. Load and Resize
            raw_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            img_in = cv2.resize(img_rgb, (TRAIN_RES, TRAIN_RES))
            
            # 2. To Tensor
            img_t = torch.from_numpy(img_in).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
            
            # 3. Model Forward
            preds = model(img_t)
            
            # 4. Get Coordinates
            detections = process_output(preds, THRESHOLD)
            
            # 5. Save results
            rel_path = os.path.relpath(img_path, TEST_ROOT)
            for d in detections:
                results_data.append([rel_path] + d)

            # 6. Visualization (Save first 20 images)
            if i < 20:
                vis_img = raw_bgr.copy()
                for cx, cy, r, conf in detections:
                    cv2.circle(vis_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"{conf:.2f}", (int(cx), int(cy)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Save with original subfolder structure
                save_path = os.path.join(VIS_DIR, rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_img)

    # Export CSV
    df = pd.DataFrame(results_data, columns=['image_id', 'cx', 'cy', 'radius', 'confidence'])
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"\n✅ Inference Complete!")
    print(f"📊 Results saved to: {CSV_OUTPUT}")
    print(f"🖼️ Visuals saved to: {VIS_DIR}")

if __name__ == "__main__":
    run_inference()
