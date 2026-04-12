import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION ---
TEST_ROOT = "/home/bora3i/crater_challenge/test/"
MODEL_PATH = "/home/bora3i/crater_challenge/crater_model_best.pth"
VIS_OUTPUT_DIR = "./output_visuals"
ORIGINAL_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUTPUT_RES = TRAIN_RES // STRIDE # 160
THRESHOLD = 0.4  # Adjust based on performance
SAVE_VISUALS = True # Set to True to save images with circles
MAX_VIS_IMAGES = 20 # Limit how many images to save to save space
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)

# --- 2. MODEL ARCHITECTURE (Must match training exactly) ---

class CraterBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.features(x)

class CraterHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.hm = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
    def forward(self, x): return {'hm': self.hm(x), 'r': self.rad(x), 'off': self.off(x)}

class CraterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CraterBackbone()
        self.head = CraterHead()
    def forward(self, x): return self.head(self.backbone(x))

# --- 3. PROCESSING FUNCTIONS ---

def get_craters(preds, threshold=0.4):
    hm = torch.sigmoid(preds['hm']).squeeze().cpu().numpy()
    rad = preds['r'].squeeze().cpu().numpy()
    off = preds['off'].squeeze().cpu().numpy()

    # Peak finding
    hm_max = cv2.dilate(hm, np.ones((3,3)))
    peaks = (hm == hm_max) & (hm > threshold)
    y_coords, x_coords = np.where(peaks)
    
    results = []
    scale = ORIGINAL_RES / OUTPUT_RES # 16.2

    for y, x in zip(y_coords, x_coords):
        conf = hm[y, x]
        # Refine with offsets and scale to 2592
        final_x = (x + off[0, y, x]) * scale
        final_y = (y + off[1, y, x]) * scale
        final_r = rad[y, x] * scale
        results.append([final_x, final_y, final_r, conf])
    return results

def draw_visuals(img_path, detections, save_name):
    """Draws predicted circles and confidence scores on the 2592x2592 image"""
    img = cv2.imread(img_path)
    for det in detections:
        cx, cy, r, conf = det
        # Draw circle (Green)
        cv2.circle(img, (int(cx), int(cy)), int(r), (0, 255, 0), 3)
        # Put confidence score
        cv2.putText(img, f"{conf:.2f}", (int(cx), int(cy) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Save to output folder (maintains subfolder structure)
    out_path = os.path.join(VIS_OUTPUT_DIR, save_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

# --- 4. MAIN INFERENCE LOOP ---

def run_test():
    model = CraterModel().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
    model.eval()

    all_data = []
    image_paths = []
    for root, _, files in os.walk(TEST_ROOT):
        for f in files:
            if f.endswith('.png'):
                image_paths.append(os.path.join(root, f))

    print(f"Processing {len(image_paths)} images...")

    vis_count = 0
    with torch.no_grad():
        for path in tqdm(image_paths):
            # Load & Preprocess
            raw_img = cv2.imread(path)
            img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (TRAIN_RES, TRAIN_RES))
            img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0

            # Inference
            preds = model(img_tensor)
            detections = get_craters(preds, THRESHOLD)
            
            # Store results
            rel_path = os.path.relpath(path, TEST_ROOT)
            for d in detections:
                all_data.append([rel_path] + d)

            # Optional Visualization
            if SAVE_VISUALS and vis_count < MAX_VIS_IMAGES:
                draw_visuals(path, detections, rel_path)
                vis_count += 1

    # Save CSV
    df = pd.DataFrame(all_data, columns=['image_id', 'cx', 'cy', 'radius', 'confidence'])
    df.to_csv("test_predictions.csv", index=False)
    print(f"\nDone! CSV saved. Visuals saved to {VIS_OUTPUT_DIR}")

if __name__ == "__main__":
    run_test()
