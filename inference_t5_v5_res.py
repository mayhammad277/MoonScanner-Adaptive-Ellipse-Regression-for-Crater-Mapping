import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. MODEL RECONSTRUCTION ---
class SwinCraterMahantiV4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t()
        self.backbone = base_swin.features 
        self.neck = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.hm = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 5, 1))
        self.axes = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 2, 1))
        self.off = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 2, 1))
        self.rot = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

# --- 2. THE INFERENCE ENGINE ---
class CraterPredictor:
    def __init__(self, model_path, threshold=0.15): # Lowered threshold to see "invisible" ellipses
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinCraterMahantiV4().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.threshold = threshold
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        self.classes = ["Fresh", "Degraded", "Intermediate", "Old", "Very Old"]
        self.colors = [(0,255,0), (255,255,0), (255,165,0), (0,0,255), (128,0,128)]

    def predict_and_visualize(self, img_path, rel_path, output_folder):
        img_raw = cv2.imread(img_path)
        if img_raw is None: return []
        
        h_orig, w_orig = img_raw.shape[:2]
        img_res = cv2.resize(img_raw, (640, 640))
        img_t = (torch.from_numpy(img_res).permute(2, 0, 1).float().to(self.device) / 255.0 - self.mean) / self.std
        
        with torch.no_grad():
            preds = self.model(img_t.unsqueeze(0))
            hm = torch.sigmoid(preds['hm'])
            
            # --- PEAK DETECTION (NMS) ---
            hmax = torch.nn.functional.max_pool2d(hm, 3, stride=1, padding=1)
            hm = hm * (hmax == hm).float()

        scale = w_orig / 160.0 
        image_results = []

        for c in range(5):
            peaks = torch.nonzero(hm[0, c] > self.threshold)
            for loc in peaks:
                iy, ix = loc[0].item(), loc[1].item()
                conf = hm[0, c, iy, ix].item()
                
                # Geometry Extraction
                dx, dy = preds['off'][0, 0, iy, ix].item(), preds['off'][0, 1, iy, ix].item()
                # Use abs() to ensure positive axes (fixes the "not showing" bug)
                maj = abs(preds['axes'][0, 0, iy, ix].item()) * scale
                min_ax = abs(preds['axes'][0, 1, iy, ix].item()) * scale
                rot = np.rad2deg(preds['rot'][0, 0, iy, ix].item())

                real_x, real_y = (ix + dx) * scale, (iy + dy) * scale

                # Store for CSV
                image_results.append({
                    'inputImage': rel_path, 'crater_classification': c,
                    'ellipseCenterX(px)': real_x, 'ellipseCenterY(px)': real_y,
                    'ellipseSemimajor(px)': maj, 'ellipseSemiminor(px)': min_ax,
                    'ellipseRotation(deg)': rot, 'confidence': conf
                })

                # Draw on original image
                cv2.ellipse(img_raw, (int(real_x), int(real_y)), (int(maj), int(min_ax)), 
                            rot, 0, 360, self.colors[c], 3)
                cv2.putText(img_raw, f"{self.classes[c]} {conf:.2f}", (int(real_x), int(real_y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[c], 2)

        # Save Preview
        if len(image_results) > 0:
            out_name = rel_path.replace("/", "_")
            cv2.imwrite(os.path.join(output_folder, f"view_{out_name}"), img_raw)
            
        return image_results

# --- 3. MAIN LOOP ---
if __name__ == "__main__":
    MODEL_P = "swin_crater_best.pth"
    TEST_DIR = "/home/bora3i/crater_challenge/test"
    PREVIEW_DIR = "previews_v15"
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    predictor = CraterPredictor(MODEL_P, threshold=0.65) # Start low to debug!
    
    image_paths = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(".png"):
                full_path = os.path.join(root, f)
                image_paths.append((full_path, os.path.relpath(full_path, TEST_DIR)))

    all_detections = []
    for full_p, rel_p in tqdm(image_paths, desc="Visualizing first 50"):
        res = predictor.predict_and_visualize(full_p, rel_p, PREVIEW_DIR)
        all_detections.extend(res)

    pd.DataFrame(all_detections).to_csv("submission_debug.csv", index=False)
    print(f"✅ Check the '{PREVIEW_DIR}' folder to see if ellipses are now visible!")
