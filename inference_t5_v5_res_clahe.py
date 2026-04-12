import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. MODEL RECONSTRUCTION (Same as your training) ---
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

# --- 2. THE CLAHE PREDICTOR ---
class CraterPredictor:
    def __init__(self, model_path, threshold=0.15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinCraterMahantiV4().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.threshold = threshold
        
        # CLAHE Setup
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Standard Stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        self.colors = [(0,255,0), (255,255,0), (255,165,0), (0,0,255), (128,0,128)]
        self.classes = ["Fresh", "Degraded", "Intermediate", "Old", "Very Old"]

    def apply_clahe(self, img):
        # Apply to each channel for BGR images
        channels = cv2.split(img)
        out_channels = []
        for chan in channels:
            out_channels.append(self.clahe.apply(chan))
        return cv2.merge(out_channels)

    def predict_and_visualize(self, img_path, rel_path, output_folder):
        img_raw = cv2.imread(img_path)
        if img_raw is None: return []
        
        h_orig, w_orig = img_raw.shape[:2]
        
        # 1. ENHANCE CONTRAST
        img_enhanced = self.apply_clahe(img_raw)
        
        # 2. PREPROCESS
        img_res = cv2.resize(img_enhanced, (640, 640))
        img_t = (torch.from_numpy(img_res).permute(2, 0, 1).float().to(self.device) / 255.0 - self.mean) / self.std
        
        with torch.no_grad():
            preds = self.model(img_t.unsqueeze(0))
            hm = torch.sigmoid(preds['hm'])
            hmax = torch.nn.functional.max_pool2d(hm, 3, stride=1, padding=1)
            hm = hm * (hmax == hm).float()

        scale = w_orig / 160.0 
        image_results = []

        # We draw on the ENHANCED image so you can see why the model made its choice
        for c in range(5):
            peaks = torch.nonzero(hm[0, c] > self.threshold)
            for loc in peaks:
                iy, ix = loc[0].item(), loc[1].item()
                conf = hm[0, c, iy, ix].item()
                
                dx, dy = preds['off'][0, 0, iy, ix].item(), preds['off'][0, 1, iy, ix].item()
                maj = abs(preds['axes'][0, 0, iy, ix].item()) * scale
                min_ax = abs(preds['axes'][0, 1, iy, ix].item()) * scale
                rot = np.rad2deg(preds['rot'][0, 0, iy, ix].item())
                real_x, real_y = (ix + dx) * scale, (iy + dy) * scale

                image_results.append({
                "ellipseCenterX(px)": real_x,
                "ellipseCenterY(px)": real_y,
                "ellipseSemimajor(px)": maj,
                 "ellipseSemiminor(px)": min_ax,
                 "ellipseRotation(deg)": rot,
                  "inputImage": rel_path,
                  "crater_classification": c
                   })
                cv2.ellipse(img_enhanced, (int(real_x), int(real_y)), (int(maj), int(min_ax)), 
                            rot, 0, 360, self.colors[c], 3)
                cv2.putText(img_enhanced, f"{self.classes[c]} {conf:.2f}", (int(real_x), int(real_y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[c], 2)

        if len(image_results) > 0:
            out_name = rel_path.replace("/", "_").replace("\\", "_")
            cv2.imwrite(os.path.join(output_folder, f"clahe_{out_name}"), img_enhanced)
            
        return image_results

if __name__ == "__main__":
    MODEL_P = "/home/bora3i/crater_challenge/swin_crater_best.pth"
    TEST_DIR = "/home/bora3i/crater_challenge/test/"
    PREVIEW_DIR = "./previews_clahe"
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    predictor = CraterPredictor(MODEL_P, threshold=0.65)
    
    # Get image list...
    image_paths = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(".png"):
                full_path = os.path.join(root, f)
                image_paths.append((full_path, os.path.relpath(full_path, TEST_DIR)))

    all_detections = []
    for full_p, rel_p in tqdm(image_paths, desc="Processing with CLAHE"):
        res = predictor.predict_and_visualize(full_p, rel_p, PREVIEW_DIR)
        all_detections.extend(res)

    pd.DataFrame(all_detections).to_csv("submission_clahe_tr.csv", index=False)
    print(f"✅ Previews saved in {PREVIEW_DIR}. Check for visible ellipses now!")
