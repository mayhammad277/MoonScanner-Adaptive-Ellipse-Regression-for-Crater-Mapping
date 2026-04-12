import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torchvision.models import swin_t, Swin_T_Weights

# --- 1. ARCHITECTURE (Must match Training) ---
class SwinCraterMahantiV4(nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

def apply_nms(results, dist_threshold=0.5):
    """
    Suppresses overlapping craters. If two craters are closer than 
    (dist_threshold * semi-major axis), the weaker one is removed.
    """
    if not results: return []
    
    # Sort by confidence (highest first)
    results = sorted(results, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    while len(results) > 0:
        best = results.pop(0)
        keep.append(best)
        
        # Compare with remaining detections
        remaining = []
        for other in results:
            if other['inputImage'] != best['inputImage']:
                remaining.append(other)
                continue
                
            # Calculate Euclidean distance between centers
            dist = np.sqrt((best['ellipseCenterX(px)'] - other['ellipseCenterX(px)'])**2 + 
                           (best['ellipseCenterY(px)'] - other['ellipseCenterY(px)'])**2)
            
            # If they are too close, suppress based on radius
            # (If distance < 50% of the crater's size, it's a duplicate)
            if dist < (best['ellipseSemimajor(px)'] * dist_threshold):
                continue
            remaining.append(other)
        results = remaining
    return keep

def apply_crater_nms(results, iou_dist=0.5):
    """
    Suppresses redundant detections. If a detection is too close to a 
    more confident detection, it is removed.
    """
    if not results: return []
    
    # Sort by heatmap confidence (highest first)
    results = sorted(results, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    while len(results) > 0:
        best = results.pop(0)
        keep.append(best)
        
        remaining = []
        for other in results:
            if other['inputImage'] != best['inputImage']:
                remaining.append(other)
                continue
                
            # Distance between centers
            dist = np.sqrt((best['ellipseCenterX(px)'] - other['ellipseCenterX(px)'])**2 + 
                           (best['ellipseCenterY(px)'] - other['ellipseCenterY(px)'])**2)
            
            # If distance is less than 50% of the crater's own size, it's a duplicate
            if dist < (best['ellipseSemimajor(px)'] * iou_dist):
                continue
            remaining.append(other)
        results = remaining
    return keep

@torch.inference_mode()
def run_inference_final(weights, test_root, output_csv="submission_v4_nms.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterMahantiV4().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    scale = 2592 / 160 
    
    raw_detections = []

    # Recursive Image Path Finding
    image_paths = []
    for r, _, files in os.walk(test_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg')):
                image_paths.append(os.path.join(r, f))

    for path in tqdm(image_paths, desc="🔭 Predicting"):
        raw = cv2.imread(path)
        if raw is None: continue
        
        inp = cv2.resize(raw, (640, 640))
        inp_t = (torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0).to(device)/255.0 - mean)/std
        
        out = model(inp_t)
        hm = torch.sigmoid(out['hm']).squeeze(0).cpu().numpy()
        max_hm = np.max(hm, axis=0)
        cls_map = np.argmax(hm, axis=0)
        off = out['off'].squeeze(0).cpu().numpy()
        axes = out['axes'].squeeze(0).cpu().numpy()
        rot = out['rot'].squeeze(0).cpu().numpy()

        # Peak Finding with Dilate
        y, x = np.where((max_hm == cv2.dilate(max_hm, np.ones((3,3)))) & (max_hm > 0.30)) # Slightly lower threshold before NMS

        for yj, xj in zip(y, x):
            smj = float(axes[0, yj, xj] * scale)
            if smj < 1.0: continue # Filter out micro-noise
            
            raw_detections.append({
                'conf': float(max_hm[yj, xj]),
                'inputImage': os.path.basename(path),
                'ellipseCenterX(px)': float((xj + off[0, yj, xj]) * scale),
                'ellipseCenterY(px)': float((yj + off[1, yj, xj]) * scale),
                'ellipseSemimajor(px)': smj,
                'ellipseSemiminor(px)': float(axes[1, yj, xj] * scale),
                'ellipseRotation(deg)': float(np.rad2deg(rot[0, yj, xj])),
                'crater_classification': int(cls_map[yj, xj])
            })

    # Apply NMS to the gathered results
    print(f"Applying NMS to {len(raw_detections)} initial detections...")
    final_results = apply_crater_nms(raw_detections)
    
    # Save to CSV
    df = pd.DataFrame(final_results).drop(columns=['conf'])
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Saved {len(df)} craters to {output_csv}")

if __name__ == "__main__":
    run_inference_final("swin_mahanti_v4.pth", "/home/bora3i/crater_challenge/test")
