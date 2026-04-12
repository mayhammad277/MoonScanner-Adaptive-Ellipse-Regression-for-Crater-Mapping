import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. THE REFINED MODEL ---
class SwinCraterV3(nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t() 
        self.backbone = base_swin.features 
        
        # Neck: Upsample from 20x20 to 160x160
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.rad = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def forward(self, x):
        # x input: [B, 3, 640, 640]
        x = self.backbone(x) 
        # Swin Output is [B, 20, 20, 768] -> Change to [B, 768, 20, 20]
        x = x.permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'r': self.rad(feat), 'off': self.off(feat)}

# --- 2. THE INFERENCE ENGINE ---
@torch.inference_mode()
def run_final_check():
    device = torch.device("cpu")
    model = SwinCraterV3().to(device)
    
    # PATHS
    test_root = "/home/bora3i/crater_challenge/test"
    model_path = "/home/bora3i/crater_challenge/swin_v4_final.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Recursive Image Search
    image_paths = []
    for r, d, f in os.walk(test_root):
        for file in f:
            if file.lower().endswith(('.png', '.jpg')):
                image_paths.append(os.path.join(r, file))

    print(f"🚀 Found {len(image_paths)} images. Processing...")

    results = []
    global_max = 0.0

    for path in tqdm(image_paths[:50]): # Check first 50 images
        raw = cv2.imread(path)
        if raw is None: continue
        
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (640, 640))
        inp_t = torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0) / 255.0
        
        preds = model(inp_t)
        
        # ACTIVATE HEATMAP
        hm = torch.sigmoid(preds['hm']).squeeze().numpy()
        
        # DIAGNOSTIC: Check max intensity
        img_max = np.max(hm)
        if img_max > global_max: global_max = img_max

        # Save one heatmap as a file to see if it's "noisy" or "empty"
        if img_max > 0.0001 and not os.path.exists("debug_hm.png"):
            normalized_hm = (hm * 255).astype(np.uint8)
            cv2.imwrite("debug_hm.png", cv2.applyColorMap(normalized_hm, cv2.COLORMAP_JET))

        # Peak Finding with very low threshold for debugging
        threshold = 0.1 
        hm_max = cv2.dilate(hm, np.ones((3,3)))
        peaks = (hm == hm_max) & (hm > threshold)
        y, x = np.where(peaks)
        
        rad = preds['r'].squeeze().numpy()
        off = preds['off'].squeeze().numpy()
        scale = 2592 / 160

        for yj, xj in zip(y, x):
            results.append([os.path.basename(path), (xj + off[0,yj,xj])*scale, (yj + off[1,yj,xj])*scale, rad[yj,xj]*scale, hm[yj,xj]])

    print(f"\n📊 FINAL RESULTS:")
    print(f"Max confidence found: {global_max:.6f}")
    if results:
        df = pd.DataFrame(results, columns=['image_id','cx','cy','radius','confidence'])
        df.to_csv("final_submission.csv", index=False)
        print(f"✅ Created final_submission.csv with {len(df)} rows.")
    else:
        print("❌ Still no predictions. Your model weights might be corrupted or the head was never trained.")

if __name__ == "__main__":
    run_final_check()
