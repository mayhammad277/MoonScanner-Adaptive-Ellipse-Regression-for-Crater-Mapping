import torch, cv2, os, numpy as np, pandas as pd, re
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. SETTINGS ---
MODEL_PATH = "swin_v4_ft.pth"
TEST_DIR = "/home/bora3i/crater_challenge/test"
OUT_VIS_DIR = "./visualizations_clahe"
CSV_OUT = "submission_ft_clahe.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. ARCHITECTURE ---
class SwinCraterV4(nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t()
        self.backbone = base_swin.features 
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.hm = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        feat = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(feat)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

def pool_nms(hm, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm, (kernel, kernel), stride=1, padding=pad)
    return hm * (hmax == hm).float()

def apply_clahe(img):
    """ Enhances local contrast of the image using CLAHE """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

@torch.no_grad()
def run_inference(model, img_info):
    full_path, rel_path = img_info
    orig = cv2.imread(str(full_path))
    if orig is None: return []
    
    # --- CLAHE ENHANCEMENT ---
    enhanced = apply_clahe(orig)
    
    h_orig, w_orig = orig.shape[:2]
    img_res = cv2.resize(enhanced, (640, 640))
    img_t = (torch.from_numpy(img_res).permute(2,0,1).float() / 255.0 - 
             torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    out = model(img_t.unsqueeze(0).to(DEVICE))
    hm = pool_nms(torch.sigmoid(out['hm']))[0].cpu().numpy()
    axes_map = out['axes'][0].cpu().numpy()
    off_map = out['off'][0].cpu().numpy()
    rot_map = out['rot'][0].cpu().numpy()

    scale = w_orig / 160
    image_results = []
    thresh = 0.08 

    for c in range(5):
        y_coords, x_coords = np.where(hm[c] > thresh)
        for iy, ix in zip(y_coords, x_coords):
            real_x = (ix + off_map[0, iy, ix]) * scale
            real_y = (iy + off_map[1, iy, ix]) * scale
            maj = np.expm1(np.clip(axes_map[0, iy, ix], 0, 10)) * scale
            min_ax = np.expm1(np.clip(axes_map[1, iy, ix], 0, 10)) * scale
            rot = np.rad2deg(rot_map[0, iy, ix])

            if maj < 1.0 or not np.isfinite(maj): continue

            image_results.append({
                "ellipseCenterX(px)": float(real_x),
                "ellipseCenterY(px)": float(real_y),
                "ellipseSemimajor(px)": float(maj),
                "ellipseSemiminor(px)": float(min_ax),
                "ellipseRotation(deg)": float(rot),
                "inputImage": rel_path,
                "crater_classification": int(c)
            })

            # Draw on original (unprocessed) image for true visual check
            color_map = [(0,255,0), (255,255,0), (0,255,255), (255,165,0), (0,0,255)]
            try:
                cv2.ellipse(orig, (int(round(real_x)), int(round(real_y))), 
                            (int(round(maj)), int(round(min_ax))), 
                            float(rot), 0, 360, color_map[c], 2)
            except: continue

    save_path = Path(OUT_VIS_DIR) / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), orig)
    return image_results

if __name__ == "__main__":
    net = SwinCraterV4().to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()
    
    all_imgs = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_p = os.path.join(root, f)
                all_imgs.append((full_p, os.path.relpath(full_p, TEST_DIR)))

    final_csv_rows = []
    for info in tqdm(all_imgs, desc="CLAHE Inference"):
        final_csv_rows.extend(run_inference(net, info))

    pd.DataFrame(final_csv_rows).to_csv(CSV_OUT, index=False)
    print(f"🏁 Finished! Results saved to {CSV_OUT}")
