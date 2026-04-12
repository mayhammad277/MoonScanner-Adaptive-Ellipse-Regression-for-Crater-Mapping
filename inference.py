import os, sys, cv2, torch, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. MODEL RECONSTRUCTION (Identical to your Train script) ---
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

# --- 2. PREDICTOR CLASS ---
class CraterPredictor:
    def __init__(self, model_path, threshold=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinCraterMahantiV4().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.threshold = threshold
        
        # Standard Normalization from Swin/ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def apply_clahe(self, img):
        channels = cv2.split(img)
        return cv2.merge([self.clahe.apply(c) for c in channels])

    @torch.no_grad()
    def predict(self, img_path, img_id, do_classify):
        img_raw = cv2.imread(img_path)
        if img_raw is None: return []
        
        img_enhanced = self.apply_clahe(img_raw)
        img_res = cv2.resize(img_enhanced, (640, 640))
        img_t = (torch.from_numpy(img_res).permute(2, 0, 1).float().to(self.device) / 255.0 - self.mean) / self.std
        
        # --- 1. RUN MODEL (This defines 'preds') ---
        preds = self.model(img_t.unsqueeze(0))
        
        # --- 2. EXTRACT HEATMAP ---
        hm = torch.sigmoid(preds['hm'])
        hmax = torch.nn.functional.max_pool2d(hm, 3, stride=1, padding=1)
        hm = hm * (hmax == hm).float()

        # Coordinate scale fixed at 16.2
        coord_scale = 16.2 
        
        image_results = []
        for c in range(5):
            peaks = torch.nonzero(hm[0, c] > 0.3) 
            for loc in peaks:
                iy, ix = loc[0].item(), loc[1].item()
                
                # Center + Offset
                dx, dy = preds['off'][0, 0, iy, ix].item(), preds['off'][0, 1, iy, ix].item()
                real_x = (ix + dx) * coord_scale
                real_y = (iy + dy) * coord_scale
                
                # --- 3. CONSERVATIVE LINEAR SCALING ---
                # We use a smaller multiplier (45) + a small base offset (15).
                # This is "less vigorous" and prevents ellipses from expanding too far.
                raw_maj = abs(preds['axes'][0, 0, iy, ix].item())
                raw_min = abs(preds['axes'][0, 1, iy, ix].item())

                maj = (raw_maj * 45.0) + 15.0
                min_ax = (raw_min * 45.0) + 12.0

                # --- 4. TIGHT OVERLAP PREVENTION ---
                # Reducing the ceiling to 180px and the floor to 28px
                maj = min(max(maj, 28.0), 180.0)
                min_ax = min(max(min_ax, 20.0), 150.0)

                # CIRCULARITY SANITY (Tighten to 1.4 for overlap prevention)
                if maj / (min_ax + 1e-6) > 1.4:
                    min_ax = maj / 1.15

                if maj < min_ax: maj, min_ax = min_ax, maj
                rot = np.rad2deg(preds['rot'][0, 0, iy, ix].item())

                image_results.append({
                    'ellipseCenterX(px)': real_x, 'ellipseCenterY(px)': real_y,
                    'ellipseSemimajor(px)': maj, 'ellipseSemiminor(px)': min_ax,
                    'ellipseRotation(deg)': rot, 'inputImage': img_id,
                    'crater_classification': int(c) if do_classify else -1
                })
        return image_results
# --- 3. EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(1)
    DATA_DIR, OUTPUT_CSV = sys.argv[1], sys.argv[2]
    DO_CLASSIFY = "--classify" in sys.argv
    
    # Path inside your submission zip
    MODEL_PATH = "swin_crater_best.pth" 
    predictor = CraterPredictor(MODEL_PATH)
    
    all_detections = []
    image_paths = []
    
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".png"):
                full_p = os.path.join(root, f)
                # Format: folder/folder/filename (no .png)
                rel_p = os.path.relpath(full_p, DATA_DIR)
                img_id = os.path.splitext(rel_p)[0]
                image_paths.append((full_p, img_id))

    for full_p, img_id in tqdm(image_paths, desc="Final Inference"):
        res = predictor.predict(full_p, img_id, DO_CLASSIFY)
        if not res:
            all_detections.append({
                'ellipseCenterX(px)': -1, 'ellipseCenterY(px)': -1,
                'ellipseSemimajor(px)': -1, 'ellipseSemiminor(px)': -1,
                'ellipseRotation(deg)': -1, 'inputImage': img_id,
                'crater_classification': -1
            })
        else:
            all_detections.extend(res)

    df = pd.DataFrame(all_detections)
    cols = ['ellipseCenterX(px)', 'ellipseCenterY(px)', 'ellipseSemimajor(px)', 
            'ellipseSemiminor(px)', 'ellipseRotation(deg)', 'inputImage', 'crater_classification']
    df[cols].to_csv(OUTPUT_CSV, index=False)
