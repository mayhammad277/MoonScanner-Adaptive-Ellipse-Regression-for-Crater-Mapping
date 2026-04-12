import os, cv2, torch, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision.models import swin_t

# --- 1. MODEL DEFINITION ---
class SwinCraterMahantiV4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base_swin = swin_t()
        self.backbone = base_swin.features 
        self.neck = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8), 
            torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True)
        )
        self.hm = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 5, 1))
        self.axes = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 2, 1))
        self.off = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 2, 1))
        self.rot = torch.nn.Sequential(torch.nn.Conv2d(256, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous() 
        feat = self.neck(x)
        return {'hm': self.hm(feat), 'axes': self.axes(feat), 'off': self.off(feat), 'rot': self.rot(feat)}

# --- 2. INFERENCE & VISUALIZATION ENGINE ---
class CraterInferenceV4:
    def __init__(self, model_path, conf_threshold=0.45):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinCraterMahantiV4().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.threshold = conf_threshold
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        self.colors = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
        self.class_names = ["Fresh", "Degraded", "Intermediate", "Old", "Very Old"]

    def run_inference(self, img_path, rel_p):
        img_raw = cv2.imread(img_path)
        if img_raw is None: return []
        h_orig, w_orig = img_raw.shape[:2]
        
        # Preprocess
        chans = cv2.split(img_raw)
        enhanced = cv2.merge([self.clahe.apply(c) for c in chans])
        img_res = cv2.resize(enhanced, (640, 640))
        img_t = (torch.from_numpy(img_res).permute(2, 0, 1).float().to(self.device) / 255.0 - self.mean) / self.std
        
        with torch.no_grad():
            preds = self.model(img_t.unsqueeze(0))
            hm = torch.sigmoid(preds['hm'])
            hmax = torch.nn.functional.max_pool2d(hm, 3, stride=1, padding=1)
            hm = hm * (hmax == hm).float()

        scale_to_orig = w_orig / 160.0
        results = []

        for c in range(5):
            peaks = torch.nonzero(hm[0, c] > self.threshold)
            for loc in peaks:
                iy, ix = loc[0].item(), loc[1].item()
                results.append({
                    'inputImage': rel_p,
                    'crater_classification': c,
                    'ellipseCenterX(px)': (ix + preds['off'][0, 0, iy, ix].item()) * scale_to_orig,
                    'ellipseCenterY(px)': (iy + preds['off'][0, 1, iy, ix].item()) * scale_to_orig,
                    'ellipseSemimajor(px)': abs(preds['axes'][0, 0, iy, ix].item()) * scale_to_orig,
                    'ellipseSemiminor(px)': abs(preds['axes'][0, 1, iy, ix].item()) * scale_to_orig,
                    'ellipseRotation(deg)': np.rad2deg(preds['rot'][0, 0, iy, ix].item()),
                    'conf': hm[0, c, iy, ix].item()
                })
        return results, enhanced

    def visualize_detections(self, img_path, rel_p, viz_dir):
        results, viz_img = self.run_inference(img_path, rel_p)
        
        # Draw and Save
        for det in results:
            cx, cy = int(det['ellipseCenterX(px)']), int(det['ellipseCenterY(px)'])
            ma, mi = int(det['ellipseSemimajor(px)']), int(det['ellipseSemiminor(px)'])
            rot = int(det['ellipseRotation(deg)'])
            c = det['crater_classification']
            
            cv2.ellipse(viz_img, (cx, cy), (ma, mi), rot, 0, 360, self.colors[c], 2)
            label = f"{self.class_names[c]} {det['conf']:.2f}"
            cv2.putText(viz_img, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[c], 1)

        # Handle subfolders in VIZ_DIR
        out_path = os.path.join(viz_dir, rel_p)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, viz_img)
        return results

# --- 3. RUN BATCH ---
if __name__ == "__main__":
    MODEL_PATH = "v4_crater_best_model.pth"
    TEST_DIR = "./test"
    VIZ_DIR = "./previews_v6"
    
    predictor = CraterInferenceV4(MODEL_PATH)
    
    image_paths = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(".png"):
                full_path = os.path.join(root, f)
                image_paths.append((full_path, os.path.relpath(full_path, TEST_DIR)))

    all_detections = []
    for full_p, rel_p in tqdm(image_paths, desc="Detecting & Visualizing"):
        # We catch the return value from visualize_detections
        res = predictor.visualize_detections(full_p, rel_p, VIZ_DIR)
        all_detections.extend(res)

    # Final CSV Cleanup (Removing the 'conf' helper column for submission)
    df = pd.DataFrame(all_detections)
    if not df.empty:
        df = df.drop(columns=['conf'])
        df.to_csv("submission_v6.csv", index=False)
        print(f"✅ Created submission_v6.csv and saved {len(image_paths)} previews in {VIZ_DIR}")
    else:
        print("⚠️ No craters detected!")
