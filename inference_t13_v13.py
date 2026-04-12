import torch, cv2, numpy as np, pandas as pd, os
from pathlib import Path

class SwinCraterV6(nn.Module):
    def __init__(self, weights=Swin_T_Weights.DEFAULT):
        super().__init__()
        base = swin_t(weights=weights)
        self.backbone = base.features
        self.neck = nn.Sequential(nn.ConvTranspose2d(768, 256, 8, 8), nn.BatchNorm2d(256), nn.ReLU(True))
        self.hm = nn.Conv2d(256, 5, 1); self.axes = nn.Conv2d(256, 2, 1)
        self.off = nn.Conv2d(256, 2, 1); self.rot = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        f = self.neck(self.backbone(x).permute(0, 3, 1, 2).contiguous())
        return {"hm": self.hm(f), "axes": self.axes(f), "off": self.off(f), "rot": self.rot(f)}

class CraterDataset(Dataset):
    def __init__(self, csv, img_dir, augment=False):
        self.df = pd.read_csv(csv); self.img_dir = img_dir
        self.imgs = self.df['inputImage'].unique()
        self.augment = augment
        # Brightness/Contrast augmentation to handle solar incidence variation
        self.aug_pipe = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3),


    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]; img_path = f"{self.img_dir}/{name}.png"
        img = cv2.imread(img_path)
        h, w = img.shape[:2]; target = np.zeros((10, 160, 160), dtype=np.float32)
        
        img = torch.from_numpy(cv2.resize(img, (640, 640))).float().permute(2,0,1)/255.
        if self.augment: img = self.aug_pipe(img)

        craters = self.df[self.df['inputImage'] == name]
        for _, r in craters.iterrows():
            if r['ellipseCenterX(px)'] == -1: continue
            gx, gy = r['ellipseCenterX(px)']*160/w, r['ellipseCenterY(px)']*160/h
            ix, iy = int(gx), int(gy)
            if 0 <= ix < 160 and 0 <= iy < 160:
                target[int(r['crater_classification']) if r['crater_classification']!=-1 else 0, iy, ix] = 1.0
                target[5:7, iy, ix] = np.log(np.array([r['ellipseSemimajor(px)'], r['ellipseSemiminor(px)']]) + 1)
                target[7:9, iy, ix] = [gx-ix, gy-iy]; target[9, iy, ix] = np.deg2rad(r['ellipseRotation(deg)'])
        return img, torch.from_numpy(target)

@torch.inference_mode()
def run_inference(model_path, test_dir):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCraterV6(weights=None).to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev)); model.eval()
    
    paths = list(Path(test_dir).rglob("*.png")); results = []
    for p in tqdm(paths):
        img_id = f"{p.parts[-3]}/{p.parts[-2]}/{p.stem}"
        raw = cv2.imread(str(p))
        if raw is None: continue
        h_o, w_o = raw.shape[:2]
        img = torch.from_numpy(cv2.resize(raw, (640, 640))).float().permute(2,0,1).to(dev).unsqueeze(0)/255.
        out = model(img); hm = torch.sigmoid(out['hm'][0])
        hmax = torch.nn.functional.max_pool2d(hm.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        hm = hm * (hmax == hm).float(); found = False
        
        scores, idxs = torch.topk(hm.view(-1), k=40)
        for s, idx in zip(scores, idxs):
            if s < 0.35: continue
            c, iy, ix = np.unravel_index(idx.cpu().numpy(), (5, 160, 160))
            ma = (torch.exp(out['axes'][0,0,iy,ix])-1).item() * (w_o/160)
            mi = (torch.exp(out['axes'][0,1,iy,ix])-1).item() * (h_o/160)
            cx = (ix + out['off'][0,0,iy,ix].item()) * (w_o/160)
            cy = (iy + out['off'][0,1,iy,ix].item()) * (h_o/160)
            rot = np.rad2deg(out['rot'][0,0,iy,ix].item())
            results.append([cx, cy, ma, mi, rot, img_id, c])
            found = True
        if not found: results.append([-1, -1, -1, -1, -1, img_id, -1])

    cols = ['ellipseCenterX(px)', 'ellipseCenterY(px)', 'ellipseSemimajor(px)', 
            'ellipseSemiminor(px)', 'ellipseRotation(deg)', 'inputImage', 'crater_classification']
    pd.DataFrame(results, columns=cols).to_csv("submission.csv", index=False)

if __name__ == "__main__": run_inference("swin_crater_best.pth", "./test")
