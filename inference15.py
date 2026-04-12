import torch, cv2, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from train import SwinCrater640
import os, cv2, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.checkpoint import checkpoint

class SwinCrater640(nn.Module):
    def __init__(self):
        super().__init__()
        base = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = base.features
        # 3-stage upsampling to reach 640x640 (20->80->320->640)
        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, 4, 4), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 4), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.hm = nn.Conv2d(64, 5, 1)
        self.axes = nn.Conv2d(64, 2, 1)
        self.off = nn.Conv2d(64, 2, 1)
        self.rot = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Using checkpointing to save VRAM
        x = self.backbone(x).permute(0, 3, 1, 2).contiguous()
        f = self.neck(x)
        return {"hm": self.hm(f), "axes": self.axes(f), "off": self.off(f), "rot": self.rot(f)}

class CraterDataset640(Dataset):
    def __init__(self, df, img_root, img_list):
        self.df = df
        self.img_root = Path(img_root)
        self.images = img_list
        self.scale = 640 / 2592
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        p = self.img_root / (img_id if img_id.endswith(".png") else img_id + ".png")
        img_raw = cv2.imread(str(p))
        if img_raw is None: return self.__getitem__(0)
        img = cv2.resize(img_raw, (640, 640))
        img = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - self.mean) / self.std
        
        gt = torch.zeros((10, 640, 640))
        annos = self.df[self.df['inputImage'] == img_id]
        for _, r in annos.iterrows():
            ma_s = r['ellipseSemimajor(px)'] * self.scale
            if ma_s < 2.5: continue # Slightly higher filter for 640 scale
            ctx, cty = r['ellipseCenterX(px)'] * self.scale, r['ellipseCenterY(px)'] * self.scale
            ix, iy = int(np.clip(ctx, 0, 639)), int(np.clip(cty, 0, 639))
            sigma = max(1.2, ma_s / 6.0)
            y, x = np.ogrid[:640, :640]
            g = np.exp(-((x - ctx)**2 + (y - cty)**2) / (2 * sigma**2))
            gt[int(r['crater_classification'])] = torch.max(gt[int(r['crater_classification'])], torch.from_numpy(g).float())
            gt[5, iy, ix] = np.log(ma_s + 1.0)
            gt[6, iy, ix] = np.log(r['ellipseSemiminor(px)'] * self.scale + 1.0)
            gt[7, iy, ix], gt[8, iy, ix] = ctx - ix, cty - iy
            gt[9, iy, ix] = np.deg2rad(r['ellipseRotation(deg)'])
        return img, gt

class SizePriorityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='sum', delta=1.0)
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, preds, target):
        gt_hm = target[:, 0:5]
        reg_mask = (gt_hm.max(dim=1, keepdim=True)[0] > 0.95)
        num = reg_mask.sum() + 1e-4
        a_loss = self.huber(preds['axes'][reg_mask.repeat(1,2,1,1)], target[:, 5:7][reg_mask.repeat(1,2,1,1)]) / num
        o_loss = self.l1(preds['off'][reg_mask.repeat(1,2,1,1)], target[:, 7:9][reg_mask.repeat(1,2,1,1)]) / num
        hm_loss = nn.functional.binary_cross_entropy_with_logits(preds['hm'], gt_hm)
        return (20.0 * hm_loss) + (50.0 * a_loss) + (15.0 * o_loss) # Extra size priority
@torch.inference_mode()
def run_inference(model_path, test_dir, output_csv):
    device = torch.device("cuda")
    SCALE = 640 / 2592 # 1:1 input to output mapping
    model = SwinCrater640().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    results = []
    
    for p in tqdm(list(Path(test_dir).rglob("*.png"))):
        img_id = f"{p.parts[-3]}/{p.parts[-2]}/{p.stem}"
        raw = cv2.imread(str(p))
        if raw is None: continue
        img_t = (torch.from_numpy(cv2.resize(raw, (640, 640))).permute(2, 0, 1).float().to(device) / 255.0 - mean) / std
        out = model(img_t.unsqueeze(0))
        hm = torch.sigmoid(out['hm'][0]).max(dim=0)[0]
        # Sharper NMS for 640 scale
        hmax = torch.nn.functional.max_pool2d(hm.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
        hm = hm * (hmax == hm).float()
        
        scores, idxs = torch.topk(hm.view(-1), k=70) # Collect more at 640 scale
        for s, idx in zip(scores, idxs):
            if s < 0.55: continue # Even stricter threshold for 640 accuracy
            iy, ix = np.unravel_index(idx.cpu().numpy(), (640, 640))
            
            ma = (torch.exp(out['axes'][0, 0, iy, ix]).item() - 1.0) / SCALE
            mi = (torch.exp(out['axes'][0, 1, iy, ix]).item() - 1.0) / SCALE
            cx = (ix + out['off'][0, 0, iy, ix].item()) / SCALE
            cy = (iy + out['off'][0, 1, iy, ix].item()) / SCALE
            rot = np.rad2deg(out['rot'][0, 0, iy, ix].item())
            results.append([cx, cy, ma, mi, rot, img_id, 0])

    pd.DataFrame(results, columns=['ellipseCenterX(px)', 'ellipseCenterY(px)', 'ellipseSemimajor(px)', 'ellipseSemiminor(px)', 'ellipseRotation(deg)', 'inputImage', 'crater_classification']).to_csv(output_csv, index=False)

if __name__ == "__main__":
    run_inference("best_size_model_640.pth", "test_data_dir", "submission.csv")
