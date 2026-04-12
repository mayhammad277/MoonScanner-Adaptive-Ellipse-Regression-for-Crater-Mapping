import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.models import swin_t, Swin_T_Weights
import torch.nn as nn


# =========================
# MODEL
# =========================
class SwinCraterMahantiV5(nn.Module):
    def __init__(self):
        super().__init__()
        base = swin_t(weights=None)
        self.backbone = base.features

        self.neck = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=8, stride=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.hm   = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 5, 1))
        self.axes = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.off  = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))
        self.rot  = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.backbone(x).permute(0,3,1,2).contiguous()
        f = self.neck(x)
        return {
            "hm": self.hm(f),
            "axes": self.axes(f),
            "off": self.off(f),
            "rot": self.rot(f)
        }


# =========================
# UTILS
# =========================
def topk_peaks(hm, K=120):
    hm = hm.reshape(-1)
    v, i = torch.topk(hm, K)
    return v, i


# =========================
# INFERENCE
# =========================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinCraterMahantiV5().to(device)
    model.load_state_dict(torch.load("swin_crater_best.pth", map_location=device))
    model.eval()

    INP = 640
    HM  = 160
    ORIG = 2592

    s_img = INP / ORIG
    s_hm  = HM / INP

    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    img_dir = Path("./test")
    out_f = open("output.csv", "w")

    for img_path in sorted(img_dir.glob("*.png")):

        name = img_path.stem  # NO .png
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_r = cv2.resize(img, (INP, INP))
        img_t = torch.from_numpy(img_r).permute(2,0,1).float()/255.0
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            p = model(img_t)

        hm = torch.sigmoid(p['hm'][0])
        axes = p['axes'][0]
        off  = p['off'][0]

        detections = []

        for c in range(5):
            v, idx = topk_peaks(hm[c], K=60)
            for score, id in zip(v, idx):
                if score < 0.15:   # LOW threshold for recall
                    continue

                y = id // HM
                x = id % HM

                dx = off[0,y,x]
                dy = off[1,y,x]

                cx = (x + dx) / s_hm / s_img
                cy = (y + dy) / s_hm / s_img

                ma = axes[0,y,x] / s_hm / s_img
                mi = axes[1,y,x] / s_hm / s_img

                detections.append((c, cx, cy, ma, mi))

        # ---- ZERO CRATER CASE ----
        if len(detections) == 0:
            out_f.write(f"-1,-1,-1,-1,-1,{name},-1\n")
            continue

        # ---- WRITE ----
        for d in detections:
            cls, cx, cy, ma, mi = d
            out_f.write(f"{cx:.2f},{cy:.2f},{ma:.2f},{mi:.2f},0,{name},{cls}\n")

    out_f.close()

