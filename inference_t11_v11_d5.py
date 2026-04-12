import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.models import swin_t, Swin_T_Weights
import torch.nn as nn

# =========================
# MODEL (must match training)
# =========================
class SwinCraterNet(nn.Module):
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
# INFERENCE
# =========================
def run_inference(data_dir, output_csv):
    DEVICE = torch.device("cpu")

    model = SwinCraterNet().to(DEVICE)
    model.load_state_dict(torch.load("swin_crater_best.pth", map_location="cpu"))
    model.eval()

    INP = 640
    HM = 160
    ORIG = 2592
    scale = ORIG / HM

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    results = []

    img_paths = sorted(Path(data_dir).glob("*.png"))

    for img_path in img_paths:
        img_id = img_path.stem  # 🔥 ID ONLY

        img = cv2.imread(str(img_path))
        img_res = cv2.resize(img, (INP, INP))
        img_t = torch.from_numpy(img_res).permute(2,0,1).float() / 255.0
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0)

        with torch.no_grad():
            preds = model(img_t)

        hm = torch.sigmoid(preds["hm"])[0]
        axes = preds["axes"][0]
        off  = preds["off"][0]
        rot  = preds["rot"][0]

        found = False

        for cls in range(5):
            heat = hm[cls]
            ys, xs = torch.where(heat > 0.3)

            for y, x in zip(ys, xs):
                found = True

                cx = (x + off[0,y,x]) * scale
                cy = (y + off[1,y,x]) * scale

                ma = axes[0,y,x] * scale
                mi = axes[1,y,x] * scale

                angle = rot[0,y,x].item() * 90.0  # 🔥 de-normalize

                results.append([
                    float(cx), float(cy), float(ma), float(mi), float(angle),
                    img_id, cls
                ])

        # 🔥 SPECIAL CASE: NO CRATERS
        if not found:
            results.append([
                -1, -1, -1, -1, -1, img_id, -1
            ])

    df = pd.DataFrame(results, columns=[
        "ellipseCenterX(px)",
        "ellipseCenterY(px)",
        "ellipseSemimajor(px)",
        "ellipseSemiminor(px)",
        "ellipseRotation(deg)",
        "inputImage",
        "crater_classification"
    ])

    df.to_csv(output_csv, index=False)
    print("✅ Saved:", output_csv)


if __name__ == "__main__":
    import sys
    run_inference(sys.argv[1], sys.argv[2])

