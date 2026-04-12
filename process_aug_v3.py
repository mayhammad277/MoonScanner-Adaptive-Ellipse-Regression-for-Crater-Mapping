import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# Double check these paths match your actual folders
DATA_ROOT = Path("/home/bora3i/crater_challenge/train-sample/")
SAVE_DIR = Path("/home/bora3i/crater_challenge/processed_data_aug_v3")
INPUT_RES = 2592
TRAIN_RES = 640
OUT_RES = 160 
SCALE = OUT_RES / INPUT_RES

# FORCE directory creation
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def generate_clean_gaussian(shape, ctx, cty, sigma):
    """Calculates Gaussian using a 2D meshgrid to prevent horizontal lines."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-((x - ctx)**2 + (y - cty)**2) / (2 * sigma**2))

def process_direct_safe():
    tasks = []
    # Search for Image/Mask pairs
    for root, dirs, files in os.walk(DATA_ROOT):
        if "truth" in dirs:
            truth_dir = Path(root) / "truth"
            img_files = [f for f in files if f.endswith('.png') and not f.endswith('_mask.png')]
            for f in img_files:
                img_path = Path(root) / f
                mask_path = truth_dir / f.replace(".png", "_mask.png")
                if mask_path.exists():
                    tasks.append((img_path, mask_path))

    if not tasks:
        print(f"❌ No pairs found! Check if 'truth' folders exist in {DATA_ROOT}")
        return

    print(f"✅ Found {len(tasks)} pairs. Saving to {SAVE_DIR}")

    for img_path, mask_path in tqdm(tasks, desc="Processing"):
        raw_img = cv2.imread(str(img_path))
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_img is None or raw_mask is None: continue

        # Fix memory strides immediately
        raw_img = np.ascontiguousarray(raw_img)
        raw_mask = np.ascontiguousarray(raw_mask)

        for suffix, flip_code in [("orig", None), ("hflip", 1)]:
            # Hard copy after flip fixes the 'ruined' heatmap look
            if flip_code is not None:
                img_aug = cv2.flip(raw_img, flip_code).copy()
                mask_aug = cv2.flip(raw_mask, flip_code).copy()
            else:
                img_aug, mask_aug = raw_img.copy(), raw_mask.copy()
            
            img_res = cv2.resize(img_aug, (TRAIN_RES, TRAIN_RES))
            _, binary = cv2.threshold(mask_aug, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            gt = np.zeros((10, OUT_RES, OUT_RES), dtype=np.float32)

            for cnt in contours:
                if cv2.contourArea(cnt) < 15: continue
                
                # Use fitEllipse for better axis/angle alignment
                if len(cnt) >= 5:
                    (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
                else:
                    (cx, cy), r = cv2.minEnclosingCircle(cnt)
                    ma, mi, angle = r*2, r*2, 0

                scx, scy = cx * SCALE, cy * SCALE
                ix, iy = int(scx), int(scy)

                if 0 <= ix < OUT_RES and 0 <= iy < OUT_RES:
                    sigma = max((ma * SCALE) / 6.0, 1.5)
                    gaussian = generate_clean_gaussian((OUT_RES, OUT_RES), scx, scy, sigma)
                    gt[0] = np.maximum(gt[0], gaussian) 
                    
                    gt[5, iy, ix] = ma * SCALE
                    gt[6, iy, ix] = mi * SCALE
                    gt[7, iy, ix] = scx - ix
                    gt[8, iy, ix] = scy - iy
                    gt[9, iy, ix] = np.deg2rad(angle)

            # Build a flat filename to avoid nested folder issues
            rel_parts = img_path.parts[len(DATA_ROOT.parts):]
            safe_prefix = "_".join(rel_parts).replace(".png", "")
            
            np.save(SAVE_DIR / f"{safe_prefix}_{suffix}_img.npy", img_res)
            np.save(SAVE_DIR / f"{safe_prefix}_{suffix}_gt.npy", gt)

    print(f"🎉 Done! Files saved in: {SAVE_DIR}")

if __name__ == "__main__":
    process_direct_safe()
