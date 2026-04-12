import os
import cv2
import numpy as np
import math
from tqdm import tqdm

# --- Configuration ---
INPUT_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUT_RES = TRAIN_RES // STRIDE  # 160
# Set this to the folder containing all altitudeXX folders
DATA_ROOT = "/home/s478608/may/crater_challenge/train/" 
SAVE_DIR = "./processed_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def draw_gaussian(heatmap, center, sigma):
    tmp_size = int(sigma * 3)
    mu_x, mu_y = int(center[0]), int(center[1])
    w, h = heatmap.shape[1], heatmap.shape[0]
    
    ul = [mu_x - tmp_size, mu_y - tmp_size]
    br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
    
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
        
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def process_and_save():
    image_tasks = []

    print("Scanning directory structure...")
    for root, dirs, files in os.walk(DATA_ROOT):
        # Skip the 'truth' folders themselves so we don't treat masks as input images
        if 'truth' in root:
            continue
            
        for f in files:
            if f.endswith('.png'):
                img_path = os.path.join(root, f)
                
                # Construct Mask Path: 
                # If image is: .../longitude01/image_1.png
                # Mask is:     .../longitude01/truth/image_1_truth.png
                img_dir = root
                img_name_no_ext = os.path.splitext(f)[0]
                mask_path = os.path.join(img_dir, "truth", f"{img_name_no_ext}_mask.png")
                
                if os.path.exists(mask_path):
                    image_tasks.append((img_path, mask_path))

    print(f"Found {len(image_tasks)} valid image-mask pairs.")

    for img_path, mask_path in tqdm(image_tasks):
        # Create a unique filename by replacing slashes with underscores
        # Example: altitude01/longitude01/tile_0 -> altitude01_longitude01_tile_0
        rel_path = os.path.relpath(img_path, DATA_ROOT)
        unique_id = rel_path.replace(os.sep, "_").replace(".png", "")
        
        # 1. Load Image and Mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            continue

        # 2. Resize Image for the Model
        img_resized = cv2.resize(img, (TRAIN_RES, TRAIN_RES))
        
        # 3. Generate Ground Truth Maps at Output Resolution (160x160)
        gt_hm = np.zeros((OUT_RES, OUT_RES), dtype=np.float32)
        gt_reg = np.zeros((3, OUT_RES, OUT_RES), dtype=np.float32) # [Radius, OffX, OffY]
        
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        scale = OUT_RES / INPUT_RES
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: 
                continue
                
            (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
            
            # Scale coordinates to 160x160 grid
            ctx, cty = cx_f * scale, cy_f * scale
            r_scaled = r * scale
            ix, iy = int(ctx), int(cty)
            
            if 0 <= ix < OUT_RES and 0 <= iy < OUT_RES:
                sigma = max(r_scaled / 3, 1.0)
                draw_gaussian(gt_hm, (ctx, cty), sigma)
                gt_reg[0, iy, ix] = r_scaled
                gt_reg[1, iy, ix] = ctx - ix
                gt_reg[2, iy, ix] = cty - iy

        # 4. Save processed data
        np.save(os.path.join(SAVE_DIR, f"{unique_id}_img.npy"), img_resized)
        np.save(os.path.join(SAVE_DIR, f"{unique_id}_gt.npy"), 
                np.concatenate([gt_hm[None, ...], gt_reg], axis=0))

if __name__ == "__main__":
    process_and_save()
