import os
import cv2
import numpy as np
import math
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = "/home/bora3i/crater_challenge/train-sample/" 
SAVE_DIR = "./processed_data_aug"
INPUT_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUT_RES = TRAIN_RES // STRIDE 
MIN_AREA_FILTER = 17
KERNEL_SIZE = 15

os.makedirs(SAVE_DIR, exist_ok=True)

def generate_gaussian_kernel(kernel_size, sigma):
    kernel_radius = (kernel_size - 1) // 2
    x_grid = np.arange(0, kernel_size, 1, np.float32)
    y_grid = x_grid[:, np.newaxis]
    x0 = y0 = kernel_radius
    # Gaussian formula with peak 1.0
    gaussian_kernel = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
    return gaussian_kernel

def create_gt_map(img_size, contours, scale):
    """Generates the 4-channel ground truth for a given set of image/mask data."""
    gt_map = np.zeros((4, OUT_RES, OUT_RES), dtype=np.float32)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA_FILTER:
            (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
            scaled_cx, scaled_cy = cx_f * scale, cy_f * scale
            scaled_r = r * scale
            ix, iy = int(scaled_cx), int(scaled_cy)
            
            if 0 <= ix < OUT_RES and 0 <= iy < OUT_RES:
                # IMPROVEMENT: Softer Gaussian (sigma = r/2 instead of r/3)
                sigma = max(scaled_r / 2.0, 1.5) 
                k_radius = math.ceil(sigma * 3)
                k_size = 2 * k_radius + 1
                if k_size > 51: k_size = 51; k_radius = 25
                
                g_kernel = generate_gaussian_kernel(k_size, sigma)
                
                x_s_i, x_e_i = max(0, ix - k_radius), min(OUT_RES, ix + k_radius + 1)
                y_s_i, y_e_i = max(0, iy - k_radius), min(OUT_RES, iy + k_radius + 1)
                x_s_k = k_radius - (ix - x_s_i)
                x_e_k = k_radius + (x_e_i - ix)
                y_s_k = k_radius - (iy - y_s_i)
                y_e_k = k_radius + (y_e_i - iy)
                
                gt_map[0, y_s_i:y_e_i, x_s_i:x_e_i] = np.maximum(
                    gt_map[0, y_s_i:y_e_i, x_s_i:x_e_i],
                    g_kernel[y_s_k:y_e_k, x_s_k:x_e_k]
                )
                gt_map[1, iy, ix] = scaled_r
                gt_map[2, iy, ix] = scaled_cx - ix
                gt_map[3, iy, ix] = scaled_cy - iy
    return gt_map

def process_and_augment():
    tasks = []
    # Find image/mask pairs (Same logic as before)
    for alt in os.listdir(DATA_ROOT):
        alt_path = os.path.join(DATA_ROOT, alt)
        if not os.path.isdir(alt_path): continue
        for lon in os.listdir(alt_path):
            lon_path = os.path.join(alt_path, lon)
            truth_path = os.path.join(lon_path, "truth")
            if not os.path.exists(truth_path): continue
            for f in os.listdir(lon_path):
                if f.endswith('.png'):
                    img_path = os.path.join(lon_path, f)
                    mask_path = os.path.join(truth_path, f.replace(".png", "_mask.png"))
                    if os.path.exists(mask_path):
                        tasks.append((img_path, mask_path))

    scale = OUT_RES / INPUT_RES

    for img_path, mask_path in tqdm(tasks):
        rel_path = os.path.relpath(img_path, DATA_ROOT)
        base_id = rel_path.replace(os.sep, "_").replace(".png", "")
        
        # Load raw data
        raw_img = cv2.imread(img_path)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Prepare 4 augmentation states
        # (Image, FlipCode) where FlipCode: None=Orig, 1=Horiz, 0=Vert, -1=Both
        augmentations = [
            ("orig", raw_img, raw_mask),
            ("hflip", cv2.flip(raw_img, 1), cv2.flip(raw_mask, 1)),
            ("vflip", cv2.flip(raw_img, 0), cv2.flip(raw_mask, 0)),
            ("rotate", cv2.flip(raw_img, -1), cv2.flip(raw_mask, -1))
        ]

        for suffix, img, mask in augmentations:
            # Resize image
            img_res = cv2.resize(img, (TRAIN_RES, TRAIN_RES))
            
            # Clean mask
            _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
            mask_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_map = create_gt_map(OUT_RES, contours, scale)
            
            # Save
            save_id = f"{base_id}_{suffix}"
            np.save(os.path.join(SAVE_DIR, f"{save_id}_img.npy"), img_res)
            np.save(os.path.join(SAVE_DIR, f"{save_id}_gt.npy"), gt_map)

if __name__ == "__main__":
    process_and_augment()
