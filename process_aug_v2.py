import os
import cv2
import numpy as np
import math
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = "/home/bora3i/crater_challenge/train-sample/" 
SAVE_DIR = "./processed_data_aug_v2"
INPUT_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUT_RES = TRAIN_RES // STRIDE 
MIN_AREA_FILTER = 15 # Slightly lowered to catch small craters

os.makedirs(SAVE_DIR, exist_ok=True)

def apply_clahe(img):
    """Enhances crater rims by increasing local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Apply to each channel
    img[:,:,0] = clahe.apply(img[:,:,0])
    img[:,:,1] = clahe.apply(img[:,:,1])
    img[:,:,2] = clahe.apply(img[:,:,2])
    return img

def generate_gaussian_kernel(kernel_size, sigma):
    kernel_radius = (kernel_size - 1) // 2
    x_grid = np.arange(0, kernel_size, 1, np.float32)
    y_grid = x_grid[:, np.newaxis]
    x0 = y0 = kernel_radius
    gaussian_kernel = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
    return gaussian_kernel

def create_gt_map(contours, scale):
    gt_map = np.zeros((4, OUT_RES, OUT_RES), dtype=np.float32)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA_FILTER:
            (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
            scaled_cx, scaled_cy = cx_f * scale, cy_f * scale
            scaled_r = r * scale
            ix, iy = int(scaled_cx), int(scaled_cy)
            
            if 0 <= ix < OUT_RES and 0 <= iy < OUT_RES:
                # IMPROVEMENT: Use an adaptive sigma based on crater size
                # Smaller craters get sharper peaks (r/3), larger get slightly softer (r/4)
                sigma = max(scaled_r / 3.0, 1.0) 
                
                k_radius = int(3 * sigma)
                k_size = 2 * k_radius + 1
                g_kernel = generate_gaussian_kernel(k_size, sigma)
                
                # Boundary management for Gaussian placement
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
                
                # Precision heads
                gt_map[1, iy, ix] = scaled_r
                gt_map[2, iy, ix] = scaled_cx - ix
                gt_map[3, iy, ix] = scaled_cy - iy
                
    return gt_map

def process_and_augment():
    # Collect tasks
    tasks = []
    for alt in os.listdir(DATA_ROOT):
        alt_path = os.path.join(DATA_ROOT, alt)
        if not os.path.isdir(alt_path): continue
        for lon in os.listdir(alt_path):
            lon_path = os.path.join(alt_path, lon)
            truth_path = os.path.join(lon_path, "truth")
            if not os.path.exists(truth_path): continue
            for f in os.listdir(lon_path):
                if f.endswith('.png'):
                    tasks.append((os.path.join(lon_path, f), 
                                  os.path.join(truth_path, f.replace(".png", "_mask.png"))))

    scale = OUT_RES / INPUT_RES

    for img_path, mask_path in tqdm(tasks):
        rel_path = os.path.relpath(img_path, DATA_ROOT)
        base_id = rel_path.replace(os.sep, "_").replace(".png", "")
        
        raw_img = cv2.imread(img_path)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply CLAHE to help the model see rims
        raw_img = apply_clahe(raw_img)

        augmentations = [
            ("orig", raw_img, raw_mask),
            ("hflip", cv2.flip(raw_img, 1), cv2.flip(raw_mask, 1)),
            ("vflip", cv2.flip(raw_img, 0), cv2.flip(raw_mask, 0)),
            ("rotate", cv2.rotate(raw_img, cv2.ROTATE_180), cv2.rotate(raw_mask, cv2.ROTATE_180))
        ]

        for suffix, img, mask in augmentations:
            img_res = cv2.resize(img, (TRAIN_RES, TRAIN_RES))
            
            # Use Otsu + Morphological Cleaning for clear contours
            _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # Smaller kernel
            mask_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt_map = create_gt_map(contours, scale)
            
            save_id = f"{base_id}_{suffix}"
            # Standardizing saves as float16/float32 for training speed
            np.save(os.path.join(SAVE_DIR, f"{save_id}_img.npy"), img_res.astype(np.uint8))
            np.save(os.path.join(SAVE_DIR, f"{save_id}_gt.npy"), gt_map.astype(np.float32))

if __name__ == "__main__":
    process_and_augment()
