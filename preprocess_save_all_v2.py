import os
import cv2
import numpy as np
import math
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = "/home/bora3i/crater_challenge/train-sample/" 
SAVE_DIR = "./processed_data"
INPUT_RES = 2592
TRAIN_RES = 640
STRIDE = 4
OUT_RES = TRAIN_RES // STRIDE # 160 grid
MIN_AREA_FILTER = 17
KERNEL_SIZE = 15

os.makedirs(SAVE_DIR, exist_ok=True)

def generate_gaussian_kernel(kernel_size, sigma):
    kernel_radius = (kernel_size - 1) // 2
    x_grid = np.arange(0, kernel_size, 1, np.float32)
    y_grid = x_grid[:, np.newaxis]
    x0 = y0 = kernel_radius
    gaussian_kernel = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
    return gaussian_kernel

def process_nested_data():
    tasks = []
    
    # 1. Loop through Altitude folders
    altitudes = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"Found {len(altitudes)} altitude folders.")

    for alt in altitudes:
        alt_path = os.path.join(DATA_ROOT, alt)
        
        # 2. Loop through Longitude folders
        longitudes = [d for d in os.listdir(alt_path) if os.path.isdir(os.path.join(alt_path, d))]
        
        for lon in longitudes:
            lon_path = os.path.join(alt_path, lon)
            truth_path = os.path.join(lon_path, "truth")
            
            if not os.path.exists(truth_path):
                continue
            
            # 3. Find Images (excluding truth folder)
            for f in os.listdir(lon_path):
                if f.endswith('.png'):
                    img_path = os.path.join(lon_path, f)
                    
                    # 4. Find the corresponding _mask.png in the truth folder
                    img_name_no_ext = os.path.splitext(f)[0]
                    mask_name = f"{img_name_no_ext}_mask.png"
                    mask_path = os.path.join(truth_path, mask_name)
                    
                    if os.path.exists(mask_path):
                        tasks.append((img_path, mask_path))

    if not tasks:
        print("❌ ERROR: No matching image/mask pairs found! Check naming convention.")
        return

    print(f"✅ Found {len(tasks)} valid pairs. Processing...")

    for img_path, mask_path in tqdm(tasks):
        # Unique ID based on folder hierarchy
        rel_path = os.path.relpath(img_path, DATA_ROOT)
        unique_id = rel_path.replace(os.sep, "_").replace(".png", "")
        
        # Load and Clean
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (TRAIN_RES, TRAIN_RES))
        
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply your Morphological Open logic
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
        mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Feature Extraction
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # [0: Heatmap, 1: Radius, 2: OffsetX, 3: OffsetY]
        gt_map = np.zeros((4, OUT_RES, OUT_RES), dtype=np.float32)
        scale = OUT_RES / INPUT_RES
        
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA_FILTER:
                (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
                
                # Scale to 160x160
                scaled_cx, scaled_cy = cx_f * scale, cy_f * scale
                scaled_r = r * scale
                ix, iy = int(scaled_cx), int(scaled_cy)
                
                if 0 <= ix < OUT_RES and 0 <= iy < OUT_RES:
                    # Heatmap logic
                    sigma = max(scaled_r / 3.0, 1.0)
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
                    
                    # Regression logic
                    gt_map[1, iy, ix] = scaled_r
                    gt_map[2, iy, ix] = scaled_cx - ix
                    gt_map[3, iy, ix] = scaled_cy - iy

        # Save as npy
        np.save(os.path.join(SAVE_DIR, f"{unique_id}_img.npy"), img_resized)
        np.save(os.path.join(SAVE_DIR, f"{unique_id}_gt.npy"), gt_map)

if __name__ == "__main__":
    process_nested_data()
