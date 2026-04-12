import os
import numpy as np
import cv2
from tqdm import tqdm

def robust_save_debug(img_root, heatmap_dir, output_dir="debug_check"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get ALL images (even in subfolders)
    print("Indexing images...")
    img_map = {f: os.path.join(r, f) for r, _, files in os.walk(img_root) 
               for f in files if f.lower().endswith(('.png', '.jpg'))}

    hm_files = [f for f in os.listdir(heatmap_dir) if f.endswith('.npy')]
    
    for hm_name in tqdm(hm_files[:80]): # Test first 20
        hm_path = os.path.join(heatmap_dir, hm_name)
        data = np.load(hm_path)

        # --- DIAGNOSTIC PRINT ---
        print(f"\nFile: {hm_name}")
        print(f"Shape: {data.shape} | Max: {data.max():.4f} | Min: {data.min():.4f}")

        # 2. Extract HM logic
        if data.ndim == 3: # If [C, H, W]
            hm = np.max(data[:5], axis=0) # Take max of first 5 Mahanti channels
        else:
            hm = data

        # 3. Find matching image
        base = hm_name.replace('.npy', '')
        # List of potential matches based on your file system
        img_path = None
        for suffix in ['', '_orig_img', '_img', '_gt']:
            candidate = f"{base}{suffix}.png"
            if candidate in img_map:
                img_path = img_map[candidate]
                break
        
        # 4. Create Visualization even if image is missing
        if img_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 640))
        else:
            print(f"⚠️ Image not found for {hm_name}")
            img = np.zeros((640, 640, 3), dtype=np.uint8)

        # 5. Amplify Heatmap (Normalize 0 to 1 so we can see faint peaks)
        if hm.max() > 0:
            hm_norm = (hm / hm.max()) 
        else:
            hm_norm = hm
            
        hm_res = cv2.resize(hm_norm, (640, 640))
        hm_color = cv2.applyColorMap((hm_res * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img, 0.7, hm_color, 0.3, 0)
        canvas = np.hstack((img, overlay))
        
        cv2.imwrite(os.path.join(output_dir, f"check_{base}.png"), canvas)

if __name__ == "__main__":
    robust_save_debug("/home/bora3i/train-sample", "/home/bora3i/crater_challenge/processed_data_aug_v3")
