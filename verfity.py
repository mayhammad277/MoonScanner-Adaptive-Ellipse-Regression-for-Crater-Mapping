import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Configuration
PROCESSED_DIR = "./processed_data_aug"

def verify():
    # 1. Check if files exist
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_img.npy')]
    if not files:
        print(f"❌ Error: No .npy files found in {PROCESSED_DIR}")
        return

    print(f"✅ Found {len(files)} samples.")
    
    # 2. Pick a random sample to inspect
    sample_id = files[0].replace('_img.npy', '')
    print(PROCESSED_DIR, f"{sample_id}_img.npy")
    img = np.load(os.path.join(PROCESSED_DIR, f"{sample_id}_img.npy"))
    gt = np.load(os.path.join(PROCESSED_DIR, f"{sample_id}_gt.npy"))
    
    # gt shape is (4, 160, 160) -> [Heatmap, Radius, OffsetX, OffsetY]
    heatmap = gt[0]
    radius_map = gt[1]

    # 3. Print Statistics
    print(f"--- Statistics for {sample_id} ---")
    print(f"Image - Shape: {img.shape}, Max: {img.max()}, Min: {img.min()}")
    print(f"Heatmap - Max Value: {heatmap.max():.4f} (Should be 1.0)")
    print(f"Radius Map - Max Radius: {radius_map.max():.4f}")
    
    if heatmap.max() == 0:
        print("❌ CRITICAL: The heatmap is empty! The masks were not found or not processed correctly.")
    else:
        print("✅ Heatmap contains data.")

    # 4. Create Visualization
    # Resize heatmap back to image size for overlay
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend image and heatmap
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    cv2.imwrite("verification_check.png", overlay)
    print("✅ Saved 'verification_check.png'. Open it to see if the red spots line up with craters!")

if __name__ == "__main__":
    verify()
