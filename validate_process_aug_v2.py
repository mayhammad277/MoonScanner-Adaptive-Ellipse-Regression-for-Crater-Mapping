import numpy as np
import cv2
import os
import random

def validate_npy_data(data_dir, output_dir="val_processed"):
    # 1. Setup folders
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Get all image files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_img.npy')]
    if not all_files:
        print(f"❌ No .npy files found in {data_dir}")
        return

    # 3. Select samples
    samples = random.sample(all_files, min(10, len(all_files)))
    print(f"🧐 Validating {len(samples)} samples...")

    for img_file in samples:
        gt_file = img_file.replace('_img.npy', '_gt.npy')
        
        # Load NPY files
        # img: (640, 640, 3), gt: (4, 160, 160)
        img = np.load(os.path.join(data_dir, img_file))
        gt = np.load(os.path.join(data_dir, gt_file))

        # Extract Heatmap channel (Channel 0)
        heatmap = gt[0] 
        
        # --- Visualization Processing ---
        # Convert heatmap to 0-255 Jet colormap
        heatmap_vis = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        # Resize heatmap back to 640x640 to match image
        heatmap_resized = cv2.resize(heatmap_color, (640, 640))

        # Combine Original Image and Heatmap side-by-side
        # If image is RGB, convert to BGR for OpenCV saving
        if img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img

        combined = np.hstack((img_bgr, heatmap_resized))

        # Save result
        save_path = os.path.join(output_dir, f"check_{img_file.replace('.npy', '.png')}")
        cv2.imwrite(save_path, combined)

    print(f"✅ Validation images saved to: {output_dir}")

if __name__ == "__main__":
    DATA_FOLDER = "./processed_data_aug_v2"
    validate_npy_data(DATA_FOLDER)
