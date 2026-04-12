import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

def draw_mahanti_previews(csv_path, img_root, output_dir, num_samples=50):
    # 1. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    # Get a list of unique images that actually have detections
    image_names = df['inputImage'].unique()
    
    # Limit samples so we don't fill up disk space
    sample_images = image_names[:num_samples]
    
    # Mahanti Class Mapping & Colors
    classes = {0: "Fresh", 1: "Degraded", 2: "Intermediate", 3: "Old", 4: "Very Old"}
    colors = {
        0: (0, 255, 0),    # Green
        1: (255, 255, 0),  # Cyan
        2: (255, 165, 0),  # Orange
        3: (0, 0, 255),    # Red
        4: (128, 0, 128)   # Purple
    }

    print(f"🖼️ Generating {len(sample_images)} previews...")

    for img_rel_path in tqdm(sample_images):
        full_img_path = os.path.join(img_root, img_rel_path)
        img = cv2.imread(full_img_path)
        
        if img is None:
            print(f"⚠️ Could not load {full_img_path}")
            continue

        # Get all detections for this specific image
        detections = df[df['inputImage'] == img_rel_path]

        for _, row in detections.iterrows():
            c_idx = int(row['crater_classification'])
            center = (int(row['ellipseCenterX(px)']), int(row['ellipseCenterY(px)']))
            axes = (int(row['ellipseSemimajor(px)']), int(row['ellipseSemiminor(px)']))
            angle = row['ellipseRotation(deg)']
            
            # Draw the ellipse
            cv2.ellipse(img, center, axes, angle, 0, 360, colors[c_idx], 3)
            
            # Label with class name
            label = classes.get(c_idx, "Unknown")
            cv2.putText(img, label, (center[0], center[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[c_idx], 2)

        # Save to output folder (flattening folder structure for easy viewing)
        save_name = img_rel_path.replace("/", "_").replace("\\", "_")
        cv2.imwrite(os.path.join(output_dir, f"pred_{save_name}"), img)

    print(f"✅ Previews saved to: {output_dir}")

if __name__ == "__main__":
    # Update these to your actual paths
    CSV_FILE = "submission_live.csv"
    IMG_ROOT = "/home/bora3i/crater_challenge/test"
    OUT_DIR = "inference_previews"
    
    draw_mahanti_previews(CSV_FILE, IMG_ROOT, OUT_DIR)
