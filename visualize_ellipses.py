import pandas as pd
import cv2
import os
import random
import numpy as np

def get_lunar_mask(image_bgr):
    """Creates the same mask used in inference to identify the moon's surface."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def visualize_robust_results(csv_path, test_root, output_dir="viz_results_v6", num_samples=15, conf_thresh=0.5):
    # 1. Load and Filter
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df = df[df['confidence'] >= conf_thresh]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Map all filenames to their actual subfolder paths
    print("🔍 Mapping image directory...")
    path_map = {f: os.path.join(root, f) for root, _, files in os.walk(test_root) 
                for f in files if f.lower().endswith(('.png', '.jpg'))}

    available_imgs = [img for img in df['inputImage'].unique() if img in path_map]
    samples = random.sample(available_imgs, min(len(available_imgs), num_samples))

    print(f"🎨 Visualizing {len(samples)} images (Threshold: {conf_thresh})...")

    for img_id in samples:
        img_path = path_map[img_id]
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Draw the "Limb" (Mask Boundary) in faint blue so you can see the 'Safe Zone'
        mask = get_lunar_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 100, 0), 2) # Light blue line

        img_data = df[df['inputImage'] == img_id]

        for _, row in img_data.iterrows():
            cx, cy = int(row['ellipseCenterX(px)']), int(row['ellipseCenterY(px)'])
            ma, mi = int(row['ellipseSemimajor(px)']), int(row['ellipseSemiminor(px)'])
            angle = int(row['ellipseRotation(deg)'])
            conf = row['confidence']
            cls = int(row['crater_classification'])

            # Color Logic: Class 1=Red, Class 2=Yellow, Class 3=Green
            colors = {1: (0, 0, 255), 2: (0, 255, 255), 3: (0, 255, 0)}
            color = colors.get(cls, (255, 255, 255))

            # Draw Ellipse
            cv2.ellipse(img, (cx, cy), (ma, mi), angle, 0, 360, color, 2)
            
            # Label with Confidence
            cv2.putText(img, f"{conf:.2f}", (cx - ma, cy - ma - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save with info in filename
        save_name = f"viz_thresh{conf_thresh}_{img_id.replace('/', '_')}"
        cv2.imwrite(os.path.join(output_dir, save_name), img)

    print(f"✅ Visualizations saved to {output_dir}")

if __name__ == "__main__":
    CSV = "/home/bora3i/crater_challenge/submission_ellipse_final.csv"
    TEST = "/home/bora3i/crater_challenge/test"
    visualize_robust_results(CSV, TEST, conf_thresh=0.9)
