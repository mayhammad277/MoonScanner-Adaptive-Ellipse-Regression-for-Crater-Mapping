import os
import cv2
import pandas as pd
from tqdm import tqdm

def visualize_csv_robust(csv_path, test_root, output_dir="vis_output", num_samples=20):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Map filenames to their full paths across all subfolders
    print("📂 Mapping image directory...")
    path_map = {}
    for root, _, files in os.walk(test_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg')):
                path_map[f] = os.path.join(root, f)

    unique_images = df['inputImage'].unique()
    samples = unique_images[:num_samples]

    for img_name in tqdm(samples):
        # 2. Retrieve the full path from our map
        img_path = path_map.get(img_name)
        
        if img_path is None or not os.path.exists(img_path):
            print(f"⚠️ Skip: {img_name} not found in {test_root}")
            continue

        img = cv2.imread(img_path)
        craters = df[df['inputImage'] == img_name]

        # --- Drawing Logic (Same as before) ---
        for _, row in craters.iterrows():
            center = (int(row['ellipseCenterX(px)']), int(row['ellipseCenterY(px)']))
            axes = (int(row['ellipseSemimajor(px)']), int(row['ellipseSemiminor(px)']))
            angle = row['ellipseRotation(deg)']
            cls = int(row['crater_classification'])
            
            # Use Class-based colors
            color = (0, 255, 0) if cls == 0 else (0, 0, 255) # Green for A, Red for C
            cv2.ellipse(img, center, axes, angle, 0, 360, color, 3)

        # Save using only the filename to avoid path issues in output
        cv2.imwrite(os.path.join(output_dir, f"vis_{img_name}"), img)

if __name__ == "__main__":
    CSV_FILE = "submission_v4_nms.csv"
    # Ensure this is the ROOT folder containing all the altitude/longitude subfolders
    TEST_ROOT ="/home/bora3i/crater_challenge/train-sample"
    
    visualize_csv_robust(CSV_FILE, TEST_ROOT)
