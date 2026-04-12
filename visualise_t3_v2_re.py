import pandas as pd
import cv2
import os
import random
import numpy as np

def visualize_craters(csv_path, test_root, output_dir="viz_results_res_t3", num_samples=20):
    # 1. Load the results
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found. Run inference first!")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ The CSV is empty. No craters to visualize!")
        return

    # 2. Create output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Pick random images that have detections
    unique_images = df['image_id'].unique()
    samples = random.sample(list(unique_images), min(len(unique_images), num_samples))

    print(f"🎨 Drawing circles on {len(samples)} sample images...")

    for img_id in samples:
        img_path = os.path.join(test_root, img_id)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❓ Could not find image: {img_path}")
            continue

        # Filter detections for this specific image
        detections = df[df['image_id'] == img_id]

        for _, row in detections.iterrows():
            cx, cy, r = int(row['cx']), int(row['cy']), int(row['radius'])
            conf = row['confidence']

            # Color coding based on confidence
            # Green = High (>= 0.7), Yellow = Med (0.5-0.7), Red = Low (< 0.5)
            if conf >= 0.7:
                color = (0, 255, 0) # Green
            elif conf >= 0.5:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255) # Red

            # Draw the circle rim
            cv2.circle(img, (cx, cy), r, color, 2)
            # Draw the center point
            cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
            # Add confidence text
            cv2.putText(img, f"{conf:.2f}", (cx, cy - r - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save the result
        save_name = img_id.replace('/', '_') # Flatten subfolder names for filename
        save_path = os.path.join(output_dir, f"viz_{save_name}")
        cv2.imwrite(save_path, img)

    print(f"✅ Done! Check the '{output_dir}' folder for your images.")

if __name__ == "__main__":
    CSV_FILE = "/home/bora3i/crater_challenge/submission_pro_final.csv"
    TEST_DIR = "/home/bora3i/crater_challenge/test"
    
    visualize_craters(CSV_FILE, TEST_DIR)
