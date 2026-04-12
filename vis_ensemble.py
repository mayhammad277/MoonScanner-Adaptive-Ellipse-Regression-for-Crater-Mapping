import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def draw_predictions(csv_path, image_dir, num_samples=3):
    df = pd.read_csv(csv_path)
    
    # Filter for valid detections
    valid_df = df[df['ellipseCenterX(px)'] != -1]
    
    if valid_df.empty:
        print("No valid detections found in CSV!")
        return

    unique_images = valid_df['inputImage'].unique()
    samples = np.random.choice(unique_images, min(len(unique_images), num_samples), replace=False)

    for img_id in samples:
        # Construct path - adjusting for subfolders if they exist in rel_no_ext
        img_path = os.path.join(image_dir, img_id + ".png")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Skipping: {img_path} not found.")
            continue

        detections = valid_df[valid_df['inputImage'] == img_id]
        
        for _, row in detections.iterrows():
            cx = int(row['ellipseCenterX(px)'])
            cy = int(row['ellipseCenterY(px)'])
            ma = int(row['ellipseSemimajor(px)'])
            mi = int(row['ellipseSemiminor(px)'])
            angle = row['ellipseRotation(deg)']
            cls = int(row['crater_classification'])

            # Colors for different classes (BGR)
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            color = colors[cls % len(colors)]

            # Draw the ellipse
            cv2.ellipse(img, (cx, cy), (ma, mi), angle, 0, 360, color, 3)
            # Center point
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image: {img_id} | Craters: {len(detections)}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Update these paths before running
    CSV_TO_CHECK = "/home/bora3i/crater_challenge/submission8_640/solution/solution2.csv"
    TEST_IMAGES_DIR = "./test" 
    
    draw_predictions(CSV_TO_CHECK, TEST_IMAGES_DIR)
