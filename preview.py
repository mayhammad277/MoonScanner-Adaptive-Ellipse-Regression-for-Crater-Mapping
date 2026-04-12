import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_craters(csv_path, image_root, num_samples=5):
    # Load the submission file
    df = pd.read_csv(csv_path)
    
    # Filter out the dummy -1 rows for visualization
    valid_df = df[df['ellipseCenterX(px)'] != -1]
    
    if valid_df.empty:
        print("No craters found in CSV to plot!")
        return

    # Pick random images to verify
    unique_images = valid_df['inputImage'].unique()
    samples = np.random.choice(unique_images, min(len(unique_images), num_samples), replace=False)

    # Color mapping for Mahanti classes (A, AB, B, BC, C)
    class_colors = {
        0: (0, 255, 0),    # Class A: Green
        1: (255, 255, 0),  # Class AB: Yellow
        2: (255, 165, 0),  # Class B: Orange
        3: (0, 0, 255),    # Class BC: Blue
        4: (128, 0, 128),  # Class C: Purple
        -1: (255, 255, 255) # Unclassified: White
    }

    for img_id in samples:
        # Reconstruct path (img_id is folder/folder/name)
        img_path = os.path.join(image_root, img_id + ".png")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not find image at {img_path}")
            continue

        # Get all craters for this image
        craters = valid_df[valid_df['inputImage'] == img_id]
        
        for _, row in craters.iterrows():
            cx = int(row['ellipseCenterX(px)'])
            cy = int(row['ellipseCenterY(px)'])
            ma = int(row['ellipseSemimajor(px)'])
            mi = int(row['ellipseSemiminor(px)'])
            angle = row['ellipseRotation(deg)']
            cls = int(row['crater_classification'])
            
            color = class_colors.get(cls, (255, 255, 255))

            # Draw the ellipse
            # Note: cv2.ellipse expects angle in degrees clockwise
            cv2.ellipse(img, (cx, cy), (ma, mi), angle, 0, 360, color, 3)
            
            # Draw center point
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        # Convert BGR to RGB for plotting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(img_rgb)
        plt.title(f"Image ID: {img_id}\nDetections: {len(craters)}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # CONFIGURATION
    SUBMISSION_CSV = "../solution/solution.csv" 
    TEST_DATA_ROOT = "../../test" # Path where your 'altitude/longitude/...' folders start
    
    plot_craters(SUBMISSION_CSV, TEST_DATA_ROOT)
