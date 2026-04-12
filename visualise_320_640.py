import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_random_prediction(csv_path, data_dir):
    # Load the results
    df = pd.read_csv(csv_path)
    
    # Filter out empty/dummy detections (-1 values)
    df = df[df['ellipseCenterX(px)'] != -1]
    
    # Pick a random image that has detections
    available_images = df['inputImage'].unique()
    img_rel_path = np.random.choice(available_images)
    
    # Construct full path (assuming .png)
    full_path = os.path.join(data_dir, img_rel_path + ".png")
    img = cv2.imread(full_path)
    if img is None:
        print(f"Could not find image at {full_path}")
        return

    # Filter detections for this specific image
    detections = df[df['inputImage'] == img_rel_path]
    
    print(f"Drawing {len(detections)} craters for {img_rel_path}")

    for _, row in detections.iterrows():
        cx = int(row['ellipseCenterX(px)'])
        cy = int(row['ellipseCenterY(px)'])
        ma = int(row['ellipseSemimajor(px)'])
        mi = int(row['ellipseSemiminor(px)'])
        angle = row['ellipseRotation(deg)']
        
        # Draw the ellipse
        # Color based on classification (BGR format)
        color = (0, 255, 0) # Green for craters
        cv2.ellipse(img, (cx, cy), (ma, mi), angle, 0, 360, color, 2)
        
        # Small dot for center to check offset accuracy
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)

    # Display using Matplotlib
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predictions: {img_rel_path}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Change these to your actual paths
    CSV_FILE = "submission_320.csv"
    DATA_FOLDER = "./test_data"
    
    visualize_random_prediction(CSV_FILE, DATA_FOLDER)
