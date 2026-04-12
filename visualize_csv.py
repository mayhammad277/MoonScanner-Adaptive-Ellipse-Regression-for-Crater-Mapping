import pandas as pd
import cv2
import os
import random
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_FILE = "test_predictions_v3.csv"
TEST_ROOT = "/home/bora3i/crater_challenge/test/"
OUTPUT_PATH = "final_verification.png"

def visualize_random_prediction():
    # 1. Load the CSV
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found!")
        return
    
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        print("⚠️ CSV is empty. No detections to visualize.")
        return

    # 2. Pick a random image that has detections
    unique_images = df['image_id'].unique()
    random_img_rel_path = random.choice(unique_images)
    full_img_path = os.path.join(TEST_ROOT, random_img_rel_path)
    
    print(f"🧐 Visualizing results for: {random_img_rel_path}")

    # 3. Load the image
    image = cv2.imread(full_img_path)
    if image is None:
        print(f"❌ Error: Could not load image at {full_img_path}")
        return
    
    # Convert BGR to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 4. Filter detections for this specific image
    img_dets = df[df['image_id'] == random_img_rel_path]

    # 5. Draw detections
    for _, row in img_dets.iterrows():
        cx, cy, r = int(row['cx']), int(row['cy']), int(row['radius'])
        conf = row['confidence']
        
        # Draw Circle
        cv2.circle(image_rgb, (cx, cy), r, (0, 255, 0), 3) # Green circle
        
        # Add Confidence Label
        label = f"{conf:.2f}"
        cv2.putText(image_rgb, label, (cx, cy - r - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 6. Display and Save
    plt.figure(figsize=(12, 12))
    plt.imshow(image_rgb)
    plt.title(f"Detections for {random_img_rel_path} ({len(img_dets)} craters)")
    plt.axis('off')
    plt.savefig(OUTPUT_PATH)
    plt.show()
    
    print(f"✅ Visualization saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    visualize_random_prediction()
