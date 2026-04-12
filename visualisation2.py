import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration based on your final settings ---
MIN_AREA_FILTER = 17
# Kernel for minimal noise cleaning (Morphological OPEN)
KERNEL_SIZE = 13


# --- File Paths (Placeholder) ---
input_image_path = "/home/bora3i/crater_challenge/train-sample/altitude01/longitude02/orientation04_light04.png"
mask_image_path = "/home/bora3i/crater_challenge/train-sample/altitude01/truth/orientation04_light04_truth.png"

# 1. Load Images
try:
    # Load raw input image (for context)
    input_image = cv2.imread(input_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Load mask image (the one to be processed)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

except FileNotFoundError:
    print("Error: One or both uploaded files were not found.")
    raise

# 2. Preprocess Mask
mask_gray = mask_image
    
# Use Otsu's method to reliably separate light grey craters from dark grey background
_, binary_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological Closing: Use a large 15x15 kernel to connect thick or broken boundaries
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)


# 3. Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# Plot 1: Raw Input Image
axes[0].imshow(input_image)
axes[0].set_title('1. Raw Input Image (Context)', fontsize=10)
axes[0].axis('off')

# Plot 2: Raw Mask Image (Before Thresholding)
axes[1].imshow(mask_image, cmap='gray')
axes[1].set_title('2. Raw Mask (Light/Dark Grey Craters)', fontsize=10)
axes[1].axis('off')

# Plot 3: Processed Binary Mask (After Otsu + 15x15 Closing)
axes[2].imshow(processed_mask, cmap='gray')
axes[2].set_title('3. Final Binary Mask (Input to Contours)', fontsize=10)
axes[2].axis('off')

plt.tight_layout()
plt.show()

def separate_and_verify_craters(input_path, mask_path):
    # 1. Load Images
    try:
        original_image = cv2.imread(input_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    visual_image = original_image.copy()

    # --- 2. Instance Separation Strategy (The Fix) ---
    mask_gray = mask_image
    
    # a. Otsu's Thresholding
    _, binary_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # b. Minimal Opening (Remove tiny noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # c. Find ALL Contours (Individual Craters) from the mask
    contours, _ = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # d. Create a brand new, empty mask to draw only SEPARATED components
    final_clean_mask = np.zeros_like(mask_gray)

    # e. Draw EACH contour back onto the clean mask (This enforces separation)
    cv2.drawContours(final_clean_mask, contours, -1, 255, thickness=cv2.FILLED) 
    
    # --- 3. Contour Finding and Feature Extraction ---
    # We use the contours found in step 2.c for the geometric analysis
    
    craters_list = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Only filter by minimum area
        if area > MIN_AREA_FILTER:
            
            # Calculate Minimum Enclosing Circle (MEC) properties (R_max and Center)
            (x_center, y_center), R_max = cv2.minEnclosingCircle(contour)
            center = (int(x_center), int(y_center))
            radius = int(R_max)
            
            # Store crater data for sorting
            craters_list.append({
                'center': center,
                'radius': radius,
                'size_metric': R_max * R_max 
            })
            
    # Sort to identify the absolute largest one (for the Red marker)
    craters_list_sorted = sorted(craters_list, key=lambda c: -c['size_metric'])
    
    print(f"Total contours detected and processed: {len(craters_list)}")
    
    # --- 4. Visualization: Draw Centers and Boundaries ---
    
    for crater in craters_list_sorted:
        center = crater['center']
        radius = crater['radius']
        
        # Color the absolute LARGEST crater's bounding circle RED for verification
        color = (255, 0, 0) if crater == craters_list_sorted[0] else (0, 255, 255) # Red for largest, Cyan for others
        
        # Draw the large bounding circle (Boundary)
        cv2.circle(visual_image, center, radius, color, 2)
        
        # Draw a small filled circle (Center)
        cv2.circle(visual_image, center, 3, (255, 255, 0), -1) # Yellow center dot for visibility

    # --- 5. Final Display ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(final_clean_mask, cmap='gray')
    axes[0].set_title('Processed Binary Mask (Instance Separated)', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(visual_image)
    axes[1].set_title('Detection Verification: Largest Crater in RED', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# --- EXECUTE THE VERIFICATION ---
# NOTE: Ensure the file paths are correctly set before running!
separate_and_verify_craters(input_image_path, mask_image_path)
