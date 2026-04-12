import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# --- Configuration based on your final settings ---
MIN_AREA_FILTER = 17
# Kernel for minimal noise cleaning (Morphological OPEN)
KERNEL_SIZE = 13


def generate_heatmap_input(input_path, mask_path):
    # 1. Load Images (Same as before)
    try:
        original_image = cv2.imread(input_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    H, W = mask_image.shape
    
    # --- 2. Instance Separation Strategy (Same as before) ---
    mask_gray = mask_image
    
    # a. Otsu's Thresholding
    _, binary_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # b. Minimal Opening (Remove tiny noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # c. Find ALL Contours
    contours, _ = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # d. Contour Filtering and Feature Extraction (Same as before)
    craters_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > MIN_AREA_FILTER:
            # Calculate Minimum Enclosing Circle (MEC) properties (R_max and Center)
            (x_center, y_center), R_max = cv2.minEnclosingCircle(contour)
            center = (int(x_center), int(y_center))
            radius = int(R_max)
            
            craters_list.append({
                'center': center,
                'radius': radius,
                'R_max': R_max # Keep the float value for sigma calculation
            })
            
    # --- 3. Heatmap Generation (The NEW Core Step) ---
    
    # Initialize the Heatmap: A zero array with the same dimensions as the image.
    # If the Swin Transformer expects multiple channels (e.g., center, width, height),
    # you would initialize a H x W x C array. Here we use 1 channel for simplicity.
    heatmap = np.zeros((H, W), dtype=np.float32)

    # Define the maximum acceptable Gaussian size (e.g., 6*sigma)
    max_kernel_size = 101 # Odd number for center alignment

    for crater in craters_list:
        cx, cy = crater['center']
        R_max = crater['R_max']
        
        # Calculate sigma and kernel size (same as before)
        sigma = R_max / 3.0
        if sigma < 1.0:
            sigma = 1.0
            
        kernel_radius = math.ceil(sigma * 3)
        kernel_size = 2 * kernel_radius + 1
        
        # Create the 2D Gaussian kernel (same as before)
        x_grid = np.arange(0, kernel_size, 1, np.float32)
        y_grid = x_grid[:, np.newaxis]
        x0 = y0 = kernel_radius
        gaussian_kernel = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))

        # --- CORRECTION STARTS HERE ---
        
        # 1. Determine the boundaries for the Heatmap (H, W are image height/width)
        # x-axis boundaries
        x_start_img = max(0, cx - kernel_radius)
        x_end_img = min(W, cx + kernel_radius + 1)
        
        # y-axis boundaries
        y_start_img = max(0, cy - kernel_radius)
        y_end_img = min(H, cy + kernel_radius + 1)
        
        # 2. Determine the corresponding boundaries for the Kernel
        # If the image start is > 0, we must slice the kernel.
        x_start_kernel = kernel_radius - (cx - x_start_img)
        x_end_kernel = kernel_radius + (x_end_img - cx) - 1
        
        y_start_kernel = kernel_radius - (cy - y_start_img)
        y_end_kernel = kernel_radius + (y_end_img - cy) - 1
        
        # 3. Final Check: Ensure the slices are the same size before operation
        if (x_end_img - x_start_img) != (x_end_kernel - x_start_kernel + 1):
             print(f"Mismatch X: Image span ({x_end_img - x_start_img}) vs Kernel span ({x_end_kernel - x_start_kernel + 1})")
             continue # Skip this crater if there's a serious calculation error
        
        # 4. Perform the operation with guaranteed matching shapes
        heatmap[y_start_img:y_end_img, x_start_img:x_end_img] = np.maximum(
            heatmap[y_start_img:y_end_img, x_start_img:x_end_img], 
            gaussian_kernel[y_start_kernel:y_end_kernel + 1, x_start_kernel:x_end_kernel + 1]
        )
        # Note: We use +1 for the kernel end slice because the index calculation
        # above gives the last *included* index, not the exclusive end index for Python slicing.


    # --- 4. Visualization: Display Heatmap ---
    
    # Create an image to overlay the heatmap on for better context
    heatmap_vis = np.stack([heatmap, heatmap, heatmap], axis=-1) * 255
    heatmap_vis = heatmap_vis.astype(np.uint8)
    
    # Convert to a color map for better visualization (like JET)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(heatmap_color)
    axes[1].set_title('Generated Crater Heatmap (Input for Swin Transformer)', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"Heatmap shape: {heatmap.shape} (H x W)")
    return heatmap, craters_list

input_image_path = "/home/bora3i/crater_challenge/train-sample/altitude01/longitude02/orientation09_light05.png"
mask_image_path = "/home/bora3i/crater_challenge/train-sample/altitude01/truth/orientation09_light05_mask.png"
heatmap_data, crater_detections = generate_heatmap_input(input_image_path, mask_image_path)
