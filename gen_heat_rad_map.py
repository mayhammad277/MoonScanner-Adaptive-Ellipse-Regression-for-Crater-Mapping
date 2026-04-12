import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --- Configuration based on your final settings ---
MIN_AREA_FILTER = 17
# Kernel for minimal noise cleaning (Morphological OPEN)
KERNEL_SIZE = 15

def generate_gaussian_kernel(kernel_size, sigma):
    """Generates a 2D Gaussian kernel with peak value of 1."""
    kernel_radius = (kernel_size - 1) // 2
    x_grid = np.arange(0, kernel_size, 1, np.float32)
    y_grid = x_grid[:, np.newaxis]
    x0 = y0 = kernel_radius
    
    # Gaussian formula: e^(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
    gaussian_kernel = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
    return gaussian_kernel

# --- MAIN FUNCTION: Generate Multi-Channel Ground Truth ---

def generate_radius_ground_truth(input_path, mask_path):
    # 1. Load Images
    if not os.path.exists(input_path) or not os.path.exists(mask_path):
        print("Error: One or both image paths are invalid. Please check the 'input_image_path' and 'mask_image_path' variables.")
        return None, None
        
    try:
        original_image = cv2.imread(input_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None

    H, W = mask_image.shape
    
    # --- 2. Instance Separation and Feature Extraction ---
    
    # Otsu's Thresholding
    _, binary_mask = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Minimal Opening (to remove tiny noise/gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find Contours (RETR_EXTERNAL ensures separate, outer boundaries)
    contours, _ = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
    craters_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA_FILTER:
            
            # Minimum Enclosing Circle (MEC) - extracts center and R_max
            (x_center_float, y_center_float), R_max = cv2.minEnclosingCircle(contour)

            craters_list.append({
                'center_float': (x_center_float, y_center_float),
                'center_int': (int(x_center_float), int(y_center_float)),
                'R_max': R_max, # The parameter to regress
            })

    # --- 3. Multi-Channel Ground Truth Generation (H x W x 4) ---
    
    # Initialize a 4-channel tensor: [H_center, R_max, O_x, O_y]
    ground_truth_map = np.zeros((H, W, 4), dtype=np.float32) 
    
    heatmap_center = ground_truth_map[:, :, 0] # Channel 0
    radius_map = ground_truth_map[:, :, 1]     # Channel 1
    offset_map_x = ground_truth_map[:, :, 2]   # Channel 2
    offset_map_y = ground_truth_map[:, :, 3]   # Channel 3

    max_kernel_size = 101 # Cap for Gaussian kernel size

    for crater in craters_list:
        cx_float, cy_float = crater['center_float']
        cx, cy = crater['center_int'] # Integer center for array indexing
        R_max = crater['R_max']
        
        # --- Channel 0: Center Heatmap (Gaussian logic) ---
        sigma = R_max / 3.0
        if sigma < 1.0: sigma = 1.0
            
        kernel_radius = math.ceil(sigma * 3)
        kernel_size = 2 * kernel_radius + 1
        if kernel_size > max_kernel_size: 
            kernel_size = max_kernel_size
            kernel_radius = (kernel_size - 1) // 2
            
        gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
        
        # Calculate Image and Kernel Slices (Robust boundary handling)
        x_start_img = max(0, cx - kernel_radius)
        x_end_img = min(W, cx + kernel_radius + 1)
        y_start_img = max(0, cy - kernel_radius)
        y_end_img = min(H, cy + kernel_radius + 1)
        
        x_start_kernel = kernel_radius - (cx - x_start_img)
        x_end_kernel = kernel_radius + (x_end_img - cx) - 1
        y_start_kernel = kernel_radius - (cy - y_start_img)
        y_end_kernel = kernel_radius + (y_end_img - cy) - 1
        
        # Apply Heatmap (Channel 0): Use np.maximum for overlapping craters
        heatmap_center[y_start_img:y_end_img, x_start_img:x_end_img] = np.maximum(
            heatmap_center[y_start_img:y_end_img, x_start_img:x_end_img], 
            # Note: +1 for slice exclusivity
            gaussian_kernel[y_start_kernel:y_end_kernel + 1, x_start_kernel:x_end_kernel + 1]
        )
        
        # --- Channel 1: Radius Map (R_max) ---
        # Only set the radius value at the integer center
        if 0 <= cy < H and 0 <= cx < W:
            radius_map[cy, cx] = R_max

        # --- Channel 2 & 3: Offset Map (Correction for center discretization) ---
        offset_x = cx_float - cx
        offset_y = cy_float - cy
        
        if 0 <= cy < H and 0 <= cx < W:
            offset_map_x[cy, cx] = offset_x
            offset_map_y[cy, cx] = offset_y


    # --- 4. Visualization of Multi-Channel Maps ---
    
    # 4a. Heatmap Visualization
    heatmap_color = cv2.applyColorMap((heatmap_center * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 4b. Radius Map Visualization (Requires blurring and contrast enhancement)
    
    # Blur the sparse radius map for visual continuity
    vis_kernel = np.ones((5, 5), np.float32) / 25
    radius_blurred = cv2.filter2D(radius_map, -1, vis_kernel)
    
    # Normalize the blurred radius to [0, 1]
    max_radius_abs = np.max(radius_blurred) 
    if max_radius_abs == 0:
        max_radius_abs = 1
        
    radius_vis_norm = radius_blurred / max_radius_abs
    
    # Apply Gamma Correction (Non-linear boost to highlight high radius values)
    GAMMA = 0.5 
    radius_vis_boosted = np.power(radius_vis_norm, GAMMA)
    
    # Apply Colormap (MAGMA is often better than JET for emphasizing intensity)
    radius_map_color = cv2.applyColorMap((radius_vis_boosted * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    # --- Final Display ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    axes[0].imshow(original_image)
    axes[0].set_title('1. Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(heatmap_color)
    axes[1].set_title('2. Center Heatmap (Channel 0)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(radius_map_color)
    axes[2].set_title(f'3. Radius Map (Channel 1) - Gamma {GAMMA}', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"Total Craters Detected: {len(craters_list)}")
    print(f"Ground Truth Tensor Shape: {ground_truth_map.shape} (H x W x 4)")
    
    return ground_truth_map, craters_list
    # -----------------------------------------------------
    
    # ... (Rest of the plotting code remains the same) ...

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    # ... (plotting code) ...
    plt.tight_layout()
    plt.show()
    
    print(f"Ground Truth Tensor Shape: {ground_truth_map.shape} (H x W x 4)")
    return ground_truth_map, craters_list

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(heatmap_color)
    axes[1].set_title('Channel 0: Center Heatmap (Gaussian Peaks)', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(radius_map_color)
    axes[2].set_title('Channel 1: Radius Map (R$_{max}$ Regression Target)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    print(f"Ground Truth Tensor Shape: {ground_truth_map.shape} (H x W x 4)")
    return ground_truth_map, craters_list
input_image_path = "/home/bora3i/crater_challenge/train-sample/altitude01/longitude02/orientation09_light05.png"
mask_image_path = "/home/bora3i/crater_challenge/train-sample/altitude01/truth/orientation09_light05_mask.png"
# --- EXECUTE RADIUS GROUND TRUTH GENERATION ---
ground_truth_tensor, crater_detections = generate_radius_ground_truth(input_image_path, mask_image_path)
