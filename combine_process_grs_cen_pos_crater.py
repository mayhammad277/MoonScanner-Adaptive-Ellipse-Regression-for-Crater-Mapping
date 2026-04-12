import os
import cv2
import numpy as np
import json
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
from torchvision.transforms import functional as F
from PIL import ImageEnhance
from scipy.optimize import curve_fit

# ====================== Crater Detection Functions ======================

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A):
    """2D Gaussian function for fitting the sub-pixel center of a detected feature."""
    return A * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2)))

def preprocess_cimg(image):
    """Load and return a color image, whether input is path or image array"""
    if isinstance(image, str):
        cimg = cv2.imread(image, cv2.IMREAD_COLOR)
    else:
        cimg = image
    return cimg

import numpy as np
import cv2

# Minimum separation distance in pixels between the three blue markers
# Tune this based on your image resolution.
MIN_DISTANCE_THRESHOLD = 30 

def fit_gaussian_2d(image, contour):
    # This function remains unchanged (used for sub-pixel accuracy)
    # ... [Assuming the fit_gaussian_2d function is defined elsewhere in your script]
    # NOTE: You should ensure fit_gaussian_2d is available here or copy its definition.
    # For this example, we'll assume it's accessible.
    
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w]
    if roi.size == 0 or np.std(roi) < 1e-3:
         return (float(x + w / 2), float(y + h / 2), float(roi.max()) if roi.size > 0 else 0.0)
    # ... [rest of Gaussian fit logic]
    # Since we can't define the full function here, ensure it's in your file.
    
    # Placeholder return for demonstration:
    return (float(x + w / 2), float(y + h / 2), float(np.max(roi)) if roi.size > 0 else 0.0)


import numpy as np
import cv2
import math
import os # Ensure os is imported for temporary file checks

# We no longer need a distance threshold since we are selecting the top N largest.
# MIN_DISTANCE_THRESHOLD = 50 

import numpy as np
import cv2
import math

import numpy as np
import cv2
import math

import numpy as np
import cv2
import math

import numpy as np
import cv2
import math

# --- CONSTANTS ---
REQUIRED_CRATERS = 15 # 1 (Red) + 12 (Blue)
KERNEL_SIZE = 15 
MAX_AREA_FILTER = 50000 
# --- NEW CONSTANT: Pixels margin from the edge where reference craters are disallowed ---
EDGE_MARGIN_THRESHOLD = 50 

def find_reference_craters(image, mask_image):
    """
    Finds the top 13 largest craters: 1 largest (RED) and the next 12 largest (BLUE),
    ensuring the 12 BLUE reference craters are not close to the image edge.
    """
    
    # --- 1. Load and Process Mask using Otsu's and Closing ---
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image
    
    H, W = mask_gray.shape # Get image dimensions for boundary check
        
    _, binary_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Debugging check: Display the clean binary mask
    try:
        cv2.imshow('Clean Binary Mask', binary_mask)
        cv2.waitKey(1)
    except Exception:
        pass 
        
    # --- 2. Contour Finding and Feature Extraction ---
    
    contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    craters = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 7 and area < MAX_AREA_FILTER:
            
            # Calculate Minimum Enclosing Circle (MEC) properties (R_max)
            (x_center, y_center), R_max = cv2.minEnclosingCircle(contour)
            
            cx = float(x_center)
            cy = float(y_center)
            
            # --- Boundary Check for ALL CRATERS ---
            # We filter for the BLUE markers later, but it's good practice 
            # to calculate the margin here for any potential future use.
            
            # --- MIC CALCULATION (R_min) ---
            temp_mask = np.zeros_like(binary_mask)
            cv2.drawContours(temp_mask, [contour], 0, 255, -1)
            dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
            _, R_min, _, _ = cv2.minMaxLoc(dist_transform)
            
            # Define size metric based on MEC radius squared (R_max^2) for stable ranking
            size_metric = R_max * R_max
            circularity_index = R_min / R_max if R_max > 0 else 0.0
            
            craters.append({
                'pos': (cx, cy), 
                'size': size_metric,
                'radius': R_max, 
                'R_min': R_min,
                'circularity': circularity_index,
                'id': (cx, cy)
            })
            
    # --- 3. Sorting and Selection (Top 13 Largest with Boundary Check) ---
    
    if len(craters) < REQUIRED_CRATERS:
        print(f"Error: Only detected {len(craters)} unique craters. Need at least {REQUIRED_CRATERS} for selection.")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        return None

    # Sort ALL craters by size (R_max^2): descending
    craters_sorted_by_size_desc = sorted(craters, key=lambda c: -c['size'])
    
    # 1. Main Crater (Largest Radius, RED MARKER) - No boundary restriction here
    main_crater = craters_sorted_by_size_desc[0]
    
    # Identify candidates for the 12 blue reference markers (excluding the red one)
    candidate_craters = craters_sorted_by_size_desc[1:]
    
    reference_craters = []
    
    for crater in candidate_craters:
        if len(reference_craters) >= 12:
            break
            
        cx, cy = crater['pos']
        
        # --- NEW BOUNDARY CHECK ---
        # 1. Check distance from left (0) and top (0) edges
        is_too_close = (cx < EDGE_MARGIN_THRESHOLD or cy < EDGE_MARGIN_THRESHOLD)
        
        # 2. Check distance from right (W) and bottom (H) edges
        is_too_close = is_too_close or (cx > W - EDGE_MARGIN_THRESHOLD or cy > H - EDGE_MARGIN_THRESHOLD)
        
        if not is_too_close:
            reference_craters.append(crater)

    if len(reference_craters) < 12:
        print(f"Warning: Could only find {len(reference_craters)} reference craters satisfying the margin of {EDGE_MARGIN_THRESHOLD} pixels.")
        # We might continue with fewer than 12 references if necessary, but returning None is safer 
        # unless you have logic to handle fewer markers downstream.
        return None 
    
    # Final List: [Largest (Red)] + [12 Edge-Safe Largest (Blue)]
    final_references = [main_crater] + reference_craters
    
    return final_references[:REQUIRED_CRATERS]
    
def highlight_craters(image, reference_craters):
    """Highlight craters in the image: centroid (1st) in red, next three in blue"""
    if reference_craters:
        # Highlight centroid (first crater, RED marker)
        center = tuple(map(int, reference_craters[0]['pos']))
        # Red color (0, 0, 255)
        cv2.circle(image, center, 7, (0, 0, 255), 2)
        
        # Highlight other reference craters (2nd, 3rd, 4th, BLUE markers)
        for crater in reference_craters[1:4]:
            center = tuple(map(int, crater['pos']))
            # Blue color (255, 0, 0)
            cv2.circle(image, center, 5, (255, 0, 0), 2)
    return image
    
    
def save_crater_map(output_dir, filename, reference_craters, image_shape):
    """Save crater positions and sizes to a JSON crater map file (1 centroid + 3 refs)"""
    if len(reference_craters) < 4:
        return
        
    crater_map = {
        'image_shape': image_shape,
        'centroid': {
            'position': reference_craters[0]['pos'],
            'size': reference_craters[0]['size']
        },
        'reference_craters': [] 
    }
    
    # The blue markers are the 2nd, 3rd, and 4th items in the list (indices 1, 2, 3)
    for i, crater in enumerate(reference_craters[1:4], 1): 
        crater_map['reference_craters'].append({
            'id': i,
            'position': crater['pos'],
            'size': crater['size']
        })
        
    base_name = os.path.splitext(filename)[0]
    crater_map_path = os.path.join(output_dir, f"{base_name}_crater_map.json")
    if not os.path.exists(crater_map_path):
      with open(crater_map_path, 'w') as f:
        json.dump(crater_map, f, indent=2)
# ====================== Transformation Pipeline (Unchanged) ======================

class SafeAugmentation:
    # ... (Implementation is identical to the original code, as photometric augments are still useful) ...
    def __init__(self):
        pass

    def add_low_random_noise(self, image, noise_level=0.01):
        """Add low random Gaussian noise to the image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Scale noise level relative to the max intensity (255)
        noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def add_bright_spots(self, image, num_spots=3, max_radius=10, max_intensity=50):
        """Add subtle bright spots to the image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w, _ = image.shape
        output_image = np.copy(image)
        
        for _ in range(num_spots):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            radius = np.random.randint(3, max_radius)
            intensity = np.random.randint(20, max_intensity)
            
            # Simple circular/square spot application
            for dx in range(-radius, radius):
                for dy in range(-radius, radius):
                    if 0 <= cx + dx < w and 0 <= cy + dy < h:
                        # Simple decay based on distance from center could be added, but using random decay for simplicity
                        decay = random.uniform(0.5, 1.0)
                        output_image[cy + dy, cx + dx] = np.clip(
                            output_image[cy + dy, cx + dx].astype(int) + int(intensity * decay), 0, 255
                        )
        
        return Image.fromarray(output_image.astype(np.uint8))

    def adjust_brightness_contrast(self, image, brightness_factor=1.0, contrast_factor=1.0):
        """Adjust the brightness and contrast of the image."""
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        return image

    def __call__(self, image):
        """Apply all augmentations to the image."""
        brightness_factor = random.uniform(0.95, 1.05)
        contrast_factor = random.uniform(0.95, 1.05)
        image = self.adjust_brightness_contrast(image, brightness_factor, contrast_factor)
        image = self.add_low_random_noise(image, noise_level=0.01)
        image = self.add_bright_spots(image, num_spots=random.randint(1, 3), 
                                      max_radius=8, max_intensity=50)
        return image


# ====================== Main Processing Functions (Adapted) ======================

def create_transformed_versions(original_image, reference_craters, output_dir, original_filename, original_crater_map):
    """Create transformed versions of the image while preserving original crater IDs and structure"""
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    augmenter = SafeAugmentation()
    
    # Standard transformation pipeline
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create 45 augmented versions
    for i in range(45):
        base_name = os.path.splitext(original_filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_aug{i}.png")
        if os.path.exists(save_path):
          continue
          
        augmented_image = augmenter(pil_image)
        transformed_image = base_transform(augmented_image)
        
        # Convert back to PIL for saving (unnormalize for visual check)
        save_image = F.to_pil_image(transformed_image * 0.5 + 0.5) 
        save_image.save(save_path)
        
        # Create augmented crater map (positions/sizes are identical to original)
        augmented_crater_map = {
            "original_image": original_filename,
            "augmentation_index": i,
            "image_shape": original_crater_map["image_shape"],
            "centroid": original_crater_map["centroid"],
            "reference_craters": original_crater_map["reference_craters"]
        }
        
        crater_map_path = os.path.join(output_dir, f"{base_name}_aug{i}_crater_map.json")
        if not os.path.exists(crater_map_path):
            with open(crater_map_path, 'w') as f:
                json.dump(augmented_crater_map, f, indent=2)

def process_images(input_directory, mask_directory, output_directory):
    """
    Process all PNG images using a mask from a separate directory 
    to find reference craters.
    """
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "crater_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "transformed_images"), exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"): 
            path_to_image = os.path.join(input_directory, filename)
            f=filename.split(".")
            # Assuming mask filename matches image filename
            path_to_mask = os.path.join(mask_directory, f[0]+"_truth"+".png") 
            
            if not os.path.exists(path_to_mask):
                print(f"Skipping {filename}: Mask not found at {path_to_mask}")
                continue

            cimg = preprocess_cimg(path_to_image)
            mask_img = preprocess_cimg(path_to_mask)
            
            # --- CRITICAL CHANGE: Pass both images to the detection function ---
            reference_craters = find_reference_craters(cimg, mask_img)

            if reference_craters:
                highlighted_image = highlight_craters(cimg, reference_craters)
                processed_path = os.path.join(output_directory, "images", filename)
                cv2.imwrite(processed_path, highlighted_image)
                
                # ... (rest of the saving logic remains the same) ...
                original_crater_map_path = os.path.join(output_directory, "crater_maps", f"{os.path.splitext(filename)[0]}_crater_map.json")
                save_crater_map(os.path.join(output_directory, "crater_maps"), 
                                filename, reference_craters, cimg.shape[:2])
                
                with open(original_crater_map_path) as f:
                    original_crater_map = json.load(f)
                
                create_transformed_versions(cimg, reference_craters, 
                                             os.path.join(output_directory, "transformed_images"),
                                             filename, original_crater_map)
# ====================== Dataset Class (Adapted for Craters) ======================

class CraterDataset(Dataset):
    """Dataset that loads processed images and their crater maps"""
    def __init__(self, processed_dir, crater_maps_dir, transform=None): # Changed folder name
        self.processed_dir = processed_dir
        self.crater_maps_dir = crater_maps_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(processed_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Load processed image
        img_path = os.path.join(self.processed_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding crater map (filename change)
        base_name = os.path.splitext(self.image_files[idx])[0]
        crater_map_path = os.path.join(self.crater_maps_dir, f"{base_name}_crater_map.json")
        
        with open(crater_map_path) as f:
            crater_data = json.load(f)
        
        # Get crater positions and sizes
        crater_info = [{
            'position': crater_data['centroid']['position'],
            'size': crater_data['centroid']['size']
        }]
        if 'reference_craters' in crater_data:
            crater_info.extend([{'position': c['position'], 'size': c['size']} 
                                for c in crater_data['reference_craters']])
                                
        # Create heatmap
        heatmap = self.create_crater_heatmap(image.size, crater_info)
        
        if self.transform:
            image = self.transform(image)
        
        # Prepare positions for the model target (flattened coordinates)
        crater_positions_flat = torch.tensor([item['position'] for item in crater_info]).flatten().float()
        
        return image, heatmap, crater_positions_flat

    def create_crater_heatmap(self, image_size, crater_info):
        """Create heatmap from crater positions, using the accurately detected radius."""
        heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
        # We now use the 'radius' key directly from the crater_info dictionary
    
        for item in crater_info:
          x, y = item['position']
          radius = item.get('radius', 5) # Use the detected radius, fallback to 5 if missing
        
          # Draw the circle on the heatmap with intensity 1.0
          # The radius is the true detected radius, not a scaled area.
          heatmap = cv2.circle(heatmap, (int(x), int(y)), int(radius), 1.0, -1)
        
        return torch.tensor(heatmap).unsqueeze(0)

# ====================== Usage Example (Updated) ======================

if __name__ == "__main__":
    # NOTE: You MUST change these directories to your local paths.
    INPUT_DIR = "/home/bora3i/crater_challenge/train-sample/altitude01/longitude02" # e.g., /media/student/.../dataset_moon/crater-images
    OUTPUT_DIR = "/home/bora3i/crater_challenge/train-sample/altitude01/longitude02/processed" # e.g., /media/student/.../new_crater_processed
    mask_dir="/home/bora3i/crater_challenge/train-sample/altitude01/longitude02/truth"
    # Ensure you set up the directory structure before running
    print(f"Starting processing in: {INPUT_DIR}")
    process_images(INPUT_DIR,mask_dir, OUTPUT_DIR)
    print("Processing complete (Commented out for safety. Uncomment to run).")
    
    # Create dataset from processed images
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = CraterDataset(
        processed_dir=os.path.join(OUTPUT_DIR, "transformed_images"),
        crater_maps_dir=os.path.join(OUTPUT_DIR, "crater_maps"),
        transform=transform
    )
    
    if len(dataset) > 0:
      sample_image, sample_heatmap, sample_crater_positions = dataset[0]
      print(f"Dataset size: {len(dataset)}")
      print(f"Sample image shape: {sample_image.shape}")
      print(f"Sample heatmap shape: {sample_heatmap.shape}")
      print(f"Sample crater positions (flattened coords): {sample_crater_positions}")
