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

# ====================== Star Detection Functions ======================

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A):
    return A * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)))

def preprocess_cimg(image):
    """Load and return a color image, whether input is path or image array"""
    if isinstance(image, str):
        cimg = cv2.imread(image, cv2.IMREAD_COLOR)
    else:
        cimg = image
    return cimg



def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A):
    return A * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2)))

def fit_gaussian_2d(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w]

    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3 or np.std(roi) < 1e-3:
        return (int(x + w // 2), int(y + h // 2), float(roi.max()) if roi.size > 0 else 0.0)

    X, Y = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
    X = X.ravel()
    Y = Y.ravel()
    Z = roi.ravel().astype(np.float64)

    initial_guess = (w / 2, h / 2, 1.0, 1.0, float(np.max(Z)))
    bounds = (
        (0, 0, 0.5, 0.5, 0),                # lower bounds
        (w, h, w, h, float(np.max(Z) * 2))  # upper bounds
    )

    try:
        popt, _ = curve_fit(
            lambda data, x0, y0, sigma_x, sigma_y, A: gaussian_2d(data[0], data[1], x0, y0, sigma_x, sigma_y, A),
            (X, Y), Z, p0=initial_guess, bounds=bounds, maxfev=10000
        )
        cx, cy, amp = float(popt[0] + x), float(popt[1] + y), float(popt[4])
        return (cx, cy, amp)
    except Exception as e:
        print("Gaussian fit did not converge:", e)
        return (float(x + w // 2), float(y + h // 2), float(np.max(Z)) if Z.size > 0 else 0.0)

def find_reference_stars(image):
    """Find the centroid star and two brightest nearby stars"""
    # Convert to grayscale if needed
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Threshold and process image
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph_ellipse = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
    dilation_img = cv2.dilate(thresholded, kernel=morph_ellipse, iterations=1)
    opening_img = cv2.morphologyEx(dilation_img, cv2.MORPH_OPEN, morph_ellipse)

    # Find contours
    contours, _ = cv2.findContours(opening_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find centroids and brightness of significant contours
    stars = []
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Filter small contours
            x, y, brightness = fit_gaussian_2d(gray_image, contour)
            stars.append({'pos': (x, y), 'brightness': brightness})
    
    if not stars:
        return None

    # Find image center
    image_center = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)
    
    # Find centroid star (closest to center)
    centroid_star = min(stars, key=lambda s: np.linalg.norm(np.array(s['pos']) - np.array(image_center)))
    
    # Remove centroid from stars list
    other_stars = [s for s in stars if s['pos'] != centroid_star['pos']]
    
    # Find two brightest stars closest to centroid
    other_stars_sorted = sorted(other_stars, 
                              key=lambda s: (-s['brightness'],  # Brightest first
                                            np.linalg.norm(np.array(s['pos']) - np.array(centroid_star['pos']))))
    
    reference_stars = [centroid_star]
    if len(other_stars_sorted) >= 1:
        reference_stars.append(other_stars_sorted[0])
    if len(other_stars_sorted) >= 2:
        reference_stars.append(other_stars_sorted[1])
    
    return reference_stars[:3]  # Return max 3 stars

def highlight_stars(image, reference_stars):
    """Highlight stars in the image: centroid in red, others in blue"""
    if reference_stars:
        # Highlight centroid (first star) in red
        center = tuple(map(int, reference_stars[0]['pos']))
        cv2.circle(image, center, 7, (0, 0, 255), -1)
        
        # Highlight other reference stars in blue
        for star in reference_stars[1:3]:
          center = tuple(map(int, star['pos']))  # Convert (float, float) → (int, int)
          cv2.circle(image, center, 5, (255, 0, 0), -1)
          #cv2.circle(image, star['pos'], 5, (255, 0, 0), -1)
    return image

def save_star_map(output_dir, filename, reference_stars, image_shape):
    """Save star positions to a JSON star map file"""
    if not reference_stars:
        return
    
    star_map = {
        'image_shape': image_shape,
        'centroid': {
            'position': reference_stars[0]['pos'],
            'brightness': reference_stars[0]['brightness']
        },
        'reference_stars': []
    }
    
    for i, star in enumerate(reference_stars[1:3], 1):
        star_map['reference_stars'].append({
            'id': i,
            'position': star['pos'],
            'brightness': star['brightness']
        })
    
    # Create star map filename by replacing image extension with .json
    base_name = os.path.splitext(filename)[0]
    star_map_path = os.path.join(output_dir, f"{base_name}_star_map.json")
    if(not os.path.exists(star_map_path)):
      with open(star_map_path, 'w') as f:
        json.dump(star_map, f, indent=2)

# ====================== Transformation Pipeline ======================

class SafeAugmentation:
    def __init__(self):
        pass

    def add_low_random_noise(self, image, noise_level=0.01):
        """Add low random Gaussian noise to the image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image + noise
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
            
            for dx in range(-radius, radius):
                for dy in range(-radius, radius):
                    if 0 <= cx + dx < w and 0 <= cy + dy < h:
                        decay = random.uniform(0.5, 1.0)
                        output_image[cy + dy, cx + dx] = np.clip(
                            output_image[cy + dy, cx + dx] + int(intensity * decay), 0, 255
                        )
        
        return Image.fromarray(output_image)

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

# ====================== Main Processing Function ======================
"""
def process_images(input_directory, output_directory):
    #Process all PNG images in input directory, find reference stars, and save results
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, "processed_images"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "star_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "transformed_images"), exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"): 
            path_to_image = os.path.join(input_directory, filename)
            cimg = preprocess_cimg(path_to_image)
            
            # Find reference stars
            reference_stars = find_reference_stars(cimg)
            
            if reference_stars:
                # Highlight stars in image and save processed version
                highlighted_image = highlight_stars(cimg, reference_stars)
                processed_path = os.path.join(output_directory, "processed_images", filename)
                cv2.imwrite(processed_path, highlighted_image)
                
                # Save star map
                save_star_map(os.path.join(output_directory, "star_maps"), 
                            filename, reference_stars, cimg.shape[:2])
                
                # Create and save transformed versions
                create_transformed_versions(cimg, reference_stars, 
                                          os.path.join(output_directory, "transformed_images"),
                                          filename)

def create_transformed_versions(original_image, reference_stars, output_dir, original_filename):
   #Create transformed versions of the image using the star positions
    # Convert OpenCV image to PIL
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    # Get star positions from reference_stars
    star_positions = [star['pos'] for star in reference_stars]
    
    # Initialize augmenter
    augmenter = SafeAugmentation()
    
    # Create base transform
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create 5 augmented versions
    for i in range(20):
        # Apply augmentations
        augmented_image = augmenter(pil_image)
        
        # Apply base transform
        transformed_image = base_transform(augmented_image)
        
        # Convert back to PIL for saving
        save_image = F.to_pil_image(transformed_image * 0.5 + 0.5)  # Unnormalize
        
        # Save filename
        base_name = os.path.splitext(original_filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_aug{i}.png")
        save_image.save(save_path)
        
        # Save corresponding star positions (same as original)
        star_map = {
            'original_image': original_filename,
            'augmentation_index': i,
            'star_positions': star_positions
        }
        
        star_map_path = os.path.join(output_dir, f"{base_name}_aug{i}_stars.json")
        with open(star_map_path, 'w') as f:
            json.dump(star_map, f, indent=2)
"""


def create_transformed_versions(original_image, reference_stars, output_dir, original_filename, original_star_map):
    """Create transformed versions of the image while preserving original star IDs and structure"""
    # Convert OpenCV image to PIL
    pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    # Initialize augmenter
    augmenter = SafeAugmentation()
    
    # Create base transform
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create 5 augmented versions
    for i in range(45):
        base_name = os.path.splitext(original_filename)[0]
        save_path = os.path.join(output_dir, f"{base_name}_aug{i}.png")
        if os.path.exists(save_path):
          continue    
        # Apply augmentations
        augmented_image = augmenter(pil_image)
        
        # Apply base transform
        transformed_image = base_transform(augmented_image)
        
        # Convert back to PIL for saving
        save_image = F.to_pil_image(transformed_image * 0.5 + 0.5)  # Unnormalize
        
        # Save filename

        save_image.save(save_path)
        
        # Create augmented star map with same structure and IDs as original
        augmented_star_map = {
            "original_image": original_filename,
            "augmentation_index": i,
            "image_shape": original_star_map["image_shape"],
            "centroid": original_star_map["centroid"],
            "reference_stars": original_star_map["reference_stars"]
        }
        
        star_map_path = os.path.join(output_dir, f"{base_name}_aug{i}_star_map.json")
        with open(star_map_path, 'w') as f:
            if os.path.exists(star_map_path):
              continue 
            
            json.dump(augmented_star_map, f, indent=2)

def process_images(input_directory, output_directory):
    """Process all PNG images in input directory, find reference stars, and save results"""
    os.makedirs(output_directory, exist_ok=True)
    #os.makedirs(os.path.join(output_directory, "processed_images"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "star_maps"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "transformed_images"), exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"): 
            path_to_image = os.path.join(input_directory, filename)
            processed_path = os.path.join(output_directory, "images", filename)
            if  os.path.exists(processed_path):
                continue            
            cimg = preprocess_cimg(path_to_image)
            
            # Find reference stars
            
            reference_stars = find_reference_stars(cimg)


            if reference_stars:
                # Highlight stars in image and save processed version
                highlighted_image = highlight_stars(cimg, reference_stars)
                
 
                  
                cv2.imwrite(processed_path, highlighted_image)
                
                  # Save original star map
                original_star_map_path = os.path.join(output_directory, "star_maps", f"{os.path.splitext(filename)[0]}_star_map.json")
                save_star_map(os.path.join(output_directory, "star_maps"), 
                            filename, reference_stars, cimg.shape[:2])
                
                # Load the original star map to pass to transformation function
                with open(original_star_map_path) as f:
                    original_star_map = json.load(f)
                
                # Create and save transformed versions
                create_transformed_versions(cimg, reference_stars, 
                                          os.path.join(output_directory, "transformed_images"),
                                          filename, original_star_map)
# ====================== Dataset Class ======================

class StarDataset(Dataset):
    """Dataset that loads processed images and their star maps"""
    def __init__(self, processed_dir, star_maps_dir, transform=None):
        self.processed_dir = processed_dir
        self.star_maps_dir = star_maps_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(processed_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load processed image
        img_path = os.path.join(self.processed_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding star map
        base_name = os.path.splitext(self.image_files[idx])[0]
        star_map_path = os.path.join(self.star_maps_dir, f"{base_name}_star_map.json")
        
        with open(star_map_path) as f:
            star_data = json.load(f)
        
        # Get star positions
        star_positions = [star_data['centroid']['position']]
        if 'reference_stars' in star_data:
            star_positions.extend([star['position'] for star in star_data['reference_stars']])
        
        # Create heatmap
        heatmap = self.create_star_heatmap(image.size, star_positions)
        
        if self.transform:
            image = self.transform(image)
        
        return image, heatmap, torch.tensor(star_positions).flatten().float()

    def create_star_heatmap(self, image_size, star_positions, sigma=5):
        """Create heatmap from star positions"""
        heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        for (x, y) in star_positions:
            heatmap = cv2.circle(heatmap, (int(x), int(y)), sigma, 1, -1)
        return torch.tensor(heatmap).unsqueeze(0)

# ====================== Usage Example ======================

if __name__ == "__main__":
    # Process images and create transformed versions
    process_images("/media/student/B076126976123098/my_data/SiT/dataset_sky/star-images", "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed")
    
    # Create dataset from processed images
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = StarDataset(
        processed_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/images",
        star_maps_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/star_maps",
        transform=transform
    )
    
    # Example of accessing data
    sample_image, sample_heatmap, sample_star_positions = dataset[0]
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample heatmap shape: {sample_heatmap.shape}")
    print(f"Sample star positions: {sample_star_positions}")
