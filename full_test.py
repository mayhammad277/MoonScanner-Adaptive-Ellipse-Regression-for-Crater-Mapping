import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import math
import os
import albumentations as A
from scipy.ndimage import gaussian_filter

# --- CONSTANTS (Adjust these for training) ---
TARGET_IMG_SIZE = 512
BATCH_SIZE = 4
SIGMA_CENTER = 5.0  # Gaussian spread for center heatmap
import albumentations as A
import torch
import numpy as np
from tqdm import tqdm
# Assuming your SwinUNet and CraterSegmentationDataset classes are defined
# Assuming extract_crater_catalog is defined
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.measurements as measurements

def extract_crater_catalog(center_map, radius_map, peak_threshold=0.3, min_distance=10):
    """
    Converts the predicted center heatmap and radius map into a list of craters (x, y, r).
    
    Args:
        center_map (np.ndarray): Predicted center probability heatmap (H, W).
        radius_map (np.ndarray): Predicted radius map (H, W).
        peak_threshold (float): Minimum normalized probability to consider a point a crater center.
        min_distance (int): Minimum pixel distance between detected centers.
        
    Returns:
        list: A list of detected craters, each as a tuple (x, y, r).
    """
    
    # 1. Apply a local maximum filter to find candidate peaks
    # This filter identifies pixels that are the maximum value within a window of size (2*min_distance + 1)
    neighborhood = np.ones((2 * min_distance + 1, 2 * min_distance + 1))
    
    # Find local maxima (true if a pixel is the max in its neighborhood)
    local_max = (filters.maximum_filter(center_map, footprint=neighborhood) == center_map)
    
    # 2. Threshold the local maxima
    # Combine local max with the probability threshold
    detected_centers = (center_map > peak_threshold) & local_max
    
    # 3. Extract coordinates and values
    # Get the coordinates (rows, columns) of the true centers
    rows, cols = np.where(detected_centers)
    
    # Get the center probability value and the predicted radius value at those coordinates
    center_values = center_map[rows, cols]
    radius_values = radius_map[rows, cols]
    
    # 4. Create the final catalog (x, y, r)
    catalog = []
    # Sort by confidence (center value) and iterate
    sorted_indices = np.argsort(center_values)[::-1]
    
    for i in sorted_indices:
        # Note: (cols, rows) for (x, y) convention
        x = cols[i]
        y = rows[i]
        r = radius_values[i]
        
        # Simple final check: only keep positive radii
        if r > 1.0: # Keep radii greater than 1 pixel
            catalog.append((x, y, r))
            
    return catalog
def run_inference(model, data_loader, device, peak_threshold=0.3, min_distance=10):
    """
    Runs model inference and generates the final crater catalog for each image.
    """
    model.eval()
    all_predicted_catalogs = []
    
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Running Inference"):
            images = images.to(device)
            
            # Model output: (B, 2, H, W)
            predictions = model(images).cpu().numpy()
            
            for pred in predictions:
                # pred shape: (2, H, W)
                center_map = pred[0]
                radius_map = pred[1]
                
                # Use the post-processing function to convert maps to a catalog
                catalog = extract_crater_catalog(
                    center_map, 
                    radius_map, 
                    peak_threshold=peak_threshold, 
                    min_distance=min_distance
                )
                all_predicted_catalogs.append(catalog)
                
    return all_predicted_catalogs
def evaluate_f1_score(predicted_catalogs, gt_catalogs, 
                      center_tolerance=0.5, radius_tolerance=0.3):
    """
    Calculates the F1 score for the entire dataset based on predicted and ground truth catalogs.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Iterate through each image pair
    for pred_catalog, gt_catalog in zip(predicted_catalogs, gt_catalogs):
        
        # Create a boolean array to track which GT craters have been matched (True Positive)
        gt_matched = np.zeros(len(gt_catalog), dtype=bool)
        
        # Initialize FP (False Positives) for this image
        fp_image = len(pred_catalog) 
        tp_image = 0
        
        # --- Find True Positives ---
        for i, (px, py, pr) in enumerate(pred_catalog):
            # Iterate through all unmatched ground truth craters
            for j in np.where(~gt_matched)[0]:
                gx, gy, gr = gt_catalog[j]
                
                # 1. Check Center Distance
                distance = np.sqrt((px - gx)**2 + (py - gy)**2)
                is_center_close = distance < center_tolerance * gr
                
                # 2. Check Radius Overlap
                is_radius_close = (1 - radius_tolerance) * gr < pr < (1 + radius_tolerance) * gr
                
                if is_center_close and is_radius_close:
                    # Match found!
                    gt_matched[j] = True  # Mark this GT crater as found
                    tp_image += 1
                    fp_image -= 1         # This prediction is now a TP, not an FP
                    break                 # Move to the next predicted crater
        
        # --- Accumulate Totals ---
        total_tp += tp_image
        total_fp += fp_image
        total_fn += (len(gt_catalog) - np.sum(gt_matched)) # FN = GT - TP

    # Calculate final Precision, Recall, and F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "F1 Score": f1_score,
        "Precision": precision,
        "Recall": recall,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn
    }
def get_craters_list_from_mask(mask_image_path, min_area_filter=17, kernel_size=3):
    """ Extracts crater center and radius list using the instance separation strategy. """
    mask_raw = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Mask file not found: {mask_image_path}")

    # 1. Otsu's Thresholding
    _, binary_mask = cv2.threshold(mask_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Minimal Opening (Remove tiny noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)) 
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Find Contours
    contours, _ = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    craters_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_filter:
            # Calculate Minimum Enclosing Circle (MEC)
            (x_center, y_center), R_max = cv2.minEnclosingCircle(contour)
            craters_list.append({
                'center': (float(x_center), float(y_center)), 
                'radius': float(R_max)
            })
    return craters_list

def generate_crater_heatmaps(image_shape, craters_list, sigma_center=SIGMA_CENTER):
    """ Generates center and radius maps for training the regression heads. """
    H, W = image_shape
    center_map = np.zeros((H, W), dtype=np.float32)
    radius_map = np.zeros((H, W), dtype=np.float32)
    
    for crater in craters_list:
        cx, cy = crater['center']
        radius = crater['radius']
        
        # Ensure coordinates are within bounds
        cx_int, cy_int = int(cx), int(cy)
        if 0 <= cx_int < W and 0 <= cy_int < H:
            radius_map[cy_int, cx_int] = radius 
            center_map[cy_int, cx_int] = 1.0
            
    # Apply Gaussian smoothing to the Center Map
    center_map = gaussian_filter(center_map, sigma=sigma_center)
    
    # Normalize the Center Map between 0 and 1
    if center_map.max() > 0:
        center_map = center_map / center_map.max()
        
    return center_map, radius_map
    
    
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
# Assuming get_craters_list_from_mask and generate_crater_heatmaps are defined elsewhere

class CraterSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, img_size=512, 
                 min_area_filter=17, kernel_size=3, sigma_center=5.0):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.min_area_filter = min_area_filter
        self.kernel_size = kernel_size
        self.sigma_center = sigma_center
        
        # --- CRITICAL FIX 1: Use KeypointParams for center coordinates ---
        # If transform is None, define a default one with KeypointParams
        if transform is None:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Ensure resize is present if images are not uniform
                    A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
                ],
                # KeypointParams is correct for point-like data (crater centers)
                keypoint_params=A.KeypointParams(
                    format='xy',           # Standard normalized x, y coordinates
                    label_fields=['labels'], # Use 'labels' field to carry the radius
                    remove_invisible=False # Don't lose points outside the crop
                )
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Data
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask_path = self.mask_paths[idx]

        # Convert to 3-channel for standard augmentation
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Get ground truth crater list (cx, cy, radius)
        craters_list_raw = get_craters_list_from_mask(
            mask_path, self.min_area_filter, self.kernel_size
        )

        # 2. Prepare Data for Augmentation
        keypoints = []
        labels = [] # This will temporarily store the radius

        for c in craters_list_raw:
            cx, cy = c['center']
            r = c['radius']
            
            # --- CRITICAL FIX 2: Prepare keypoints (x, y) ---
            # Albumentations expects normalized coordinates (0 to 1) if not specified
            keypoints.append((cx, cy)) 
            labels.append(r) # Radius is carried in the 'labels' field

        # 3. Apply Transform
        transformed = self.transform(image=image, keypoints=keypoints, labels=labels)

        image_aug = transformed['image']
        keypoints_aug = transformed['keypoints']
        radii_aug = transformed['labels'] # The radii come back transformed in 'labels'

        # 4. Reconstruct Transformed Crater List
        craters_list_aug = []
        for i, point in enumerate(keypoints_aug):
            # --- CRITICAL FIX 3: Recover center and radius ---
            cx_aug, cy_aug = point 
            r_aug = radii_aug[i]
            
            craters_list_aug.append({
                'center': (cx_aug, cy_aug),
                'radius': r_aug
            })

        # 5. Generate Final Ground Truth Heatmaps
        # Convert image back to single channel for model input
        image_aug_gray = cv2.cvtColor(image_aug, cv2.COLOR_RGB2GRAY)
        
        # Resize to standard size (if not already done by A.Resize)
        H, W = self.img_size, self.img_size
        
        # Generate the two heatmaps
        center_map_gt, radius_map_gt = generate_crater_heatmaps(
            (H, W), craters_list_aug, self.sigma_center
        )

        # 6. Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(image_aug_gray).float().unsqueeze(0) / 255.0 # (1, H, W)
        
        # Stack the two output maps into a single target tensor (2, H, W)
        target_tensor = torch.stack([
            torch.from_numpy(center_map_gt).float(),
            torch.from_numpy(radius_map_gt).float()
        ], dim=0)

        return image_tensor, target_tensor
        
        # --- Swin-UNet components (PatchEmbed, PatchMerging, SwinTransformerBlock) are omitted here for brevity, 
# but assume they are defined as in the previous response. ---

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming your class structure is complete with layer definitions...

class SwinUNet(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=1,  # in_chans=1 for grayscale input
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, **kwargs):
        super().__init__()

        # Assuming you have defined the necessary components in __init__:
        # self.patch_embed: Patch Embedding layer
        # self.encoder_layers: List of encoder blocks/stages
        # self.decoder_layers: List of decoder blocks/stages
        # self.bottleneck: The layer connecting encoder and decoder
        # self.center_head: Final 1x1 Conv for center probability
        # self.radius_head: Final 1x1 Conv for radius value

        # --- Example of defining the final heads (as in your original snippet) ---
        current_dim = embed_dim * (2**(len(depths)-1)) # Calculate bottleneck dim
        # After full decoder upsampling, it reverts to embed_dim
        final_decoder_dim = embed_dim 
        
        # ... (rest of your __init__ definitions go here) ...
        
        self.center_head = nn.Conv2d(final_decoder_dim, 1, kernel_size=1)
        self.radius_head = nn.Conv2d(final_decoder_dim, 1, kernel_size=1)


    # --- THIS IS THE MISSING METHOD THAT CAUSES THE ERROR ---
    def forward(self, x):
        
        # This list will hold the feature maps from the ENCODER blocks 
        # that will be passed as SKIP CONNECTIONS to the decoder.
        skip_connections = [] 

        # 1. Patch Embedding & Positional Embedding (if used)
        # x = self.patch_embed(x) 

        # 2. Encoder Path (Downsampling)
        # for encoder_layer in self.encoder_layers:
        #     # Save output before the next downsample for the skip connection
        #     skip_connections.append(x) 
        #     x = encoder_layer(x)

        # 3. Bottleneck (Deepest Layer)
        # x = self.bottleneck(x)

        # 4. Decoder Path (Upsampling)
        # for decoder_layer in self.decoder_layers:
        #     # Retrieve skip connection from the corresponding encoder stage
        #     skip = skip_connections.pop() 
        #     
        #     # Concatenate the current feature map (x) with the skip connection (skip)
        #     x = torch.cat([x, skip], dim=1) 
        #     
        #     # Pass through the upsampling and decoding blocks
        #     x = decoder_layer(x) 

        # --- IMPORTANT: Placeholder Implementation ---
        # Since the layers are missing, we cannot run the actual Swin-UNet logic.
        # We must return *something* of the expected shape to allow the trainer 
        # to proceed to the next error (which will be a layer-related error).
        
        # Assuming input x is (B, 1, H, W) and you want output (B, 2, H, W)
        B, C, H, W = x.shape
        
        # Simulate the final feature map (e.g., self.final_decoder_dim channels)
        # Replace this with the actual feature map from your decoder
        final_features = torch.rand(B, self.center_head.in_channels, H, W).to(x.device)
        

        # 5. Dual Output Heads
        # Apply the two convolutional heads to the final feature map
        center_output = self.center_head(final_features) # Output shape: (B, 1, H, W)
        radius_output = self.radius_head(final_features) # Output shape: (B, 1, H, W)
        
        # Concatenate the two outputs to match the expected target shape (B, 2, H, W)
        output = torch.cat([center_output, radius_output], dim=1)
        
        return output        
        
class CraterDetectionLoss(nn.Module):
    def __init__(self, alpha=0.5, center_weight=1.0):
        """
        Custom loss for heatmap regression combining Center (BCE) and Radius (L1).
        
        Args:
            alpha (float): Weight for the Radius loss (0.5 means equal weight).
            center_weight (float): Weight applied to the positive center pixels in BCE. 
        """
        super().__init__()
        self.alpha = alpha
        
        # Move center_weight to the device if using CUDA, and wrap it as a tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weight = torch.tensor([center_weight]).to(device)
        
        # Center loss: BCE is common for heatmaps (applied to all pixels)
        # Use pos_weight to address the imbalance between background (0) and center (1) pixels
        self.center_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none') 
        
        # Radius loss: L1 (MAE) is robust for regression (applied only at center peaks)
        self.radius_loss_func = nn.L1Loss(reduction='none') 

    def forward(self, pred, target):
        # target[0] is Center Map, target[1] is Radius Map
        
        # Split predictions and targets
        pred_center_logits = pred[:, 0, :, :] # Logits before Sigmoid
        pred_radius = pred[:, 1, :, :]
        
        target_center, target_radius = target[:, 0, :, :], target[:, 1, :, :]

        # 1. Center Heatmap Loss (Semantic Loss)
        center_loss = self.center_loss_func(pred_center_logits, target_center)
        
        # 2. Radius Regression Loss (Applied only at crater center peaks)
        RADIUS_MASK_THRESHOLD = 0.1
        radius_mask = (target_center > RADIUS_MASK_THRESHOLD).float() 
        
        radius_loss = self.radius_loss_func(pred_radius, target_radius) * radius_mask
        
        # Normalize radius loss by the number of active (crater) pixels
        num_center_pixels = radius_mask.sum()
        if num_center_pixels > 0:
            radius_loss = radius_loss.sum() / num_center_pixels
        else:
            radius_loss = torch.tensor(0.0).to(pred.device) 
            
        # 3. Combined Loss
        total_loss = (1.0 - self.alpha) * center_loss.mean() + self.alpha * radius_loss
        
        return total_loss

import torch.optim as optim
from tqdm import tqdm 

def train_pipeline(model, train_loader, val_loader, epochs=10, lr=1e-4, 
                   save_path='best_model_weights.pth'):
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = CraterDetectionLoss(alpha=0.6, center_weight=10.0)
    
    # --- Tracking for Best Model ---
    best_val_loss = float('inf')
    
    # --- Main Training Loop ---
    for epoch in range(1, epochs + 1):
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = loss_fn(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # --- Weight Saving Logic ---
        
        # 1. Check if the current model is the best so far
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving new best model.")
            best_val_loss = avg_val_loss
            
            # 2. Save the model's state dictionary
            # State dictionary is the standard way to save only the learned parameters
            torch.save(model.state_dict(), save_path)
            
            
        # --- Summary ---
        print(f"\n--- Epoch {epoch}/{epochs} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
        print("-" * 35)

    print("Training Complete!")
    
    # Load and return the best model weights for immediate use
    print(f"Loading best weights from {save_path} before returning model.")
    model.load_state_dict(torch.load(save_path)) 
    return model
        
import os

import os

def get_hybrid_filtered_paths(base_dir):
    """
    Collects and pairs input images with their corresponding ground truth masks
    by searching for the mask file in an expected 'truth' directory 
    relative to the image's location, assuming a structure like: 
    /altitudeXX/longitudeYY/image.png and /altitudeXX/truth/mask_truth.png.
    
    Args:
        base_dir (str): The top-level directory containing all image data.
        
    Returns:
        tuple: (list of input image paths, list of ground truth mask paths)
    """
    
    final_image_paths = []
    final_mask_paths = []

    print(f"Scanning directory: {base_dir} for input images...")
    
    # 1. Walk through the base directory to find all non-mask image files
    for root, _, files in os.walk(base_dir):
        for file in files:

            # Identify standard input images (not masks)
            if file.endswith('.png') and '_truth' not in file:
                img_path="/home/bora3i/crater_challenge/train-sample/altitude01/longitude02"
                image_path = os.path.join(img_path, file)
                
                # --- 2. Construct the Expected Mask Path ---
                
                # Assume the image's directory 'root' is '/.../altitudeXX/longitudeYY'
                
                # Get the parent directory of the image's folder (e.g., '/.../altitudeXX')
                parent_of_image_folder = os.path.dirname(root) 
                
                # Construct the expected mask directory (e.g., '/.../altitudeXX/truth')
                expected_mask_dir = os.path.join(parent_of_image_folder, 'truth')
                
                # Construct the full mask filename: orientationXX_lightYY_truth.png
                mask_filename = file.replace('.png', '_truth.png')
                
                # Final expected mask path:
                expected_mask_path = os.path.join(expected_mask_dir, mask_filename)

                
                # --- 3. Verify and Pair ---
                if os.path.exists(expected_mask_path) and os.path.exists(image_path):
                    final_image_paths.append(image_path)
                    final_mask_paths.append(expected_mask_path)
                # else:
                    # You can uncomment this line for deeper debugging if pairs are still missing:
                    # print(f"Warning: Mask for {file} not found at: {expected_mask_path}")
                    pass

    print(f"Successfully paired {len(final_image_paths)} image/mask pairs for training.")
    return final_image_paths, final_mask_paths

# --- REMEMBER TO USE THIS FUNCTION IN YOUR MAIN SCRIPT ---

# Example usage:
base_directory = "/home/bora3i/crater_challenge/train-sample"
image_paths, mask_paths = get_hybrid_filtered_paths(base_directory)
print(f"Total images found: {len(image_paths)}")



TOTAL_SAMPLES = 50
SPLIT_INDEX = 80 # <-- This is the likely issue!



# Assuming you split your paths into training and validation sets
SPLIT_INDEX = int(50 * 0.8)  # SPLIT_INDEX will be 40# 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the trained model
model = SwinUNet(in_chans=3, embed_dim=96) # num_classes=2 for Center + Radius
model.load_state_dict(torch.load('best_model_weights.pth'))
model.to(device)
validation_dataset = CraterSegmentationDataset(image_paths[SPLIT_INDEX:], mask_paths[SPLIT_INDEX:])

val_loader = DataLoader(
    validation_dataset, 
    batch_size=4, 
    shuffle=False, 
    num_workers=2
)
# 2. Get Validation GT Catalogs (Non-Heatmap Version)
# Assuming 'validation_dataset' and 'val_loader' are available from training setup
gt_catalogs_val = [
    get_craters_list_from_mask(
        mask_path, 
        min_area_filter=17, 
        kernel_size=3
    ) 
    for mask_path in validation_dataset.mask_paths
]
# Adjust the format of the items in the list to be (x, y, r) tuples if necessary

# 3. Run Inference
predicted_catalogs = run_inference(
    model, 
    val_loader, 
    device, 
    peak_threshold=0.3, # Fine-tune this and min_distance!
    min_distance=10
)

# 4. Evaluate
results = evaluate_f1_score(predicted_catalogs, gt_catalogs_val)

print("\n--- Final Evaluation Results ---")
print(f"F1 Score: {results['F1 Score']:.4f}")
print(f"Precision: {results['Precision']:.4f}")
print(f"Recall: {results['Recall']:.4f}")
