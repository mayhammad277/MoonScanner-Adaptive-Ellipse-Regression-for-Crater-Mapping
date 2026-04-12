import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import math
import cv2
import pandas as pd
from tqdm import tqdm

# --- 1. CONFIGURATION ---

# Paths (Update these to your actual data locations)
DATA_ROOT = "/home/s478608/may/crater_challenge/train-sample/altitude01/"
GT_CSV_PATH = "/home/s478608/may/crater_challenge/train-gt.csv" 

# Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
MODEL_STRIDE = 4 # Stride of the Swin feature map output (e.g., 1/4 resolution)

# Loss weights (Hyperparameters often tuned via grid search)
LOSS_WEIGHTS = {
    'hm': 1.0,  # Heatmap
    'reg': 0.1, # Regression (R_max, Offset, Ellipse)
    'cls': 1.0  # Classification
}

# --- 2. IMPORT CUSTOM MODULES (Simplified Imports) ---

# Assume ModifiedFocalLoss is implemented as in the previous step

# Assume CraterPredictionHead and CraterDetectionLoss are implemented as in previous steps

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CraterPredictionHead(nn.Module):
    def __init__(self, in_channels,num_classes, head_channels=64):
        super(CraterPredictionHead, self).__init__()
        self.num_classes=num_classes
        # 1. Heatmap Head (Channel 0)
        self.hm_head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, kernel_size=1)
        )
        
        # 2. Radius Head (Channel 1)
        self.radius_head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 1, kernel_size=1)
        )
        
        # 3. Offset Head (Channels 2 & 3)
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Predicts both X and Y offset
            nn.Conv2d(head_channels, 2, kernel_size=1) )

    def forward(self, x):
         hm = self.hm_head(x)
         radius = self.radius_head(x)
         offset = self.offset_head(x)
        
        # We return them in a dictionary for easy access in the loss function
         return {
            'heatmap': hm,
            'r_max': radius,
            'offset': offset
         }

class CraterDetectionLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.focal_loss = ModifiedFocalLoss()
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.w = weights

    def forward(self, preds, targets):
        # 1. Heatmap Loss
        hm_loss = self.focal_loss(preds['heatmap'], targets['heatmap'])
        
        # 2. Regression Mask (Only calculate loss where a crater center exists)
        mask = targets['heatmap'] == 1.0 
        num_pos = mask.float().sum()
        
        if num_pos == 0:
            return hm_loss * self.w['hm']

        # 3. Radius & Ellipse Loss
        # We extract only the pixels that are crater centers
        reg_loss = 0
        for key in ['r_max', 'offset', 'ellipse']:
            p = preds[key][mask.repeat(1, preds[key].shape[1], 1, 1)]
            t = targets[key][mask.repeat(1, targets[key].shape[1], 1, 1)]
            reg_loss += self.l1_loss(p, t)
            
        return (self.w['hm'] * hm_loss) + (self.w['reg'] * (reg_loss / num_pos))
def generate_radius_ground_truth(input_path, mask_path):
    # Load Images
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None: return None, None
    H, W = mask_image.shape
    
    # 1. Process Mask to find Craters
    _, binary_mask = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
    # 2. Initialize 8-Channel Ground Truth Map
    # [HM, R_max, OffX, OffY, SemiMaj, SemiMin, Rot, Class]
    gt_map = np.zeros((H, W, 8), dtype=np.float32)
    gt_map[:, :, 7] = -1  # Default classification to -1 (missing)

    for contour in contours:
        if cv2.contourArea(contour) < 17: continue
            
        # Geometric Properties
        (cx_f, cy_f), r_max = cv2.minEnclosingCircle(contour)
        cx, cy = int(cx_f), int(cy_f)
        
        # Ellipse Properties
        if len(contour) >= 5:
            (ecx, ecy), (d_maj, d_min), angle = cv2.fitEllipse(contour)
            semi_maj, semi_min = d_maj / 2, d_min / 2
        else:
            semi_maj, semi_min, angle = r_max, r_max, 0

        # Fill Heatmap (Channel 0)
        sigma = max(r_max / 3, 1.0)
        radius = math.ceil(sigma * 3)
        # ... (Gaussian splicing logic here) ...
        # (For brevity in this block, assume the Gaussian splicing from previous steps is applied to gt_map[..., 0])
        
        # Fill Regression Targets at Center (Channels 1-7)
        if 0 <= cy < H and 0 <= cx < W:
            gt_map[cy, cx, 1] = r_max
            gt_map[cy, cx, 2] = cx_f - cx # Offset X
            gt_map[cy, cx, 3] = cy_f - cy # Offset Y
            gt_map[cy, cx, 4] = semi_maj
            gt_map[cy, cx, 5] = semi_min
            gt_map[cy, cx, 6] = angle
            # gt_map[cy, cx, 7] = class_label # If available from CSV
            
    return gt_map, contours

import torch.nn.functional as F

class ModifiedFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits, gt_heatmap):
        """
        pred_logits: [Batch, 1, 648, 648] (from model)
        gt_heatmap:  [Batch, 1, 2592, 2592] (from dataset)
        """
        
        # --- DYNAMIC SIZE FIX ---
        # If the GT is larger than the Prediction, downsample the GT to match.
        if gt_heatmap.shape[-2:] != pred_logits.shape[-2:]:
            # print(f"Fixing mismatch: GT {gt_heatmap.shape[-2:]} -> Pred {pred_logits.shape[-2:]}")
            gt_heatmap = F.interpolate(gt_heatmap, size=pred_logits.shape[-2:], mode='nearest')
        # -------------------------

        preds = torch.sigmoid(pred_logits)
        eps = 1e-12
        preds = torch.clamp(preds, min=eps, max=1.0 - eps)

        pos_inds = gt_heatmap.eq(1).float()
        neg_inds = gt_heatmap.lt(1).float()

        neg_weights = torch.pow(1 - gt_heatmap, self.beta)

        pos_loss = torch.log(preds) * torch.pow(1 - preds, self.alpha) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss

class ModifiedFocalLoss2(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits, gt_heatmap):
        preds = torch.sigmoid(pred_logits)
        eps = 1e-12
        preds = torch.clamp(preds, min=eps, max=1.0 - eps)

        pos_inds = gt_heatmap.eq(1).float()
        neg_inds = gt_heatmap.lt(1).float()

        neg_weights = torch.pow(1 - gt_heatmap, self.beta)

        pos_loss = torch.log(preds) * torch.pow(1 - preds, self.alpha) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class CraterDataset(Dataset):
    def __init__(self, data_root, gt_csv_path=None, stride=4):
        self.data_root = data_root
        self.stride = stride
        self.image_files = []
        
        # Build the list of images by walking through altitude/longitude folders
        for root, dirs, files in os.walk(data_root):
            if 'truth' in root: continue # Skip truth folders during search
            for file in files:
                if file.endswith('.png'):
                    # Create the relative path ID (e.g., altitude01/longitude01/image_name)
                    rel_path = os.path.relpath(os.path.join(root, file), data_root)
                    # Remove the extension for the ID
                    image_id = os.path.splitext(rel_path)[0]
                    self.image_files.append(image_id)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_id = self.image_files[idx]
        
        # 1. Define paths
        input_image_path =os.path.join( "/home/s478608/may/crater_challenge/train-sample/altitude01/", f"{image_id}.png")
        mask_image_path = os.path.join("/home/s478608/may/crater_challenge/train-sample/altitude01/", f"{image_id}.png")

        # Mask is usually in a 'truth' folder at the same level as the images
        img_dir = os.path.dirname(image_id)
        img_base = os.path.basename(image_id)
        #mask_image_path = os.path.join(self.data_root, img_dir, 'truth', f"{img_base}_truth.png")
        
        # 2. Load and normalize the input image
        img = cv2.imread(input_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {input_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor and change shape to (Channels, Height, Width)
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 3. Generate Ground Truth (using the function we defined earlier)
        # full_gt_map is [2592, 2592, 8]
        full_gt_map, _ = generate_radius_ground_truth(input_image_path, mask_image_path)
        
        # 4. Resize Ground Truth to match Model Stride (2592 -> 648)
        H, W, _ = full_gt_map.shape
        target_h, target_w = H // self.stride, W // self.stride
        
        # Heatmap (Channel 0): Use INTER_AREA for smooth downsampling
        gt_hm_down = cv2.resize(full_gt_map[:, :, 0], (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Regression Channels (1-7): Use INTER_NEAREST to preserve exact values (Radius, Angle, etc.)
        gt_reg_down = cv2.resize(full_gt_map[:, :, 1:], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 5. Pack into Tensors
        gt_hm_tensor = torch.from_numpy(gt_hm_down).unsqueeze(0).float() # [1, 648, 648]
        gt_reg_tensor = torch.from_numpy(gt_reg_down).permute(2, 0, 1).float() # [7, 648, 648]

        gt_dict = {
            'heatmap': gt_hm_tensor,
            'r_max': gt_reg_tensor[0:1, :, :],    # Radius
            'offset': gt_reg_tensor[1:3, :, :],   # Ox, Oy
            'ellipse': gt_reg_tensor[3:6, :, :],  # Semi-maj, Semi-min, Rot
            'class_labels': gt_reg_tensor[6, :, :] # Class index
        }

        return img_tensor, gt_dict, image_id
class CraterDataset2(Dataset):
    def __init__(self, data_root, gt_csv_path, transform=None, stride=4):
        self.data_root = data_root
        # Placeholder: This should be the function you wrote to generate the 4+ channel ground truth
        self.gt_generator = generate_radius_ground_truth 
        self.transform = transform
        self.stride = stride
        
        # NOTE: A robust dataset would pre-generate or cache all GT maps 
        # to avoid recalculating them during every epoch.
        
        # Load all image paths (Simplified structure retrieval)
        self.image_files = []
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith('.png') and 'mask' not in root:
                    # Construct image ID relative to DATA_ROOT
                    rel_path = os.path.relpath(root, data_root)
                    image_id = os.path.join(rel_path, os.path.splitext(file)[0])
                    self.image_files.append(image_id)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_id = self.image_files[idx]
        
        # Construct full paths
        input_path =os.path.join( "/home/s478608/may/crater_challenge/train-sample/altitude01/", f"{image_id}.png")
        mask_path = os.path.join("/home/s478608/may/crater_challenge/train-sample/altitude01/", f"{image_id}.png")
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {input_image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to Tensor (CHW format)
        # From [H, W, 3] to [3, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0        

        full_gt_map, _ = generate_radius_ground_truth(input_path, mask_path)
        
        # 2. Calculate the target size (Input / Stride)
        # If input is 2592, target is 648
        H, W, C = full_gt_map.shape
        target_h, target_w = H // self.stride, W // self.stride
        
        # 3. Downsample the Heatmap (Channel 0)
        # Use INTER_NEAREST or INTER_AREA to keep the peaks sharp
        gt_hm_down = cv2.resize(full_gt_map[:, :, 0], (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # 4. Downsample the Regression Maps (Radius, Offsets, etc.)
        gt_reg_down = cv2.resize(full_gt_map[:, :, 1:], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 5. Convert to PyTorch Tensors
        # Ensure they are (Channels, Height, Width)
        gt_hm_tensor = torch.from_numpy(gt_hm_down).unsqueeze(0) # [1, 648, 648]
        gt_reg_tensor = torch.from_numpy(gt_reg_down).permute(2, 0, 1) # [C-1, 648, 648]

        return img_tensor, {
            'heatmap': gt_hm_tensor,
            'r_max': gt_reg_tensor[0:1, :, :],
            'offset': gt_reg_tensor[1:3, :, :],
            'ellipse': gt_reg_tensor[3:6, :, :]
        }, image_id

# --- 4. FULL MODEL INTEGRATION ---

class FullCraterModel(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        
        # 1. Swin Transformer Backbone (PSEUDO-CODE)
        # In a real setup, this would be a loaded Swin Transformer:
        # self.backbone = models.swin_transformer_base(pretrained=True)
        # And you would extract features from a specific layer (e.g., stage 3 or 4)
        
        # Placeholder for the Swin Backbone output features (C=1024 typical)
        backbone_out_channels = 1024 
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, backbone_out_channels, kernel_size=1) # Dummy downsampling
        )

        # 2. Custom Prediction Head
        self.prediction_head = CraterPredictionHead(
            in_channels=backbone_out_channels, 
            num_classes=num_classes
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.prediction_head(features)
        return predictions

# --- 5. TRAINING LOOP ---

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Wrap with tqdm for progress bar
    for batch_idx, (images, gt_targets, _) in enumerate(tqdm(train_loader, desc="Training")):
        
        # 1. Move data to device
        images = images.to(device)
        gt_targets = {k: v.to(device) for k, v in gt_targets.items()}
        
        # 2. Zero gradients
        optimizer.zero_grad()
        
        # 3. Forward pass
        predictions = model(images)
        
        # 4. Calculate Loss
        total_loss = criterion(predictions, gt_targets)
        
        # 5. Backward pass and Optimization
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch Loss: {avg_loss:.4f}")

# --- EXECUTION ---

if __name__ == '__main__':
    
    # 0. Setup Device
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # 1. Initialize Dataset and DataLoader
    # NOTE: You must update DATA_ROOT and GT_CSV_PATH
    try:
        train_dataset = CraterDataset(DATA_ROOT, GT_CSV_PATH, stride=4)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        print(f"Dataset initialized with {len(train_dataset)} samples.")
    except Exception as e:
        print(f"ERROR: Dataset initialization failed. Check your data paths and GT generator logic. Error: {e}")
        exit()

    # 2. Initialize Model, Loss, and Optimizer
    model = FullCraterModel(in_channels=3, num_classes=5).to(device)
    criterion = CraterDetectionLoss(LOSS_WEIGHTS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Start Training Loop
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{NUM_EPOCHS} ---")
        train_model(model, train_loader, criterion, optimizer, device)
        
    print("\nTraining complete!")
    
    # 4. Save Model Checkpoint
    torch.save(model.state_dict(), 'crater_detector_swin.pth')
