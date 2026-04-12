import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import random
import albumentations as A

# Note: Install albumentations package (pip install albumentations)

class CraterSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=512, augment=True):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
        
        # --- Define Augmentation Pipeline ---
        if self.augment:
            self.transform = A.Compose([
                # Spatial Augmentations (applies to image and mask simultaneously)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.7, interpolation=cv2.INTER_LINEAR),
                A.RandomCrop(height=img_size, width=img_size, p=1.0), # Crop to target size
                
                # Appearance Augmentations (applies ONLY to image)
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ], 
            # Define targets: 'image' and 'mask' must be augmented together
            additional_targets={'mask': 'image'}) 
        else:
            # If no augmentation, just resize/crop to the target size
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
            ])
        
        # Ensure lists are of the same length
        assert len(self.image_paths) == len(self.mask_paths), "Image and mask lists must have the same length."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load Images (Input and Ground Truth)
        # Load raw image (input)
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Swin models often prefer RGB
        
        # Load mask image (ground truth)
        mask_raw = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 2. Preprocess Ground Truth Mask (Otsu's Thresholding)
        # Separate light grey craters (foreground) from darker grey background
        if mask_raw is None:
            raise FileNotFoundError(f"Mask file not found or corrupted: {self.mask_paths[idx]}")
            
        _, mask_binary = cv2.threshold(mask_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean the mask slightly (optional, but good practice for ground truth)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Convert mask to 0s and 1s and set its type (H, W, 1)
        mask = (mask_binary / 255.0).astype(np.float32)
        
        # 3. Apply Transformations (Augmentation)
        # Augmentation requires image and mask to be in (H, W, C) format
        transformed = self.transform(image=image, mask=mask)
        image_aug = transformed['image']
        mask_aug = transformed['mask']

        # 4. Final Conversion to PyTorch Tensors
        
        # Normalize image (0-255 RGB to 0-1.0 float32)
        image_tensor = torch.from_numpy(image_aug / 255.0).float()
        
        # Convert to Channel-First format (C, H, W) for PyTorch
        # Input image: (3, 512, 512)
        image_tensor = image_tensor.permute(2, 0, 1) 
        
        # Mask: (1, 512, 512). Mask must remain 0/1.
        mask_tensor = torch.from_numpy(mask_aug).float().unsqueeze(0) 

        return image_tensor, mask_tensor
        
        
        
        
        
# --- Example Data Path Setup (Adjust paths as necessary) ---

# Assume your files are in these directories
base_dir = '/path/to/your/crater/data/'
train_image_dir = os.path.join(base_dir, 'train/images')
train_mask_dir = os.path.join(base_dir, '/home/bora3i/crater_challenge/train-sample/altitude01/longitude02/truth/')

# Generate file lists (e.g., orientation05_light05.jpg)
# This is a placeholder; you'd use os.listdir() and sorting in a real scenario
train_image_files = os.listdir(train_image_dir) 
train_mask_files = os.listdir(train_mask_dir) 

train_image_paths = [os.path.join(train_image_dir, f) for f in train_image_files]
train_mask_paths = [os.path.join(train_mask_dir, f) for f in train_mask_files]        
