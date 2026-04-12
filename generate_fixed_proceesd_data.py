import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def process_crater_data(csv_path, output_dir, shape=(160, 160)):
    """
    Regenerates clean 10-channel heatmaps and regression targets.
    Ensures 2D spatial integrity to avoid banding noise.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Scale factor from raw image (2592) to heatmap (160)
    scale = 160 / 2592  

    for img_name in tqdm(df['inputImage'].unique(), desc="Processing Heatmaps"):
        # Initialize 10-channel target: [5 Classes, Maj, Min, dx, dy, rot]
        gt = np.zeros((10, 160, 160), dtype=np.float32)
        annos = df[df['inputImage'] == img_name]

        for _, row in annos.iterrows():
            # Mahanti Classification (0-4)
            cls = int(row['crater_classification'])
            
            # Center coordinates scaled to 160x160
            ctx = row['ellipseCenterX(px)'] * scale
            cty = row['ellipseCenterY(px)'] * scale
            ix, iy = int(ctx), int(cty)

            if 0 <= ix < 160 and 0 <= iy < 160:
                # 1. GENERATE 2D GAUSSIAN (Channels 0-4)
                major_scaled = row['ellipseSemimajor(px)'] * scale
                # Sigma controls the "glow" size; 1.5 is a safe minimum
                sigma = max(1.5, major_scaled / 3.0)
                
                y_grid, x_grid = np.ogrid[:160, :160]
                # Proper 2D spatial distribution formula
                gaussian = np.exp(-((x_grid - ctx)**2 + (y_grid - cty)**2) / (2 * sigma**2))
                
                # Update Heatmap (using max to handle crater overlaps)
                gt[cls] = np.maximum(gt[cls], gaussian)
                
                # 2. UPDATE REGRESSION (Channels 5-9) at the peak pixel center only
                gt[5, iy, ix] = major_scaled
                gt[6, iy, ix] = row['ellipseSemiminor(px)'] * scale
                gt[7, iy, ix] = ctx - ix  # Sub-pixel offset X
                gt[8, iy, ix] = cty - iy  # Sub-pixel offset Y
                gt[9, iy, ix] = np.deg2rad(row['ellipseRotation(deg)'])

        # Save as explicit (C, H, W) float32 array
        base_filename = img_name.split('.')[0]
        np.save(os.path.join(output_dir, f"{base_filename}.npy"), gt)

if __name__ == "__main__":
    # Adjust paths to your environment
    CSV_PATH = "train-gt.csv"
    OUT_DIR = "./processed_data_aug_v3"
    process_crater_data(CSV_PATH, OUT_DIR)
