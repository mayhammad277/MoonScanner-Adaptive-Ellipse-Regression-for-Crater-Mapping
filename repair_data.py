import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def repair_data(csv_path, output_dir, shape=(160, 160)):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    scale = 160 / 2592  # Scale factor from raw image to heatmap

    for img_name in tqdm(df['inputImage'].unique()):
        # Initialize 10-channel target: [Classes(0-4), Maj, Min, dx, dy, rot]
        gt = np.zeros((10, 160, 160), dtype=np.float32)
        annos = df[df['inputImage'] == img_name]

        for _, row in annos.iterrows():
            cls = int(row['crater_classification'])
            # Center coordinates scaled to 160x160
            ctx = row['ellipseCenterX(px)'] * scale
            cty = row['ellipseCenterY(px)'] * scale
            ix, iy = int(ctx), int(cty)

            if 0 <= ix < 160 and 0 <= iy < 160:
                # Generate 2D Gaussian for the specific class channel
                major_scaled = row['ellipseSemimajor(px)'] * scale
                sigma = max(1.5, major_scaled / 3.0)
                
                y_grid, x_grid = np.ogrid[:160, :160]
                gaussian = np.exp(-((x_grid - ctx)**2 + (y_grid - cty)**2) / (2 * sigma**2))
                
                # Update Heatmap (Channels 0-4)
                gt[cls] = np.maximum(gt[cls], gaussian)
                
                # Update Regression (Channels 5-9) at the peak pixel
                gt[5, iy, ix] = major_scaled
                gt[6, iy, ix] = row['ellipseSemiminor(px)'] * scale
                gt[7, iy, ix] = ctx - ix  # Offset x
                gt[8, iy, ix] = cty - iy  # Offset y
                gt[9, iy, ix] = np.deg2rad(row['ellipseRotation(deg)'])

        # Save with fixed naming convention
        np.save(os.path.join(output_dir, f"{img_name.split('.')[0]}.npy"), gt)

if __name__ == "__main__":
    repair_data("train-gt.csv", "./processed_data_aug_v2")
