import pandas as pd

def ensemble_craters(csv640, csv320, output_csv):
    df640 = pd.read_csv(csv640)
    df320 = pd.read_csv(csv320)
    
    # Remove dummy rows (-1) before merging
    df640 = df640[df640['ellipseCenterX(px)'] != -1]
    df320 = df320[df320['ellipseCenterX(px)'] != -1]
    
    # Combine both
    combined = pd.concat([df640, df320], ignore_index=True)
    
    # Optional: Remove duplicates based on image and center (rounded)
    # This prevents the same crater being counted twice if both models hit it
    combined['rounded_cx'] = combined['ellipseCenterX(px)'].round(-1) # Round to nearest 10px
    combined['rounded_cy'] = combined['ellipseCenterY(px)'].round(-1)
    
    final = combined.drop_duplicates(subset=['inputImage', 'rounded_cx', 'rounded_cy'])
    
    # Drop the helper columns
    final = final.drop(columns=['rounded_cx', 'rounded_cy'])
    
    final.to_csv(output_csv, index=False)
    print(f"Ensemble complete. {len(final)} craters saved to {output_csv}")

if __name__ == "__main__":
    ensemble_craters("/home/bora3i/crater_challenge/submission8_640/solution/solution.csv", "/home/bora3i/crater_challenge/submission8_320/solution/solution.csv", "final_ensemble_submission.csv")
