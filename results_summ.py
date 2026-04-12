import pandas as pd
import os

# --- Configuration ---
CSV_INPUT = "test_predictions_v3.csv"

def generate_report():
    if not os.path.exists(CSV_INPUT):
        print(f"❌ Error: {CSV_INPUT} not found. Run inference first.")
        return

    # Load data
    df = pd.read_csv(CSV_INPUT)
    
    if df.empty:
        print("⚠️ The CSV is empty. No craters were detected.")
        return

    # 1. Extract Altitude from image_id (e.g., 'altitude01/...')
    df['altitude'] = df['image_id'].apply(lambda x: x.split('/')[0])

    # 2. Basic Stats
    total_craters = len(df)
    avg_conf = df['confidence'].mean()
    avg_rad = df['radius'].mean()

    # 3. Group by Altitude
    summary = df.groupby('altitude').agg(
        crater_count=('image_id', 'count'),
        avg_confidence=('confidence', 'mean'),
        avg_radius=('radius', 'mean')
    ).reset_index()

    # --- Print Report ---
    print("\n" + "="*40)
    print("      CRATER DETECTION SUMMARY REPORT")
    print("="*40)
    print(f"Total Craters Found:  {total_craters}")
    print(f"Average Confidence:   {avg_conf:.2%}")
    print(f"Average Radius (px):  {avg_rad:.2f}")
    print("-" * 40)
    print(summary.to_string(index=False))
    print("-" * 40)

    # 4. Identifying potential issues
    low_conf = summary[summary['avg_confidence'] < 0.4]
    if not low_conf.empty:
        print("\n💡 Tip: Altitudes with low confidence may need more training data")
        print(f"Check: {low_conf['altitude'].tolist()}")

if __name__ == "__main__":
    generate_report()
