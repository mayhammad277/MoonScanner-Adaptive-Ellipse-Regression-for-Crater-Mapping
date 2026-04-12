import os
import csv
import argparse
import math
from typing import List, Set


def combine_detections(root_dir: str, out_path: str) -> int:
    """
    Walk root_dir, find every file named detections.csv and concatenate them into out_path.
    A column named "inputImage" will be modified to contain two elements taken from
    the file's full path: the 4th-from-last and 3rd-from-last path components joined
    (for example "grandparentdir/parentdir").

    Certain columns that are not needed in the combined output are dropped.
    Returns number of rows written (excluding header).
    """
    # collect csv and image paths
    csv_paths: List[str] = []
    image_ids: Set[str] = set()
    image_ids_written: Set[str] = set()

    for dirpath, _, files in os.walk(root_dir):
        if 'detections.csv' in files:
            csv_paths.append(os.path.join(dirpath, "detections.csv"))
        if not 'truth' in dirpath:
            norm = os.path.normpath(dirpath)
            dirparts = [part for part in norm.split(os.sep) if part]
                    
            for f in files:
                if f.endswith('.png'):
                    # Build ID from the full path parts: 4th-from-last and 3rd-from-last.
                    # join using os.path.join so it is platform-correct
                    id = os.path.join(dirparts[-2], dirparts[-1], f[:-4])  # remove .png
                    image_ids.add(id)
    
    if not csv_paths:
        return 0

    # build canonical header union preserving order
    header_union: List[str] = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for h in reader.fieldnames:
                    if h not in header_union:
                        header_union.append(h)

    # columns to remove from final output (if present)
    drop_cols = [
        "detectionConfidence",
        "boundingBoxMinX(px)",
        "boundingBoxMinY(px)",
        "boundingBoxMaxX(px)",
        "boundingBoxMaxY(px)",
        "crater_id_Robbins"
    ]

    # remove any dropped columns from header union while preserving order
    header_union = [h for h in header_union if h not in drop_cols]

    # We will combine 'source' and 'inputImage' into a single 'inputImage' column.
    # Remove any existing 'source' entry from the header.
    if "source" in header_union:
        header_union = [h for h in header_union if h != "source"]

    # Ensure 'inputImage' exists in the header (append at end if missing).
    if "inputImage" not in header_union:
        header_union.append("inputImage")

    # write combined CSV streaming rows
    rows_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=header_union)
        writer.writeheader()
        for p in csv_paths:
            reldir = os.path.relpath(os.path.dirname(p), root_dir)
            with open(p, newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    # drop unwanted columns from the row if present
                    for dc in drop_cols:
                        if dc in row:
                            row.pop(dc, None)

                    # Build source from the full path parts: 4th-from-last and 3rd-from-last.
                    norm = os.path.normpath(p)
                    parts = [part for part in norm.split(os.sep) if part]
                    # join using os.path.join so it is platform-correct
                    source_val = os.path.join(parts[-4], parts[-3])
                
                    # Combine source and inputImage into a single inputImage column.
                    img_val = row.get("inputImage", "") or ""
                    # remove trailing .png if present (case-insensitive)
                    if isinstance(img_val, str) and img_val.lower().endswith(".png"):
                        img_val = img_val[:-4]

                    combined = os.path.join(source_val, img_val) if img_val else source_val
                    
                    # set the combined value back to inputImage and remove source key
                    row["inputImage"] = combined
                    row.pop("source", None)

                    # check if this crater passes all criteria to be written
                    cx = float(row.get("ellipseCenterX(px)"))
                    if cx < 0 or cx >= 2592:
                        continue    
                    cy = float(row.get("ellipseCenterY(px)"))
                    if cy < 0 or cy >= 2048:
                        continue    
                    a = float(row.get("ellipseSemimajor(px)"))
                    b = float(row.get("ellipseSemiminor(px)"))
                    alpha = float(row.get("ellipseRotation(deg)"))
                    # see https://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse
                    ax = math.pow(a * math.cos(math.radians(alpha)), 2)
                    bx = math.pow(b * math.sin(math.radians(alpha)), 2)
                    dx = math.sqrt(ax + bx)
                    ay = math.pow(a * math.sin(math.radians(alpha)), 2)
                    by = math.pow(b * math.cos(math.radians(alpha)), 2)
                    dy = math.sqrt(ay + by)
                    if cx - dx < 0 or cx + dx >= 2592:
                        continue
                    if cy - dy < 0 or cy + dy >= 2048:
                        continue   
                    if 2 * (dx + dy) > 0.6 * 2048:
                        continue

                    # Ensure numeric output is formatted to two decimal places.
                    out_row = {}
                    for field in header_union:
                        # Prefer the value from the current row; fall back to empty string.
                        val = row.get(field, "")
                        # Don't try to format None or empty strings
                        if val is None or val == "":
                            out_row[field] = ""
                            continue
                        # keep crater_classification as is
                        if field == "crater_classification":
                            out_row[field] = val
                            continue
                        # Attempt to parse as float; if successful, format to 2 decimal places
                        try:
                            # Some CSV readers return strings; convert safely
                            f = float(val)
                        except (ValueError, TypeError):
                            out_row[field] = val
                        else:
                            out_row[field] = f"{f:.2f}"

                    writer.writerow(out_row)
                    image_ids_written.add(combined)
                    rows_written += 1

        # report any image ids that were not written
        missing_ids = image_ids - image_ids_written
        if missing_ids:
            for id in sorted(missing_ids):
                print(f"Warning: no craters for image id '{id}'")
                out_row = {}
                for field in header_union:
                    if field == "inputImage":
                        out_row[field] = id
                    else:
                        out_row[field] = "-1" 
                writer.writerow(out_row)

    return rows_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine detections.csv files under a directory tree.")
    # ignore the defaults here, or rewrite them to match your setup
    parser.add_argument("--root", default="../data/train", help="Root directory to walk")
    parser.add_argument("--out", default="./train-gt.csv", help="Output CSV path")
    args = parser.parse_args()

    count = combine_detections(args.root, args.out)
    print(f"Wrote {count} rows to {args.out}")
