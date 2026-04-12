#!/bin/bash
set -e

DATA_DIR=$1
OUTPUT_CSV=$2
CLASSIFY_FLAG=$3

echo "===================================="
echo "Topcoder Inference Script Started"
echo "Data directory: ${DATA_DIR}"
echo "Output CSV: ${OUTPUT_CSV}"
echo "Classify flag: ${CLASSIFY_FLAG}"
echo "===================================="

if [ "$CLASSIFY_FLAG" == "--classify" ]; then
    echo "Running WITH classification"
    python3 inference.py "$DATA_DIR" "$OUTPUT_CSV" --classify
else
    echo "Running WITHOUT classification"
    python3 inference.py "$DATA_DIR" "$OUTPUT_CSV"
fi

echo "===================================="
echo "Topcoder Inference Script Finished"
echo "===================================="

