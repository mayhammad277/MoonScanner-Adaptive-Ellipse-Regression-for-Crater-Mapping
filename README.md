# MoonScanner-.-
Deep learning-based lunar topography analysis. This project implements a precise ellipse-regression model calibrated to high-resolution satellite imagery, utilizing adaptive scaling to accurately map craters of all sizes while preventing geometric overlaps.
-------------------------------------------------------------------------------------------------------------

📌 Overview: Lunar-GeoNet is a high-precision computer vision pipeline designed to automate the detection and measurement of lunar craters from high-resolution satellite imagery (Mahanti Dataset).

Unlike standard object detection that uses axis-aligned bounding boxes, this project utilizes Deep Heatmap Regression to predict center points and Adaptive Ellipse Geometry to model the exact physical rims of craters.The core challenge addressed in this repository is the extreme variance in crater scales—from tiny primary pits to massive impact basins—and the prevention of "geometric swallowing" in high-density regions.🚀 Key Technical Pillars1.

Heatmap-Based Center Regression. Instead of traditional anchor boxes, the model views the lunar surface as a probability field.
Point Localization: The network predicts a Sigmoid-activated heatmap where intensity represents crater-center probability.Sub-pixel Offsets: To overcome the limitations of the feature map resolution ($160 \times 160$), a dedicated offset head predicts local adjustments, allowing for precise center placement on the original $2592 \times 2592$ images.

Deep Learning Ellipse GeometryThe model's architecture branches into multiple regression heads to solve for the 5 parameters of an ellipse:Location: $(x, y)$ coordinates.Scale: Semimajor and Semiminor axes $(a, b)$.Orientation: Rotation angle $(\theta)$ in radians.


Adaptive Scaling & Overlap Prevention. Standard linear scaling often fails in planetary science due to the power-law distribution of crater sizes. We implemented a Custom Inference Engine featuring:Non-Linear Scaling: A hybrid Log-Power Law formula that aggressively boosts the visibility of tiny craters while "braking" the growth of large ones to prevent them from swallowing neighboring data.Geometric NMS (Non-Maximum Suppression): A custom spatial filtering algorithm that evaluates the distance between centers relative to the predicted radii, eliminating redundant false positives in crowded clusters.



IoU-Optimized Circularity: Automated aspect-ratio clamping ($1.25:1$) to maximize Intersection over Union (IoU) scores, ensuring geological realism.🛠️ Performance TuningThrough iterative refinement of the inference logic, the pipeline achieved a significant score improvement by focusing on:Reducing False Positives: Raising heatmap sensitivity thresholds in shadow-heavy regions.Size Calibration: Tuning the floor (minimum size) to $32\text{px}$ to align with Mahanti ground-truth standards.Stability: Forcing rotation-neutrality for nearly circular craters to reduce noise in the final submission.
  

├── inference.py        # Optimized prediction engine with Adaptive Scaling
├── model/              # Neural network architecture (Heads for axes, rot, hm)
├── utils/              # CLAHE enhancement and geometric conversion tools
├── weights/            # Pre-trained model checkpoints
└── README.md           # Project documentation



📈 Results
The adaptive inference logic successfully balanced the detection of minute surface features while maintaining distinct boundaries for large impact basins, significantly improving the Mean Average Precision (mAP) and IoU over standard linear regression baselines.

🤝 Contributing
If you're interested in planetary defense, lunar geology, or geometric deep learning, feel free to fork this repo and submit a PR.

"The moon is a cold mistress, but her geometry is perfect."
