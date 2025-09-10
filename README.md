# YOLOv8 Person Detection Validation Results

This repository contains the validation results 

---

##  Validation Summary

- **Model:** YOLOv8 Nano  
- **Dataset:** Person detection (Zenodo)  
- **mAP@0.5:** 0.421  
- **Best F1-score:** 0.47 at confidence ≈ 0.104  
- **Recall:** 0.45 at confidence 0.000 (maximum recall achieved)  
- **Precision:** 1.00 at confidence 0.925 (maximum precision achieved)  

---

##  Results & Plots

The following evaluation plots are included in this repository:

- Confusion Matrix (`confusion_matrix.png`)
- Normalized Confusion Matrix (`confusion_matrix_normalized.png`)
- F1 Curve (`BoxF1_curve.png`)
- Precision Curve (`BoxP_curve.png`)
- Precision–Recall Curve (`BoxPR_curve.png`)
- Recall Curve (`BoxR_curve.png`)

All results are also summarized in the PDF report:  
 [YOLOv8 Validation Report](yolov8_validation_report.pdf)

---

##  Key Observations

- **High precision** at higher confidence thresholds (up to 1.00) → very few false positives
- **Moderate recall** (0.45 maximum) → some persons not detected, indicating room for improvement
- **Balanced performance** at optimal F1 threshold (~0.104 confidence)
- **Class distribution**: 4,553 true person instances, 7,029 true background instances
- **Detection accuracy**: 3,560 correct person detections, 993 false negatives

---

##  Performance Analysis

The model shows:
- Strong precision capability with minimal false positives when tuned appropriately
- Reasonable recall performance but could benefit from improvements to catch more person instances
- Well-balanced performance at the optimal F1 operating point
- Good discrimination between person and background classes

---
