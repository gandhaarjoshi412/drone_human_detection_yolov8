This repository contains the validation results of a **YOLOv8 Nano** model trained on the [Zenodo Images.zip dataset](https://zenodo.org/record/7740081).  
The dataset consists of **person vs background** images, used to evaluate object detection performance.

---

##  Validation Summary

- **Model:** YOLOv8 Nano  
- **Dataset:** Person detection (Zenodo)  
- **mAP@0.5:** ~0.392  
- **Best F1-score:** ~0.45 at confidence ≈ 0.107  
- **Recall:** ~0.33 (many missed detections)  
- **Precision:** High at high thresholds (few false positives)  

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

##  Observations

- High **precision** at higher confidence thresholds → low false positives.  
- Low **recall** → many persons not detected.  
- The model is underfitting and struggles with small or occluded persons.

---
