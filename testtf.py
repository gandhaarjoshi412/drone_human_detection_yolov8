import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mean_average_precision import MetricBuilder

# -------------------
# CONFIG
# -------------------
TFLITE_MODEL = "yolov8.tflite"
IMAGES_DIR = "D:\\pycharm projects\\Yolov5n\\yolov8\\dataset\\images\\valid"   # folder with validation images
LABELS_DIR = "D:\\pycharm projects\\Yolov5n\\yolov8\\dataset\\labels\\valid"   # folder with YOLO-format labels
IMG_SIZE = 640              # must match model input
NUM_CLASSES = 80            # set this to your dataset classes (COCO=80)

# -------------------
# LOAD TFLITE MODEL
# -------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------
# PREPROCESS FUNCTION
# -------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    return np.expand_dims(img_norm.astype(np.float32), axis=0), img.shape[:2]

# -------------------
# YOLO LABEL LOADER
# -------------------
def load_yolo_labels(label_path, img_shape):
    h, w = img_shape
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            classes.append(int(cls))
    return boxes, classes

# -------------------
# INFERENCE FUNCTION
# -------------------
def infer_tflite(img_path):
    input_data, orig_shape = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(o['index']) for o in output_details]
    return output_data, orig_shape

# -------------------
# POSTPROCESS
# -------------------
def postprocess(output, orig_shape, conf_thresh=0.25):
    h, w = orig_shape
    predictions = output[0][0]  # [N, 85] (x, y, w, h, conf, cls...)
    boxes, scores, classes = [], [], []
    for pred in predictions:
        conf = pred[4]
        if conf < conf_thresh:
            continue
        cls = np.argmax(pred[5:])
        score = conf * pred[5:][cls]

        cx, cy, bw, bh = pred[0:4]
        # scale back to original image size
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h

        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        classes.append(cls)
    return boxes, scores, classes

# -------------------
# VALIDATION LOOP
# -------------------
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=NUM_CLASSES)

for img_file in tqdm(os.listdir(IMAGES_DIR)):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    
    img_path = os.path.join(IMAGES_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")

    # ground truth
    gt_boxes, gt_classes = load_yolo_labels(label_path, cv2.imread(img_path).shape[:2])

    # prediction
    output, orig_shape = infer_tflite(img_path)
    pred_boxes, pred_scores, pred_classes = postprocess(output, orig_shape)

    preds = [[*box, cls, score] for box, cls, score in zip(pred_boxes, pred_classes, pred_scores)]
    gts   = [[*box, cls, 1.0] for box, cls in zip(gt_boxes, gt_classes)]

    metric_fn.add(preds, gts)

# -------------------
# RESULTS
# -------------------
print("mAP@0.5:", metric_fn.value(iou_thresholds=[0.5])["mAP"])
print("mAP@[.5:.95]:", metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05))["mAP"])
