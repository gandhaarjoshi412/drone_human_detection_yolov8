import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["person"] 

def run_inference_tflite(model_path, frame):

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    
        input_shape = input_details[0]['shape']
        input_width = input_shape[1]
        input_height = input_shape[2]

        image = cv2.resize(frame, (input_width, input_height))
        input_data = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        if output_data.shape[1] < output_data.shape[2]:
            output_data = np.transpose(output_data, (0, 2, 1))

        if output_data.shape[2] == 5:
            boxes = output_data[0, :, :4]
            scores = output_data[0, :, 4]
            class_ids = np.zeros_like(scores, dtype=np.int32)
        else:
            boxes = output_data[0, :, :4]
            scores = output_data[0, :, 4]
            class_ids = np.argmax(output_data[0, :, 5:], axis=1)

        confidence_threshold = 0.4
        valid_detections = scores > confidence_threshold

        filtered_boxes = boxes[valid_detections]
        filtered_scores = scores[valid_detections]
        filtered_class_ids = class_ids[valid_detections]

        x_center, y_center, w, h = np.transpose(filtered_boxes)
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        nms_boxes = np.vstack([x1, y1, x2, y2]).T

        nms_threshold = 0.4  
        indices = tf.image.non_max_suppression(
            boxes=nms_boxes,
            scores=filtered_scores,
            max_output_size=50, 
            iou_threshold=nms_threshold
        )
        
        final_indices = indices.numpy()
        final_boxes = filtered_boxes[final_indices]
        final_scores = filtered_scores[final_indices]
        final_class_ids = filtered_class_ids[final_indices]

        return final_boxes, final_scores, final_class_ids

    except Exception as e:
        print(f"Error during TFLite inference: {e}")
        return [], [], []

def draw_boxes(frame, boxes, scores, class_ids, class_names):

    for box, score, class_id in zip(boxes, scores, class_ids):
        x_center, y_center, w, h = box
        frame_h, frame_w, _ = frame.shape
        x1 = int((x_center - w/2) * frame_w)
        y1 = int((y_center - h/2) * frame_h)
        x2 = int((x_center + w/2) * frame_w)
        y2 = int((y_center + h/2) * frame_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{class_names[class_id]}: {score:.2f}"
        
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return frame

def main():

    model_path = "models\\best_float32.tflite"

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        boxes, scores, class_ids = run_inference_tflite(model_path, frame.copy())

        annotated_frame = draw_boxes(frame, boxes, scores, class_ids, CLASS_NAMES)

        cv2.imshow("TFLite Human Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
