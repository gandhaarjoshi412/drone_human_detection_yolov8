from ultralytics import YOLO
import cv2

model = YOLO("models\\best.pt")

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device=0, conf=0.4) 

    annotated_frame = results[0].plot()

    cv2.imshow("Human Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
