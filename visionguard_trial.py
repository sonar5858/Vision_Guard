from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8n (nano) model - fastest
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction on the frame
    results = model.predict(frame, conf=0.5, verbose=False)

    # Draw detection results
    annotated_frame = results[0].plot()

    # Show live result
    cv2.imshow("VisionGuard - YOLOv8 Live Detection", annotated_frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
