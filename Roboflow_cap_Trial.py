# from ultralytics import YOLO
# import cv2

# # Load your newly trained model
# model = YOLO("J:/projects Yash python/Vision Guard/runs/detect/train/weights/best.pt")


# # Start webcam
# cap = cv2.VideoCapture(1)  # or 1 depending on your camera

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Predict on the frame with lower threshold
#     # results = model.predict(frame, conf=0.1, verbose=False)
#     # results = model.predict(source="C:/Users/admin/Pictures/Camera Roll/WIN_20250503_20_33_00_Pro.jpg", conf=0.1)
#     # results[0].show()
#     results = model.predict(source="J:/projects Yash python/Vision Guard/train/images/WIN_20250503_20_32_34_Pro_jpg.rf.670250edf24f7e529500aeeb9bf8ef7e.jpg", conf=0.1, imgsz=640)
#     detections = results[0].boxes
# # results[0].show()


#     if detections is not None and len(detections) > 0:
#         for box in detections:
#             print(box.cls, box.conf)
#     else:
#         print("❌ Still no detections")

#     detections = results[0].boxes
#     found_cap = False
#     found_missing = False

#     if detections is not None and len(detections) > 0:
#         for box in detections:
#             cls = int(box.cls[0].item())
#             conf = float(box.conf[0].item())
#             print(f"🔍 Detected class {cls} with confidence {conf:.2f}")


#             if cls == 0:
#                 print(f"✅ CAP DETECTED (conf: {conf:.2f})")
#                 found_cap = True
#             elif cls == 1:
#                 print(f"❌ MISSING CAP DETECTED (conf: {conf:.2f})")
#                 found_missing = True

#     if not found_cap and not found_missing:
#         print("🔍 No cap-related object detected.")

#     # OPTIONAL: Show webcam output with detection
#     annotated = results[0].plot()
#     cv2.imshow("VisionGuard - Live Detection", annotated)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO

# Load the trained model
model = YOLO("J:/projects Yash python/Vision Guard/runs/detect/train/weights/best.pt")  # Or provide full path if needed

# Test on a training image that you labeled
results = model.predict(
    source="train/images/WIN_20250503_20_32_34_Pro_jpg.rf.670250edf24f7e529500aeeb9bf8ef7e.jpg",  # Replace with a real path
    conf=0.1,
    imgsz=640
)

# Show prediction
results[0].show()

# Print detection info
detections = results[0].boxes
if detections is not None and len(detections) > 0:
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"✅ Detected class {cls} with confidence {conf:.2f}")
else:
    print("❌ No detections")

