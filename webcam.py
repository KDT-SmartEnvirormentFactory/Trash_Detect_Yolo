import cv2
import supervision as sv
from ultralytics import YOLO

try:

    from picamera2 import Picamera2
    USE_PICAMERA = True
    print("Using Raspberry Pi camera mode")

except ImportError:
    USE_PICAMERA = False
    print("Using standard webcam mode")


from threading import Thread
import time

class ThreadedCamera:
    def __init__(self, use_picamera: bool, width: int = 640, height: int = 480):
        self.use_picamera = use_picamera
        self.frame = None
        self.running = True
        if self.use_picamera:
            self.picam2 = Picamera2()
            # Disable RAW(CFE) stream by setting raw=None, use video configuration for continuous capture
            config = self.picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                raw=None
            )

            self.picam2.configure(config)
            self.picam2.start()
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                raise RuntimeError("Camera is not connected.")

        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.use_picamera:
                frame = self.picam2.capture_array()
            else:
                ret, frame = self.cap.read()
                if not ret:
                    continue
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        if self.use_picamera:
            self.picam2.stop()
        else:
            self.cap.release()

# YOLO model initialization

# Use the YOLO model specified in the best.pt file

import os
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
model = YOLO(model_path)
print(f"Model loading completed: {model_path}")
print(f"Trained classes: {model.names}")

# Tracking and detection processing
# Use supervision to display bounding boxes around detected objects and label them
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Camera connection and configuration
try:
    camera = ThreadedCamera(USE_PICAMERA)
    if USE_PICAMERA:
        print("Raspberry Pi camera startup completed")
except Exception as e:
    print(f"Camera initialization failed: {e}")
    raise

# Frame processing and object detection loop
# Read frames from the camera in an infinite loop and detect objects using YOLO model
while True:
    frame = camera.read()
    if frame is None:
        time.sleep(0.01)
        continue

    # Convert YOLOv10 results to supervision Detections object using sv.Detections.from_ultralytics()

    # Lower confidence threshold to detect more objects
    results = model(frame, conf=0.1, verbose=False)[0]  # Set confidence threshold to 10%
    
    # Check raw results
    if len(results.boxes) > 0:
        print(f"\nRaw detection results: {len(results.boxes)} objects")
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  Class: {model.names[cls]}, Confidence: {conf:.2%}")

    detections = sv.Detections.from_ultralytics(results)

    # Print number of detected objects
    print(f"Detected objects: {len(detections)} objects")
    if len(detections) > 0:
        for i, detection in enumerate(detections):
            print(f"  - Object {i+1}: Confidence {detection[2]:.2%}")

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    # Display the frame from the webcam using cv2.imshow()
    cv2.imshow('webcam', annotated_image)

    # Press ESC key to exit the loop and terminate the program
    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Object detection terminated")
        break

# When the program terminates, release camera resources and close all OpenCV windows
camera.stop()
print("Camera stopped")
cv2.destroyAllWindows()
cv2.destroyAllWindows()