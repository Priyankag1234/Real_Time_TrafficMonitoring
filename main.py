import cv2
import numpy as np
from ultralytics import YOLO
from datetime import timedelta
from PIL import Image
import os

# ========== SETTINGS ==========
VIDEO_PATH = VIDEO_PATH = r"C:\Users\Priyanka Gupta\Downloads\real time traffic monitoring\input_video.mp4"
MODEL_PATH = "yolov8n.pt"
STOP_LINE_Y = 250
MAX_FRAMES = 100
SPEED_THRESHOLD = 30  # pixels/frame
TRAFFIC_SIGNAL = "RED"

# ========== LOAD MODEL ==========
model = YOLO(MODEL_PATH)

# ========== VIDEO CAPTURE ==========
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = 0
tracking_memory = {}

# ========== HELPER FUNCTIONS ==========
def compute_speed(object_id, center, current_frame):
    if object_id not in tracking_memory:
        tracking_memory[object_id] = (center, current_frame)
        return 0.0

    prev_center, prev_frame = tracking_memory[object_id]
    distance = np.linalg.norm(np.array(center) - np.array(prev_center))
    elapsed = current_frame - prev_frame
    speed = distance / elapsed if elapsed > 0 else 0
    tracking_memory[object_id] = (center, current_frame)
    return speed

def draw_annotations(frame, box, label, speed=None):
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if speed is not None:
        cv2.putText(frame, f"Speed: {speed:.1f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 2)

def process_frame(frame, frame_index):
    detections = model(frame)
    timestamp = timedelta(seconds=frame_index / fps)
    warnings = []

    for i, box in enumerate(detections[0].boxes.data):
        x1, y1, x2, y2, conf, cls = box
        label = model.names[int(cls)]

        if label in ['car', 'bus', 'truck']:
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            obj_id = f"{label}_{i}"
            speed = compute_speed(obj_id, center, frame_index)

            draw_annotations(frame, (x1, y1, x2, y2), label, speed)

            if speed > SPEED_THRESHOLD:
                warnings.append(f"[{timestamp}] 🚨 Speeding: {label}, Speed={speed:.1f}")

            if TRAFFIC_SIGNAL == "RED" and y2 < STOP_LINE_Y:
                warnings.append(f"[{timestamp}] 🚨 Red Light Violation: {label}")

    cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (0, 0, 255), 2)
    return frame, warnings

# ========== MAIN LOOP ==========
while cap.isOpened() and frame_num < MAX_FRAMES:
    success, frame = cap.read()
    if not success:
        break

    frame, alerts = process_frame(frame, frame_num)

    for alert in alerts:
        print(alert)

    cv2.imshow("Traffic Violation Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
