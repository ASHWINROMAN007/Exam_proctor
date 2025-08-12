# monitor.py
import cv2
import csv
import os
import time
from datetime import datetime
import mediapipe as mp
from ultralytics import YOLO

# -------------- CONFIG --------------
FACE_MISSING_SECONDS = 2.0     # seconds of no-face before flag
PHONE_CONF_THRESHOLD = 0.5
VIDEO_SOURCE = 0               # 0 = default webcam
LOG_CSV = "logs/violations.csv"
YOLO_WEIGHTS = "yolov8n.pt" # default; change to your trained weights for headphones
# ------------------------------------

os.makedirs("logs", exist_ok=True)

# create CSV header if not exists
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "violation_type", "confidence"])

# init mediapipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.55)

# init YOLOv8
model = YOLO(YOLO_WEIGHTS)  # will download yolov8n.pt if not present

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions and device index.")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
missing_start = None

def log_violation(vtype, confidence):
    ts = datetime.now().isoformat()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, vtype, f"{confidence:.3f}"])
    print(f"[{ts}] {vtype} ({confidence:.2f})")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Face detection (Mediapipe) ---
        face_res = face_detector.process(rgb)
        face_present = bool(face_res.detections)

        if face_present:
            missing_start = None
            cv2.putText(frame, "Face: present", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(frame, "Face: MISSING", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if missing_start is None:
                missing_start = time.time()
            else:
                elapsed = time.time() - missing_start
                if elapsed >= FACE_MISSING_SECONDS:
                    log_violation("face_missing", 1.0)
                    missing_start = None  # avoid repeated logs immediately

        # --- Object detection (YOLOv8) ---
        # run model on the frame (ultralytics accepts numpy arrays)
        results = model(frame)[0]  # get first result
        if hasattr(results, "boxes") and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # draw bbox + label
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,200,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

                # phone check (COCO label is "cell phone")
                if label.lower() in ("cell phone", "phone", "mobile phone", "cellphone") and conf >= PHONE_CONF_THRESHOLD:
                    log_violation("phone", conf)

                # headphone: placeholder (if you train a 'headphones' class, check label here)
                if label.lower() in ("headphones", "earphones", "headset") and conf >= 0.4:
                    log_violation("headphone", conf)

        # show frame
        cv2.imshow("Exam Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    print("Stopped.")
