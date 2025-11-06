#! /usr/bin/env python3
"""
vision_service.py
-Opens camera (tries multiple indices)
-Captures a frame
-Runs a simple detector (Haar cascade face detector as a fallback)
-Saves a frame to jetson_services/vision/captures/
-Returns a short text summary and the saved filename
"""

import cv2
import torch
import ultralytics

try:
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except Exception:
    pass

import torch.serialization

torch.serialization._legacy_load = torch.load
def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return torch.serialization._legacy_load(*args, **kwargs)
torch.load = trusted_torch_load
print("Patched torch.load() for YOLO compatibility.")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

#Load YOLO once globally
print("Loading YOLOv8 model...")
print("Model loaded successfully.")


def find_camera_index(max_index=4, timeout=1.0):
    """
    Try camera indices 0..max_index-1 and return the first openable index.
    """
    for idx in range(max_index):
        cam = cv2.VideoCapture(idx)
        start = time.time()
        # small wait to let camra initialize
        while time.time() - start < timeout and not cam.isOpened():
            pass
        if cam.isOpened():
            cam.release()
            return idx
        cam.release()
    return None

def open_camera(index=None):
    """Open the camera. If index is None, try to auto-deect."""
    if index is None:
        index = find_camera_index()
        if index is None:
            raise RuntimeError("No camera found on indices 0..3. Try plugging camera or change index.")
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError(f"Failed to open camera at index {index}")
    # Try to set a small fixed resolution for speed
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam, index

def capture_frame(cam):
    """Capture a single frame from an already-open VideoCapture."""
    ret, frame = cam.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to read frame from camera.")
    return frame

def detect_objects_yolo(frame):
    """
    Detect people and other objects using YOLOv8-nano.
    Returns: Summary string, list of detections
    """
    import cv2

    #Run detection
    results = model.predict(frame, imgsz=640, verbose=False)
    detections = []

    #Each result is one image
    for result in results:
        boxes = result.boxes
        names = result.names

        if boxes is None or len(boxes) == 0:
            continue

        #iterate over every detected box
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            label = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2)
            })

            #Draw label and box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 -5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #Build summary
        labels = [d["label"] for d in detections]
        summary = "No objects detected."
        if labels:
            from collections import Counter
            counts = Counter(labels)
            parts = [f"{n} {lbl}(s)" for lbl, n in counts.items()]
            summary = "Detected: " + ", ".join(parts)

        return summary, detections

def save_frame(frame):
    """Save frame to captures dir and return path string."""
    ts = int(time.time())
    filename = CAPTURE_DIR / f"frame_{ts}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)

def run_vision_once(cam_index=0):
    import cv2, time, os

    try:
        cam = cv2.VideoCapture(cam_index)
        if not cam.isOpened():
            return {"ok": False, "summary": "Error: cannot open camera.", "img_path": None, "cam_index": cam_index}

        ret, frame = cam.read()
        if not ret:
            cam.release()
            return {"ok": False, "summary": "Error: failed to capture frame.", "img_path": None, "cam_index": cam_index}

        summary, detections = detect_objects_yolo(frame)

        CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captures")
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        ts = int(time.time())
        out_path = os.path.join(CAPTURE_DIR, f"frame_{ts}.jpg")
        cv2.imwrite(out_path, frame)
        cam.release()

        return {
            "ok": True,
            "summary": summary,
            "img_path": out_path,
            "cam_index": cam_index,
            "detections": detections
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "summary": f"Error: {e}", "img_path": None, "cam_index": cam_index}

if __name__ == "__main__":
    res = run_vision_once()
    print(res["summary"])
