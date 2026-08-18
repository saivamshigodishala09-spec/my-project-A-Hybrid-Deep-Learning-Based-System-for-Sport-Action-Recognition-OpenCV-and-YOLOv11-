import cv2
import numpy as np
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict

# ---------------- DATA CLASSES ----------------
@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class PlayerState:
    track_id: int
    centroid: Tuple[int, int]
    velocity: Tuple[float, float]
    trajectory: deque


# ---------------- YOLO DETECTOR ----------------
class YOLODetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n.pt")
        self.names = self.model.names

    def detect(self, frame, mode="sports"):
        results = self.model(frame, conf=0.25, iou=0.5, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.names[cls]
            conf = float(box.conf[0])

            # filter
            if mode == "sports":
                if name not in ["person", "sports ball"]:
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, name, conf))

        return detections


# ---------------- SIMPLE STABLE TRACKER ----------------
class Tracker:
    def __init__(self):
        self.objects = {}
        self.next_id = 0

    def update(self, detections):
        persons = [d for d in detections if d[4] == "person"]
        new_objects = {}

        for (x1, y1, x2, y2, _, _) in persons:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            assigned = False
            for oid, (px, py) in self.objects.items():
                if abs(cx - px) < 50 and abs(cy - py) < 50:
                    new_objects[oid] = (cx, cy)
                    assigned = True
                    break

            if not assigned:
                new_objects[self.next_id] = (cx, cy)
                self.next_id += 1

        self.objects = new_objects
        return self.objects


# ---------------- ACTION CLASSIFIER ----------------
class ActionClassifier:
    COLORS = {
        "Standing": (200, 200, 200),
        "Walking": (255, 255, 0),
        "Running": (0, 255, 0),
        "Attacking": (0, 0, 255),
        "Defending": (255, 165, 0)
    }

    def classify(self, velocity, trajectory):
        vx, vy = velocity
        speed = np.linalg.norm([vx, vy])

        if len(trajectory) >= 5:
            dx = trajectory[-1][0] - trajectory[0][0]
            dy = trajectory[-1][1] - trajectory[0][1]
        else:
            dx, dy = vx, vy

        if speed < 1.5:
            return "Standing"
        elif speed < 5:
            return "Walking"
        elif speed < 12:
            return "Running"
        elif abs(dx) > abs(dy):
            return "Attacking"
        else:
            return "Defending"


# ---------------- MAIN SYSTEM ----------------
class System:
    def __init__(self):
        self.detector = YOLODetector()
        self.tracker = Tracker()
        self.classifier = ActionClassifier()

        self.players: Dict[int, PlayerState] = {}
        self.mode = "sports"

        self.start_time = time.time()
        self.frame_count = 0

    def process_frame(self, frame):
        self.frame_count += 1

        detections = self.detector.detect(frame, self.mode)
        tracks = self.tracker.update(detections)

        output = frame.copy()

        # -------- DRAW OBJECTS --------
        for (x1, y1, x2, y2, name, conf) in detections:
            if name == "person":
                continue

            if name == "sports ball":
                color = (0, 255, 255)
                label = f"BALL {conf:.2f}"
            else:
                color = (255, 0, 0)
                label = name

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # -------- PLAYER ACTION --------
        persons = [d for d in detections if d[4] == "person"]

        for i, ((x1, y1, x2, y2, _, _)) in enumerate(persons):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # find closest track id
            pid = None
            for oid, (tx, ty) in tracks.items():
                if abs(cx - tx) < 50 and abs(cy - ty) < 50:
                    pid = oid
                    break

            if pid is None:
                continue

            if pid not in self.players:
                self.players[pid] = PlayerState(
                    pid, (cx, cy), (0, 0), deque(maxlen=10)
                )

            p = self.players[pid]

            vx = cx - p.centroid[0]
            vy = cy - p.centroid[1]

            p.velocity = (vx, vy)
            p.centroid = (cx, cy)
            p.trajectory.append((cx, cy))

            action = self.classifier.classify(p.velocity, p.trajectory)
            color = self.classifier.COLORS[action]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"ID:{pid} {action}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        fps = self.frame_count / (time.time() - self.start_time)
        cv2.putText(output, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output

    def run_video(self, path, save_path=None):
        self.mode = "sports"
        cap = cv2.VideoCapture(path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_frame = self.process_frame(frame)

            if save_path and out is None:
                h, w = out_frame.shape[:2]
                out = cv2.VideoWriter(save_path,
                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (w, h))

            if out:
                out.write(out_frame)

            cv2.imshow("Video", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def run_camera(self):
        self.mode = "general"
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_frame = self.process_frame(frame)
            cv2.imshow("Live", out_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ---------------- MAIN ----------------
sys = System()

print("1. Video")
print("2. Live")

choice = input("Enter choice: ")

if choice == "1":
    video = input("Enter video path: ")
    save = input("Save output? (y/n): ")

    if save == "y":
        out = input("Enter full output path (.mp4): ")
    else:
        out = None

    sys.run_video(video, out)

elif choice == "2":
    sys.run_camera()