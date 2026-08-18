import cv2
import numpy as np
import time
from ultralytics import YOLO

# ---------------- DETECTOR ----------------
class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")  # 🔥 stronger model
        self.names = self.model.names

    def detect(self, frame):
        # 🔥 improved detection settings (for ball + small objects)
        results = self.model(frame, conf=0.15, iou=0.5, imgsz=832, verbose=False)[0]

        detections = []

        allowed = [
            "person", "sports ball",
            "bottle", "cup", "cell phone",
            "laptop", "book", "backpack",
            "mouse", "keyboard"
        ]

        for box in results.boxes:
            cls = int(box.cls[0])
            name = self.names[cls]

            if name in allowed:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((name, (x1, y1, x2, y2)))

        return detections


# ---------------- ACTION ----------------
def classify_action(speed):
    # 🔥 improved sensitivity
    if speed < 5:
        return "Standing", (200,200,200)
    elif speed < 20:
        return "Running", (0,255,0)
    elif speed < 40:
        return "Defending", (255,165,0)
    else:
        return "Attacking", (0,0,255)


# ---------------- SYSTEM ----------------
class SportsSystem:
    def __init__(self):
        self.detector = YOLODetector()
        self.prev_center = None
        self.frame_count = 0
        self.start_time = time.time()
        self.last_objects = {}  # 🔥 stability memory

    def process(self, frame):
        self.frame_count += 1
        detections = self.detector.detect(frame)

        annotated = frame.copy()
        person_done = False
        stable_objects = {}

        for name, (x1,y1,x2,y2) in detections:

            cx = (x1 + x2)//2
            cy = (y1 + y2)//2

            # 🔥 smoothing (stable detection)
            if name in self.last_objects:
                prev = self.last_objects[name]
                cx = int(0.7 * prev[0] + 0.3 * cx)
                cy = int(0.7 * prev[1] + 0.3 * cy)

            stable_objects[name] = (cx, cy)

            # ---- PERSON ----
            if name == "person" and not person_done:
                person_done = True

                if self.prev_center:
                    vx = cx - self.prev_center[0]
                    vy = cy - self.prev_center[1]
                    speed = np.sqrt(vx*vx + vy*vy)
                else:
                    speed = 0

                action, color = classify_action(speed)

                # 🔥 clear action text
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated, f"Person: {action}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1,color,3)

                self.prev_center = (cx,cy)

            # ---- OTHER OBJECTS ----
            else:
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,0),2)
                cv2.putText(annotated, name, (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        self.last_objects = stable_objects

        # 🔥 ball fallback display
        ball_found = any(name == "sports ball" for name, _ in detections)
        if not ball_found:
            cv2.putText(annotated, "Ball: Not Detected", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # FPS
        fps = self.frame_count / (time.time() - self.start_time)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        return annotated


# ---------------- VIDEO ----------------
def run_video(path):
    cap = cv2.VideoCapture(path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(3))
    h = int(cap.get(4))

    # 🔥 SAVE IN YOUR MAJOR PROJECT FOLDER
    output_path = r"C:\Users\91733\Desktop\major project\output.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    system = SportsSystem()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = system.process(frame)

        out.write(output)
        cv2.imshow("Video Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Saved at: {output_path}")


# ---------------- LIVE ----------------
def run_live():
    cap = cv2.VideoCapture(0)

    system = SportsSystem()

    print("✅ Live started (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = system.process(frame)

        cv2.imshow("Live Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- MAIN ----------------
print("1. Live Camera")
print("2. Recorded Video")

choice = input("Enter choice: ")

if choice == "1":
    run_live()
elif choice == "2":
    path = input("Enter video path: ")
    run_video(path)
else:
    print("Invalid choice")