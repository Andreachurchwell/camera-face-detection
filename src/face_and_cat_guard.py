import time
from datetime import datetime
import ctypes

import cv2
from ultralytics import YOLO


def save_frame(frame, folder="captures"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{folder}/cat_{ts}.jpg"
    cv2.imwrite(path, frame)
    return path


def lock_computer():
    print("[ACTION] Locking workstation")
    ctypes.windll.user32.LockWorkStation()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1) if needed.")

    # Face detector (cheap)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade.")

    # Cat detector (heavier)
    model = YOLO("yolov8n.pt")

    # Motion gate settings
    prev_gray = None
    motion_area_threshold = 1500
    motion_cooldown_sec = 20  # recommended to avoid repeated locks
    last_trigger_time = 0

    # YOLO throttling
    yolo_min_interval_sec = 0.5
    last_yolo_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- FACE DETECTION ----------
        gray_for_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_for_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            cv2.putText(
                frame,
                "FACE",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

        # ---------- MOTION GATE ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected = False
        if prev_gray is not None:
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < motion_area_threshold:
                    continue
                motion_detected = True
                (mx, my, mw, mh) = cv2.boundingRect(c)
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 255, 255), 2)

        prev_gray = gray

        # ---------- CAT DETECTION (only if motion) ----------
        cat_found = False
        if motion_detected:
            now = time.time()
            if now - last_yolo_time >= yolo_min_interval_sec:
                last_yolo_time = now

                results = model.predict(frame, verbose=False)
                r0 = results[0]

                if r0.boxes is not None and len(r0.boxes) > 0:
                    for box in r0.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        # "cat" is class 15 in COCO
                        if cls_id == 15 and conf >= 0.60:
                            cat_found = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"CAT {conf:.2f}",
                                (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                if cat_found and (now - last_trigger_time) >= motion_cooldown_sec:
                    last_trigger_time = now
                    path = save_frame(frame)
                    print(f"[CAT DETECTED] saved: {path}")
                    lock_computer()

        # ---------- OVERLAY ----------
        status = "MOTION" if motion_detected else "still"
        cv2.putText(
            frame,
            f"status: {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Face + Cat Guard (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()