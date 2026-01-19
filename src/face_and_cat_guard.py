import time
from datetime import datetime
import ctypes

import cv2
from ultralytics import YOLO


# -----------------------------
# Utility: Save a snapshot frame
# -----------------------------
def save_frame(frame, folder="captures"):
    """
    Saves the current frame to disk with a timestamp filename.

    Why this exists:
    - When a cat is detected, we want proof (a screenshot).
    - Using a timestamp prevents overwriting previous captures.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{folder}/cat_{ts}.jpg"

    # cv2.imwrite writes the image array (frame) to a file.
    cv2.imwrite(path, frame)
    return path


# ------------------------------------------
# Utility: Lock the computer (Windows-only)
# ------------------------------------------
def lock_computer():
    """
    Locks the Windows workstation (same as pressing Win + L).

    Why this is safer than "disabling keyboard":
    - It's a standard OS lock screen action.
    - It’s reversible (you just log back in).
    - It doesn't mess with device drivers or low-level input hooks.
    """
    print("[ACTION] Locking workstation")
    ctypes.windll.user32.LockWorkStation()


def main():
    # ------------------------------------------
    # 1) Open the webcam
    # ------------------------------------------
    # VideoCapture(0) usually points to the built-in laptop camera.
    # If it fails, try 1 or 2 (external cameras or different indexes).
    cap = cv2.VideoCapture(0)

    # Always verify the webcam opened successfully.
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1) if needed.")

    # ------------------------------------------
    # 2) Load the face detector (Haar cascade)
    # ------------------------------------------
    # Haar cascades are classic OpenCV face detectors:
    # - fast on CPU
    # - good for simple real-time face boxes
    # - not as accurate as modern deep learning, but lightweight
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # If empty() is True, the XML file didn't load.
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade.")

    # ------------------------------------------
    # 3) Load the cat detector (YOLO model)
    # ------------------------------------------
    # YOLO is heavier than Haar cascades.
    # That’s why we "gate" it behind motion detection (only run YOLO when motion happens).
    model = YOLO("yolov8n.pt")

    # ------------------------------------------
    # 4) Motion gate settings
    # ------------------------------------------
    # prev_gray:
    # - holds the previous frame (grayscale & blurred)
    # - used to compare with the current frame and detect changes (motion)
    prev_gray = None

    # motion_area_threshold:
    # - ignore small movement (noise, tiny lighting changes)
    # - raise this if you're getting too many false motion triggers
    # - lower this if you miss motion
    motion_area_threshold = 1500

    # motion_cooldown_sec:
    # - prevents repeated locking and repeated snapshot spam
    # - if your cat stays in frame, you don't want it to lock every second
    motion_cooldown_sec = 20
    last_trigger_time = 0  # stores the last time we triggered a lock/snapshot

    # ------------------------------------------
    # 5) YOLO throttling settings
    # ------------------------------------------
    # YOLO is expensive. Even when motion is happening, we don't want to run it every frame.
    # yolo_min_interval_sec:
    # - runs YOLO at most once every X seconds when motion is detected
    yolo_min_interval_sec = 0.5
    last_yolo_time = 0

    # ------------------------------------------
    # Main loop: read frames until user quits
    # ------------------------------------------
    while True:
        # ret tells us if we successfully got a frame.
        # frame is the actual image array.
        ret, frame = cap.read()
        if not ret:
            break

        # ==========================
        # FACE DETECTION (every frame)
        # ==========================
        # Convert frame to grayscale because Haar cascade expects grayscale input.
        gray_for_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectMultiScale returns rectangles (x, y, w, h) for detected faces.
        faces = face_cascade.detectMultiScale(
            gray_for_face,
            scaleFactor=1.1,  # how much to shrink image at each scale
            minNeighbors=5,   # higher = fewer false positives (but might miss small faces)
            minSize=(40, 40), # ignore very small detections
        )

        # Draw face boxes + label on the original color frame.
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

        # ==========================
        # MOTION DETECTION (motion gate)
        # ==========================
        # We detect motion by comparing current frame vs previous frame.
        # Using grayscale + blur reduces noise and makes motion detection more stable.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected = False

        # Only do motion diff if we have a previous frame to compare to.
        if prev_gray is not None:
            # delta is the absolute difference between the two frames.
            delta = cv2.absdiff(prev_gray, gray)

            # threshold turns the delta image into black/white.
            # Anything above 25 becomes white (motion), else black.
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate fills in holes and makes motion blobs more solid.
            thresh = cv2.dilate(thresh, None, iterations=2)

            # findContours finds “blobs” of motion in the thresholded image.
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop over motion blobs and ignore tiny ones
            for c in contours:
                if cv2.contourArea(c) < motion_area_threshold:
                    continue

                # If we hit here, we found meaningful motion.
                motion_detected = True

                # Draw motion bounding box in white.
                (mx, my, mw, mh) = cv2.boundingRect(c)
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 255, 255), 2)

        # Update prev_gray for the next loop iteration.
        prev_gray = gray

        # ==========================
        # CAT DETECTION (only if motion)
        # ==========================
        cat_found = False

        if motion_detected:
            now = time.time()

            # Throttle YOLO so it doesn't run every frame.
            if now - last_yolo_time >= yolo_min_interval_sec:
                last_yolo_time = now

                # Run YOLO on the current frame.
                # verbose=False keeps terminal output clean.
                results = model.predict(frame, verbose=False)
                r0 = results[0]  # first (and only) frame result

                # YOLO returns detected boxes with class IDs and confidences.
                if r0.boxes is not None and len(r0.boxes) > 0:
                    for box in r0.boxes:
                        cls_id = int(box.cls[0])     # class ID (COCO classes)
                        conf = float(box.conf[0])    # confidence score 0..1

                        # In the COCO dataset, class 15 = cat.
                        # conf threshold controls how strict detection is.
                        if cls_id == 15 and conf >= 0.60:
                            cat_found = True

                            # box.xyxy gives [x1, y1, x2, y2]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Draw cat box in green + label.
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

                # If we found a cat, trigger the action (but respect cooldown).
                if cat_found and (now - last_trigger_time) >= motion_cooldown_sec:
                    last_trigger_time = now

                    # Save a snapshot for proof
                    path = save_frame(frame)
                    print(f"[CAT DETECTED] saved: {path}")

                    # Lock workstation to stop the cat from typing chaos
                    lock_computer()

        # ==========================
        # UI Overlay / status
        # ==========================
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

        # Show the live window.
        cv2.imshow("Face + Cat Guard (press q to quit)", frame)

        # Quit cleanly when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ------------------------------------------
    # Cleanup: release camera + close windows
    # ------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()