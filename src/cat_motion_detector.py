import time
from datetime import datetime
import ctypes
import cv2
from ultralytics import YOLO


# -------------------------------------------------
# Utility function: save a snapshot when a cat
# is detected so we have proof of the event
# -------------------------------------------------
def save_frame(frame, folder="captures"):
    """
    Saves the current video frame to the captures folder.

    - Uses a timestamp so files never overwrite each other
    - Called only when a cat is confidently detected
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{folder}/cat_{ts}.jpg"

    # Write the image to disk
    cv2.imwrite(path, frame)
    return path


# -------------------------------------------------
# Utility function: safely lock the computer
# -------------------------------------------------
def lock_computer():
    """
    Locks the Windows workstation (same as pressing Win + L).

    Why this is safe:
    - Uses the built-in Windows lock screen
    - No keyboard drivers or inputs are disabled
    - Logging back in restores everything normally
    """
    print("[ACTION] Locking workstation")
    ctypes.windll.user32.LockWorkStation()


def main():
    # -------------------------------------------------
    # 1) Open the webcam
    # -------------------------------------------------
    # VideoCapture(0) usually refers to the built-in camera.
    # If this fails, trying 1 or 2 can help.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try VideoCapture(1) if needed.")

    # -------------------------------------------------
    # 2) Load YOLO model (cat detector)
    # -------------------------------------------------
    # YOLOv8 nano is lightweight and fast enough for CPU use.
    # This model will only run AFTER motion is detected.
    model = YOLO("yolov8n.pt")

    # -------------------------------------------------
    # 3) Motion detection configuration
    # -------------------------------------------------
    prev_gray = None  # stores the previous frame for comparison

    # Ignore very small movements (lighting flicker, noise)
    motion_area_threshold = 1500

    # Cooldown prevents repeated locks if the cat stays in frame
    motion_cooldown_sec = 20
    last_trigger_time = 0

    # -------------------------------------------------
    # 4) YOLO throttling
    # -------------------------------------------------
    # Even when motion is happening, YOLO is expensive.
    # This limits how often we run object detection.
    yolo_min_interval_sec = 0.5
    last_yolo_time = 0

    # -------------------------------------------------
    # Main loop: process frames until user quits
    # -------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =============================
        # Stage 1: Motion detection
        # =============================
        # Convert frame to grayscale and blur it
        # (makes motion detection more stable)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected = False

        # Only compare frames if we have a previous one
        if prev_gray is not None:
            # Difference between current frame and previous frame
            delta = cv2.absdiff(prev_gray, gray)

            # Convert difference image to black/white
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]

            # Fill in gaps so motion blobs are solid
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours (moving regions)
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                # Ignore tiny movements
                if cv2.contourArea(c) < motion_area_threshold:
                    continue

                motion_detected = True

                # Draw bounding box around motion
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2
                )

        # Store this frame for the next iteration
        prev_gray = gray

        # =============================
        # Stage 2: Cat detection (YOLO)
        # =============================
        cat_found = False

        # Only run YOLO if motion is happening
        if motion_detected:
            now = time.time()

            # Throttle YOLO calls
            if now - last_yolo_time >= yolo_min_interval_sec:
                last_yolo_time = now

                # Run YOLO on the current frame
                results = model.predict(frame, verbose=False)
                r0 = results[0]

                # Loop through detected objects
                if r0.boxes is not None and len(r0.boxes) > 0:
                    for box in r0.boxes:
                        cls_id = int(box.cls[0])   # class ID
                        conf = float(box.conf[0]) # confidence score

                        # COCO dataset class ID 15 = cat
                        if cls_id == 15 and conf >= 0.65:
                            cat_found = True

                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Draw cat box + label
                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )
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

                # If a cat was found and cooldown passed, trigger action
                if cat_found and (now - last_trigger_time) >= motion_cooldown_sec:
                    last_trigger_time = now

                    # Save proof image
                    path = save_frame(frame)
                    print(f"[CAT DETECTED] saved: {path}")

                    # Lock the computer
                    lock_computer()

        # =============================
        # UI overlay
        # =============================
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

        # Display the live video window
        cv2.imshow("Cat Motion Detector (press q to quit)", frame)

        # Exit cleanly when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()