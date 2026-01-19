# Camera Face & Cat Detection

This is a personal computer vision project built with Python that uses a webcam to detect faces, motion, and cats in real time.

If a cat is detected walking across the computer, the application automatically locks the keyboard by locking the Windows workstation.

This started as a fun idea and turned into a project I really enjoyed building, with a lot of help from ChatGPT along the way.

## What it does
- Opens a live webcam feed
- Detects faces in real time
- Uses motion detection to gate heavier processing
- Detects cats using a YOLO object detection model
- Saves a snapshot when a cat is detected
- Automatically locks the computer to prevent keyboard chaos

## Demo

### Face and cat detection running together
![Face and Cat Detection](images/face_and_cat_detection.png)

### Cat detected and trigger activated
![Cat Detected](images/cat_detection.jpg)

## Tech used
- Python
- OpenCV
- YOLO (Ultralytics)
- Haar Cascades for face detection

## Notes
- The virtual environment and model weights are intentionally not tracked.
- Captured images are ignored by git and saved locally at runtime.
- Detection thresholds and cooldowns can be adjusted in the script.

## How to run
```
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python src/face_and_cat_guard.py
```