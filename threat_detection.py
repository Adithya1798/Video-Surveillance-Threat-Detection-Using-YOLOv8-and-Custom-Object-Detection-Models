import cv2
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 model
model = YOLO("C:\\AI_Files\\Results\\detect\\train2\\weights\\best.pt")

# Define threat objects
THREAT_OBJECTS = {"knife", "gun", "firearm", "weapon"}  # Modify based on YOLO's labels

def process_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    threat_detected = False
    detected_objects = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        frame_count += 1
        results = model(frame)  # Run YOLO detection

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Get class index
                label = model.names[cls]  # Get class label
                
                # Track detected objects
                detected_objects.add(label.lower())

                # Check if a threat object is detected with a person
                if label.lower() in THREAT_OBJECTS:
                    threat_detected = True

    cap.release()

    # Write classification results
    with open(output_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if threat_detected:
            f.write(f"[{timestamp}] {video_path} classified as THREAT due to {', '.join(detected_objects)}\n")
        else:
            f.write(f"[{timestamp}] {video_path} classified as NOT a THREAT\n")

    print(f"Processed {video_path}: {'THREAT' if threat_detected else 'NOT a THREAT'}")

# Run detection on two example videos
process_video("Data/Video_Gun_Man_Women.mp4", "Results/results_1.txt")
process_video("Data\Book_Obj.mp4", "Results/results_2.txt")