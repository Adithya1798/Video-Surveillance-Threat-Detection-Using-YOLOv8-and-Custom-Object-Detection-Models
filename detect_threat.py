import os
import cv2
import winsound
import logging
from ultralytics import YOLO

# Initialize logging
log_file = "detection_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load trained YOLOv8 model (weapons-trained)
model = YOLO("Results\\detect\\train2\\weights\\best.pt")

# Define threat-related object classes
THREAT_CLASSES = ["gun", "knife", "weapon", "cash", "drugs"]

# Alarm file path
ALARM_SOUND_PATH = "alarm.wav"

def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    frame_threat_detected = False
    threat_objects_detected = set()

    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_name}")
        print(f"Error opening {video_name}")
        return

    logging.info(f"Processing {label}: {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection on frame
        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes.data.tolist():
                class_id = int(box[5])
                class_name = model.names[class_id]

                if class_name.lower() in THREAT_CLASSES:
                    threat_objects_detected.add(class_name.lower())
                    frame_threat_detected = True

    cap.release()

    if frame_threat_detected:
        # Trigger alarm
        try:
            winsound.PlaySound(ALARM_SOUND_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            logging.error(f"Alarm failed: {e}")

        # Log threat
        detected_list = ', '.join(threat_objects_detected)
        msg = f"{label} - {video_name} classified as a THREAT due to: {detected_list}"
        logging.warning(msg)
        print(msg)
    else:
        msg = f"{label} - {video_name} classified as NOT a threat."
        logging.info(msg)
        print(msg)

if __name__ == "__main__":
    # Define specific input videos
    video_threat = "Data\Video_NoThreat.mp4"
    video_not_threat = "Data\Video_WomenGun.mp4"

    # Process both videos
    process_video(video_threat, label="Video 1")
    process_video(video_not_threat, label="Video 2")

    print(f"\nDetection complete. Check '{log_file}' for details.")
