# scripts/run_detection.py
import cv2
import os
import time
import pandas as pd

# Set up important folder paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_LOGS = os.path.join(BASE_DIR, "outputs", "logs")
OUTPUT_VIDEOS = os.path.join(BASE_DIR, "outputs", "annotated_videos")

# Create output folders if they don’t exist
os.makedirs(OUTPUT_LOGS, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS, exist_ok=True)

# Paths to model files
CAFFE_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFE_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
HAAR_MODEL = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")


# Function to load the Caffe deep learning model
def load_caffe():
    print("Loading Caffe SSD model...")
    return cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)


# Function to load the Haar Cascade model
def load_haar():
    print("Loading Haar Cascade model...")
    return cv2.CascadeClassifier(HAAR_MODEL)


# Run face detection on a given video with a specific model
def run_detection(video_path, model_name):
    print(f"\nProcessing: {os.path.basename(video_path)} using {model_name} model")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    output_path = os.path.join(OUTPUT_VIDEOS, f"{model_name}_{os.path.basename(video_path)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

    frame_count = 0
    log_data = []
    start_time = time.time()

    # Choose model
    if model_name == "caffe":
        net = load_caffe()
    elif model_name == "haar":
        net = load_haar()
    else:
        print("Invalid model name.")
        return

    # Go through all frames in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Use Caffe model
        if model_name == "caffe":
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * [width, height, width, height]
                    (x1, y1, x2, y2) = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    log_data.append([frame_count, confidence])

        # Use Haar model
        elif model_name == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = net.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                log_data.append([frame_count, 1.0])  # Haar doesn’t provide confidence

        out.write(frame)

    # Calculate FPS and save log
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    print(f"Finished {model_name}: {frame_count} frames at {avg_fps:.2f} FPS")

    df = pd.DataFrame(log_data, columns=["frame", "confidence"])
    log_path = os.path.join(OUTPUT_LOGS, f"{model_name}_{os.path.basename(video_path)}.csv")
    df.to_csv(log_path, index=False)
    print(f"Log saved: {log_path}")

    cap.release()
    out.release()


# Main function to run detection on all .mp4 videos
def main():
    videos = [v for v in os.listdir(VIDEOS_DIR) if v.endswith(".mp4")]
    if not videos:
        print("No .mp4 videos found in the videos folder.")
        return

    for video in videos:
        path = os.path.join(VIDEOS_DIR, video)
        run_detection(path, "caffe")
        run_detection(path, "haar")


if __name__ == "__main__":
    main()
