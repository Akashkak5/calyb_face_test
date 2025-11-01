# scripts/extract_sample_frames.py
import cv2
import os

# --- Set up folders ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
VIDEOS_FOLDER = os.path.join(BASE_DIR, "outputs", "annotated_videos")
FRAMES_FOLDER = os.path.join(BASE_DIR, "outputs", "annotated_frames")
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Extract one frame every 100 frames (you can adjust this)
FRAME_INTERVAL = 100

def extract_frames(video_path):
    """Takes a video and saves sample frames for review."""
    video_name = os.path.basename(video_path).replace(".mp4", "")
    save_path = os.path.join(FRAMES_FOLDER, video_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return

    frame_number = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every Nth frame
        if frame_number % FRAME_INTERVAL == 0:
            filename = f"frame_{frame_number:04d}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)
            saved_frames += 1

        frame_number += 1

    cap.release()
    print(f"✅ {saved_frames} frames extracted from {video_name}")

def main():
    videos = [v for v in os.listdir(VIDEOS_FOLDER) if v.endswith(".mp4")]
    if not videos:
        print("⚠️ No annotated videos found in outputs/annotated_videos/")
        return

    for video in videos:
        extract_frames(os.path.join(VIDEOS_FOLDER, video))

if __name__ == "__main__":
    main()
