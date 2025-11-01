import os
import pandas as pd

# Set up main project paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "outputs", "annotated_frames")
GT_DIR = os.path.join(BASE_DIR, "outputs", "gt_samples")

# Make sure the ground truth folder exists
os.makedirs(GT_DIR, exist_ok=True)

def create_ground_truth(video_name):
    """Make a CSV template for checking detection results manually."""
    folder_path = os.path.join(FRAMES_DIR, video_name)
    if not os.path.exists(folder_path):
        print(f"Skipping {video_name} — no frames found.")
        return

    # Get all frame images
    frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    if not frames:
        print(f"No .jpg files found in {video_name}.")
        return

    # Create rows for CSV
    records = []
    for frame_file in frames:
        frame_num = int(frame_file.split("_")[-1].split(".")[0])
        records.append({
            "frame": frame_num,
            "image_path": os.path.join(folder_path, frame_file),
            "true_positive": "",
            "false_positive": "",
            "missed_detection": ""
        })

    # Save the CSV file
    output_path = os.path.join(GT_DIR, f"gt_{video_name}.csv")
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Ground truth file created: {output_path}")

def main():
    # Find all folders inside annotated_frames
    videos = [v for v in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, v))]
    if not videos:
        print("No folders found in annotated_frames/")
        return

    for video in videos:
        create_ground_truth(video)

if __name__ == "__main__":
    main()
