# scripts/analyze_confidence.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Folder setup ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "outputs", "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def analyze_confidence():
    # Check if logs folder exists
    if not os.path.exists(LOGS_DIR):
        print("⚠️ Logs folder not found! Run run_detection.py first.")
        return

    # Collect all CSV files from logs
    csv_files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("⚠️ No CSV logs found inside outputs/logs/")
        return

    results = []

    # Go through each CSV file and calculate average confidence
    for file in csv_files:
        file_path = os.path.join(LOGS_DIR, file)
        df = pd.read_csv(file_path)

        avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
        total_frames = df["frame"].nunique() if "frame" in df.columns else 0

        # Split filename into model and video name
        parts = file.replace(".csv", "").split("_")
        model = parts[0]
        video_name = "_".join(parts[1:])

        results.append({
            "Model": model,
            "Video": video_name,
            "Avg Confidence": round(avg_conf, 3),
            "Frames": total_frames
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    print("\n📊 Confidence Analysis Summary:\n")
    print(results_df.to_string(index=False))

    # --- Create bar chart ---
    plt.figure(figsize=(8, 5))
    for video_name, group in results_df.groupby("Video"):
        plt.bar(group["Model"], group["Avg Confidence"], label=video_name)

    plt.title("Average Confidence by Model and Video")
    plt.xlabel("Model Name")
    plt.ylabel("Average Confidence Score")
    plt.legend(title="Video")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save chart to folder
    plot_path = os.path.join(PLOTS_DIR, "confidence_summary.png")
    plt.savefig(plot_path)
    print(f"\n✅ Chart saved to: {plot_path}")

    # Show chart
    plt.show()

if __name__ == "__main__":
    analyze_confidence()
