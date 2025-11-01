import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Folder setup
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GT_DIR = os.path.join(BASE_DIR, "outputs", "gt_samples")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_model(gt_file):
    """Read ground truth CSV and calculate model metrics."""
    df = pd.read_csv(gt_file)

    # Replace blank cells with 0 and convert to integers
    df = df.fillna(0)
    df["true_positive"] = df["true_positive"].astype(int)
    df["false_positive"] = df["false_positive"].astype(int)
    df["missed_detection"] = df["missed_detection"].astype(int)

    # Define true values and predictions
    y_true = (df["true_positive"] | df["missed_detection"]).astype(int)
    y_pred = (df["true_positive"] | df["false_positive"]).astype(int)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy

def main():
    """Compare performance of each model and save results."""
    results = []
    print("\nModel Evaluation Summary:\n")

    # Go through each ground truth CSV file
    for file in os.listdir(GT_DIR):
        if not file.endswith(".csv"):
            continue

        gt_path = os.path.join(GT_DIR, file)
        precision, recall, f1, accuracy = evaluate_model(gt_path)

        # Identify model and video names from file name
        model_name = "Caffe" if "caffe" in file else "Haar"
        video_name = "Video Benchmark" if "benchmark" in file else "Video Stress Test"

        results.append([model_name, video_name, precision, recall, f1, accuracy])
        print(f"{model_name:<8} | {video_name:<18} | "
              f"Precision: {precision:.2f} | Recall: {recall:.2f} | "
              f"F1: {f1:.2f} | Accuracy: {accuracy:.2f}")

    # Create summary table
    df = pd.DataFrame(results, columns=["Model", "Video", "Precision", "Recall", "F1", "Accuracy"])
    summary_path = os.path.join(PLOTS_DIR, "evaluation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nEvaluation summary saved to: {summary_path}")

    # Plot model performance for each video
    plt.figure(figsize=(8, 5))
    df.set_index("Video")[["Precision", "Recall", "F1", "Accuracy"]].plot(
        kind="bar", color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"], edgecolor="black"
    )

    plt.title("Face Detection Model Comparison", fontsize=14)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    chart_path = os.path.join(PLOTS_DIR, "model_comparison_bar_chart.png")
    plt.savefig(chart_path)
    print(f"Chart saved to: {chart_path}")
    plt.show()

if __name__ == "__main__":
    main()
