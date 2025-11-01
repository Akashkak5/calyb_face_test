import os
import pandas as pd
import matplotlib.pyplot as plt

# Base folders
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GT_DIR = os.path.join(BASE_DIR, "outputs", "gt_samples")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def visualize_performance():
    # Read all CSV files from ground truth folder
    csv_files = [f for f in os.listdir(GT_DIR) if f.endswith(".csv")]

    for file in csv_files:
        df = pd.read_csv(os.path.join(GT_DIR, file)).fillna(0)

        # Count total detections
        tp = int(df["true_positive"].sum())
        fp = int(df["false_positive"].sum())
        md = int(df["missed_detection"].sum())

        total = tp + fp + md
        if total == 0:
            print(f"Skipping {file} - no detections found.")
            continue

        labels = ["True Positive", "False Positive", "Missed Detection"]
        values = [tp, fp, md]
        colors = ["#4CAF50", "#FFC107", "#F44336"]

        # Create pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(values,
                labels=[f"{l} ({v})" for l, v in zip(labels, values)],
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                textprops={"fontsize": 11})
        plt.title(file.replace(".csv", ""), fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()

        # Save chart
        save_path = os.path.join(PLOTS_DIR, f"{file.replace('.csv', '')}_piechart.png")
        plt.savefig(save_path, dpi=220)
        plt.close()
        print(f"Pie chart saved: {save_path}")

if __name__ == "__main__":
    visualize_performance()
