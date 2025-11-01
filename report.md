                            Face Detection Model Evaluation


Project Overview:
  
In this project, two different videos—a difficult one (Video Stress Test) and a normal one (VideoBenchmark)—are used to compare the performance of two face detection models, Caffe SSD and Haar Cascade.
The main goal is to measure and visualize the accuracy of face detection for each model under different conditions.

Experimental Setup:

1.Models Used
    Caffe SSD (Deep Learning-based) — a modern detector trained with deep neural networks.
    Haar Cascade (Classical Approach) — a traditional model that uses feature-based detection.

2.Datasets
    video_benchmark.mp4 — a regular, well-lit video.
    video_stress_test.mp4 — a complex video with varied lighting and motion.

3.Tools and Libraries:
    numpy
    opencv-contrib-python
    matplotlib
    pandas
    tqdm
    requests
    shapely
    scikit-learn

4.Folder structure:

project image/
├── .venv/
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── haarcascade_frontalface_default.xml
├── videos/
│   ├── video_benchmark.mp4
│   └── video_stress_test.mp4
├── scripts/
│   ├── download_models.py
│   ├── run_detection.py
│   ├── analyze_confidence.py
│   ├── extract_sample_frames.py
│   ├── create_ground_truth_template.py
│   ├── evaluate_detections.py
│   └── visualize_model_performance.py
├── outputs/
│   ├── logs/
│   ├── annotated_frames/
│   │   ├── caffe_video_benchmark/
│   │   ├── caffe_video_stress_test/
│   │   ├── haar_video_benchmark/
│   │   └── haar_video_stress_test/
│   ├── annotated_videos/
│   ├── gt_samples/
│   │   ├── gt_caffe_video_benchmark.csv
│   │   ├── gt_caffe_video_stress_test.csv
│   │   ├── gt_haar_video_benchmark.csv
│   │   └── gt_haar_video_stress_test.csv
│   └── plots/
│       ├── model_comparison_bar_chart.png
│       ├── gt_caffe_video_benchmark_piechart.png
│       ├── gt_caffe_video_stress_test_piechart.png
│       ├── gt_haar_video_benchmark_piechart.png
│       ├── gt_haar_video_stress_test_piechart.png
│       └── evaluation_summary.csv
├── report.md
└── requirements.txt


5.Summary:
    1.Downloaded models using download_models.py
    2.Ran detections on videos with run_detection.py
    3.Extracted sample frames with extract_sample_frames.py
    4.Created manual ground truth templates with create_ground_truth_template.py
    5.Evaluated detection metrics using evaluate_detections.py
    6.Visualized confidence and performance with analyze_confidence.py and       visualize_model_performance.py

Evaluation Metrics:

 Metric             	Description
Precision	    Ratio of correct detections among all detections made
Recall	        Ratio of actual faces correctly detected
F1-Score	    Harmonic mean of precision and recall
Accuracy	    Overall correctness of model predictions


Results:

| Model | Video             | Precision | Recall | F1-Score | Accuracy |
| :---- | :---------------- | :-------- | :----- | :------- | :------- |
| Caffe | Video Benchmark   | 0.86      | 0.86   | 0.86     | 0.75     |
| Caffe | Video Stress Test | 0.78      | 0.78   | 0.78     | 0.64     |
| Haar  | Video Benchmark   | 0.86      | 0.86   | 0.86     | 0.75     |
| Haar  | Video Stress Test | 0.00      | 0.00   | 0.00     | 1.00     |


Graphs and Visualizations:

1.Model Comparison Chart: A bar chart comparing Precision, Recall, F1, and Accuracy for all models and videos.
File: outputs/plots/model_comparison_bar_chart.png

2.Confidence Analysis: Confidence trends across different models and frames were visualized from log files.
File: outputs/plots/confidence_chart.pngConfidence Analysis

3.Ground Truth Pie Charts: Individual performance breakdowns for each video.
Files:
gt_caffe_video_benchmark_piechart.png
gt_caffe_video_stress_test_piechart.png
gt_haar_video_benchmark_piechart.png
gt_haar_video_stress_test_piechart.png


Observations:
   Throughout both videos, the Caffe SSD model consistently outperformed the others, retaining good recall and precision.The Haar Cascade model did well in simpler settings but poorly in complex lighting (stress test).For real-world video face detection tasks, deep learning models such as Caffe are more dependable.
   Although faster, Haar cascades are not appropriate for dynamic or low-quality video input.

Conclusion:

  The experiment demonstrates how deep learning-based face detection techniques differ from traditional approaches.
  Despite the fact that Haar cascades are simple and efficient, the Caffe SSD model offers more accuracy and resilience in a range of video scenarios.



  