
# 🧪 Pseudo-Labeled Tree Species Detection – Evaluation Pipeline

This repository contains an evaluation script for a **Faster R-CNN** model trained on pseudo-labeled aerial forest imagery. The script runs inference, filters predicted boxes, and evaluates classification metrics against ground truth.

---

## 📂 Project Structure

```bash
root_dir/
├── val_labels.csv                     # Ground-truth labels (validation set)
├── checkpoints/
│   └── COMBINE_RESULT/
│       ├── predicted_boxes/          # Raw predictions for each image (CSV)
│       ├── filtered_predicted_dots/  # Filtered predictions using IoU matching
│       ├── merged_predictions.csv    # Aggregated prediction results
│       └── classification_metrics.csv # Precision, Recall, F1 Score
````

---

## 🚀 Pipeline Overview

### 1️⃣ Dataset Loader

A custom PyTorch `Dataset` class (`trDataset`) loads validation images and filenames using `val_labels.csv`.

```python
Image.open(image_path).convert('RGB') → torchvision.transforms.functional.to_tensor(img)
```

### 2️⃣ Model Loading and Inference

* Loads a Faster R-CNN model with a ResNet-50 FPN backbone.
* Custom head set for 5 tree species.
* Weights loaded from:

  ```
  checkpoints/fasterrcnn_selftrained_final.pth
  ```
* Predictions saved per image in `.csv` format under `predicted_boxes/`.

### 3️⃣ IoU-Based Box Filtering

* Matches each predicted bounding box with ground-truth boxes using **IoU > 0.5**.
* Keeps only the best-matching box per GT object (based on confidence score).
* Saves filtered boxes to `filtered_predicted_dots/`.

### 4️⃣ CSV Merging

* Combines all filtered box CSVs into one file: `merged_predictions.csv`.
* Removes `.0` suffixes in filenames and strips out confidence scores.
* Used for classification metric calculation.

### 5️⃣ Metric Evaluation and Confusion Matrix

* Compares predicted vs. ground-truth labels.
* Computes:

  * **Precision**
  * **Recall**
  * **F1 Score**
* Displays a 5×5 confusion matrix using Seaborn.
* Saves metrics to `classification_metrics.csv`.

---

## 📈 Sample Output (Confusion Matrix)

A heatmap is generated showing true vs. predicted classes:

```
          Predicted
          0  1  2  3  4
True  0  ██ ██ ██ ██ ██
       1  ██ ██ ██ ██ ██
       2  ██ ██ ██ ██ ██
       3  ██ ██ ██ ██ ██
       4  ██ ██ ██ ██ ██
```

---

## 📊 Example Metric Output

```
Precision:     0.8421
Recall:        0.8157
F1 Score:      0.8273
```

Saved to:

```text
classification_metrics.csv
```

---

## 🛠 Requirements

* Python 3.8+
* PyTorch ≥ 1.10
* torchvision
* pandas
* PIL (Pillow)
* seaborn
* matplotlib
* scikit-learn
* tqdm

---

## 📌 Notes

* The model detects and classifies trees into **5 categories**.
* The evaluation pipeline uses a **single image batch (batch size = 1)**.
* Ground-truth and predictions are matched based on filename.

---

## 📧 Citation

If you use this pipeline in your research, please cite or refer to the associated model and training script used to generate `fasterrcnn_selftrained_final.pth`.

