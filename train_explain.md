# Repeated Training and Evaluation Pipeline using Faster R-CNN

This script implements a repeated training pipeline for object detection using **Faster R-CNN** on a tree species dataset. The goal is to evaluate the stability and consistency of model performance across multiple independent runs.

---

## 🎯 Purpose

To train the same model architecture multiple times from scratch, allowing comparison of performance metrics across runs. This setup helps analyze the robustness of the training process and the effects of random initialization and data shuffling.

---

## 🔁 Overview of Training Loop

- The script performs **10 independent training runs**.
- Each run trains a new Faster R-CNN model using the same training and validation datasets.
- The best-performing model (based on validation loss) is saved separately for each run.
- After each training session, a test-time evaluation is triggered by calling a separate script: `test_evaluate.py`.

---

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| **Model** | `Faster R-CNN with ResNet-50 FPN` backbone |
| **Loss Function** | Multi-component object detection loss (inherent in torchvision's Faster R-CNN) |
| **Optimizer** | Stochastic Gradient Descent (SGD) with momentum and weight decay |
| **Early Stopping** | Stops training after 3 epochs without improvement in validation loss |
| **Repeatability** | Training is repeated 10 times (`run_1` to `run_10`) to assess model stability |
| **Checkpointing** | Each run saves the best model as `checkpoints/run_X/best_model.pth` |

---

## 📁 Input Requirements

- `train_labels.csv`: Ground-truth annotations for training images
- `val_labels.csv`: Ground-truth annotations for validation images
- `images_rename/`: Directory containing all input images (referenced by filename)
- `test_evaluate.py`: Script to evaluate each saved model on a test set

---

## 📤 Output

- **Model Checkpoints**: Saved in `checkpoints/run_X/` (10 folders for 10 runs)
- **Printed Logs**: Training loss and validation loss per epoch
- **Test Evaluation**: Each best model is automatically passed to `test_evaluate.py` for performance analysis

---

## 🔧 Configuration

- Batch size: 1
- Max epochs per run: 10
- Early stopping patience: 3 epochs
- Number of classes: Automatically inferred from training labels

---

## 📈 Use Case

This script is ideal for:
- Evaluating model generalizability across different random seeds
- Selecting the most consistent or highest-performing model
- Feeding all models into a later ensemble or comparative analysis

---

## 📝 Note

Make sure `test_evaluate.py` is properly configured to load the model checkpoints and evaluate them accordingly. The dataset CSV files must include the following columns:  
`filename`, `xmin`, `ymin`, `xmax`, `ymax`, `class`

---

## 📄 License

MIT License
