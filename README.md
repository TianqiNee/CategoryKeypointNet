# CategoryKeypointNet

A category-guided keypoint detection framework for industrial binary images. This project introduces a semantic embedding module and hybrid loss design to achieve precise and stable localization under severe foreground-background imbalance.

## 🧠 Overview

Industrial keypoint detection poses unique challenges due to:
- Structural fragmentation caused by binarization
- Extreme foreground-background imbalance


This repository implements a U-Net-based model enhanced with:
- **Category-guided semantic embedding**
- **Hybrid focal and cross-entropy losses**
- **Morphological dilation preprocessing**
- **Custom evaluation metrics**

## 📁 Project Structure

```
CategoryKeypointNet/
│
├── train.py               # Training pipeline with hybrid imbalance loss supervision
├── test.py                # Evaluation script using matching accuracy and localization error
├── model.py               # Model definition (CategoryKeypointNet and UNet)
├── module.py              # U-Net building blocks (DoubleConv, Down, Up, OutConv)
├── loss.py                # Balanced focal loss, cross-entropy loss
├── dataset.py             # Custom PyTorch Dataset for heatmaps and classification masks
├── evaluation.py          # Matching accuracy, localization error calculation
├── dilate.py              # Morphological dilation preprocessing
├── data/                  # Training/test images and labels
└── logs/                  # TensorBoard logs
```

## 🏗️ Model Architecture

The network consists of:
- A U-Net backbone for multi-scale feature extraction
- A pixel-wise classification head to predict category maps
- A learnable embedding layer to inject category context
- A dual-branch keypoint head:
  - **Heatmap score head** (sigmoid)
  - **Heatmap class head** (softmax)

Final keypoint heatmap:
```
H = HeatmapScore ⊙ argmax(HeatmapClass)
```

## 🧪 Dataset Preparation

Directory structure for training/test data:

```
data/
└── square/
    ├── train_data/             # Input grayscale images
    ├── train_label/
    │   ├── cls/                # Pickle files storing classification labels
    │   └── heatmap/            # Pickle files storing 2D keypoint centers
    ├── test_data/
    └── test_label/
```

## 🚀 Training

Run the following command to train the model on the `square` dataset:

```bash
python train.py
```

Training details:
- Optimizer: Adam
- Initial LR: 0.005 with manual scheduling
- Batch size: 8
- Epochs: 30
- Loss: Focal heatmap loss (custom-designed for sparse structure)

## 📈 Evaluation

To evaluate trained models:

```bash
python test.py
```

Metrics:
- **Matching Accuracy**: Percentage of predicted points within a threshold
- **Localization Error**: Mean distance of matched keypoints

## 🔍 Preprocessing with Morphological Dilation

To apply dilation on raw images:

```bash
python dilate.py
```

This improves structural continuity in binarized industrial scans.

## 🧩 Custom Loss Function Highlights

- **Balanced Focal Loss**: Region-stratified, normalizing sparse positives
- **Cross-Entropy Loss**: Supervises classification masks


## 📬 Contact

For questions or collaborations, contact:

- Tianqi Ni (213223763@seu.edu.cn)

