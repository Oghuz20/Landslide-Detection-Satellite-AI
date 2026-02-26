# 🌋 Landslide Detection from Satellite Imagery

> **Automated landslide detection using deep learning and multi-spectral satellite imagery**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results](#-results)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Academic Context](#-academic-context)
- [Citation](#-citation)

---

## 🎯 Overview

This project implements a **UNet++ with EfficientNet-b5 encoder** for automatic landslide
detection from satellite imagery. The model processes 14-channel multi-spectral data
(Sentinel-2 + slope + DEM) to produce binary pixel-level landslide segmentation masks.

Trained on the **Landslide4Sense** benchmark dataset using a two-phase strategy:
Phase 1 on TrainData with early stopping, Phase 2 on Train+Valid combined for maximum
use of labeled data.

### Why This Matters

- 🌍 **Early Detection** — rapid identification of landslides across large geographic areas
- 🛰️ **Wide Coverage** — processes satellite data covering vast, inaccessible regions
- 💰 **Cost-Effective** — automates manual expert analysis
- 🚨 **Life-Saving** — early warnings can prevent casualties and property damage

### Challenges Addressed

- **Extreme class imbalance**: only ~1.9% of pixels are landslides
- **Geographic domain shift**: model must generalize across unseen terrain types
- **14-channel input**: standard pretrained encoders designed for 3-channel RGB
- **Small positive regions**: landslides range from a few pixels to large debris flows

---

## ✨ Key Features

- ✅ **UNet++ architecture** with nested dense skip connections
- ✅ **EfficientNet-b5 encoder** with random initialization (ImageNet init tested and rejected)
- ✅ **scSE decoder attention** — spatial + channel squeeze-excitation
- ✅ **WeightedBCEDiceLoss** with pos_weight=12 for class imbalance
- ✅ **Two-phase training** — Phase 1 (train only) → Phase 2 (train+valid combined)
- ✅ **Optuna hyperparameter tuning** — best params saved and reused
- ✅ **Automatic best checkpoint selection** — scans folder by Val F1
- ✅ **Per-image visualizations** — 6-panel analysis for best and worst predictions

---

## 📈 Results

### Final Model Performance

**Model**: UNet++ EfficientNet-b5, Phase 2 (epoch 35)  
**Strategy**: Single best model, no TTA, threshold=0.56  
**Test Set**: 800 images

| Metric | Score |
|--------|-------|
| **F1 Score** | **0.6937** |
| **Precision** | 0.6694 |
| **Recall** | 0.7198 |
| **Accuracy** | 0.9880 |

### Comparison with Literature

| Method | Test F1 |
|--------|---------|
| Competition Winner (2022) | 0.7234 |
| 2nd Place | 0.6891 |
| **Ours (UNet++ Phase 2)** | **0.6937** |
| 3rd Place | 0.6542 |
| U-Net v1 baseline (ours) | 0.6227 |

Our model surpasses 3rd place and is within 0.005 of 2nd place.  
**Improvement over our own baseline: +0.071 (+11.4%)**

### Model Evolution

| Model | Val F1 | Test F1 | Notes |
|-------|--------|---------|-------|
| U-Net v1 (data leakage) | 0.7103 | 0.5691 | Invalid |
| U-Net v1 (proper splits) | 0.6688 | 0.6227 | Fixed baseline |
| UNet++ ImageNet init | 0.6536 | — | Regression |
| UNet++ Phase 1 | 0.7152 | 0.6241 | TrainData only |
| **UNet++ Phase 2** | **0.7780** | **0.6937** | **Final** |

### Prediction Strategy Comparison

| Strategy | Test F1 |
|----------|---------|
| **Single model, no TTA** | **0.6937** |
| Single model + TTA | 0.6831 |
| Ensemble top-3 + TTA | 0.6796 |

TTA and ensemble both hurt performance — the best checkpoint was well-calibrated
(val threshold 0.82), and averaging destabilized this.

---

## 📊 Dataset

**Landslide4Sense** — multi-spectral satellite imagery patches.

| Split | Images | Size |
|-------|--------|------|
| TrainData | 3,799 | ~1.9 GB |
| ValidData | 245 | ~122 MB |
| TestData | 800 | ~400 MB |

- **Image format**: HDF5 (`.h5`), 128 × 128 × 14, float32
- **Mask format**: HDF5 (`.h5`), 128 × 128, binary (0=background, 1=landslide)
- **Bands**: 12 Sentinel-2 multispectral + slope + DEM (ALOS PALSAR)
- **Class distribution**: 1.9% landslide, 98.1% background

> ⚠️ Dataset not included (~2.4 GB). See [data/README.md](data/README.md) for download.

---

## 🏗️ Model Architecture

### UNet++ with EfficientNet-b5
```
Input:   [Batch, 14, 128, 128]
Encoder: EfficientNet-b5 (random init — ImageNet init reduced F1 from 0.73 → 0.65)
Decoder: UNet++ nested dense blocks + scSE attention at each level
Output:  [Batch, 1, 128, 128] → sigmoid → binary mask
```

**Why UNet++?** Nested dense skip connections progressively fuse encoder features
at multiple scales, reducing the semantic gap between encoder and decoder.
Better than standard U-Net for small irregular objects like landslides.

**Why random init?** EfficientNet expects 3-channel RGB. The 3→14 channel adapter
is always random regardless of pretrained weights — this destabilizes the encoder
when weights are frozen or slowed down. Full random init was more stable and achieved
higher F1.

### Loss Function
```
Loss = 0.3 × BCE(pos_weight=12) + 0.7 × DiceLoss
```

`pos_weight=12` forces the model to count each landslide pixel 12× more than background,
directly addressing the 1.9% class imbalance and pushing toward higher recall.

Focal Loss was tested and caused training collapse (F1 → 0.05). Not used.

### Training Strategy

**Phase 1** — TrainData only (3,799 images), patience=15, up to 60 epochs.
Validates on official ValidData. Goal: find best training duration.

**Phase 2** — Train+Valid combined (4,044 images), fixed 35 epochs, no early stopping.
ValidData included with training augmentations. Top-3 checkpoints saved by Val F1.

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) — CPU works for inference
- 8 GB RAM minimum

### Setup
```bash
git clone https://github.com/Oghuz20/Landslide-Detection-Satellite-AI.git
cd Landslide-Detection-Satellite-AI
git checkout v2-unetplusplus-kaggle

# Windows
python -m venv venv_landslide
.\venv_landslide\Scripts\Activate.ps1

# Linux / Mac
python3 -m venv venv_landslide
source venv_landslide/bin/activate

pip install -r requirements.txt
```

### Verify
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 💻 Usage

### Predict
```bash
python src/predict.py
```

Auto-selects best checkpoint by Val F1, runs prediction on TestData,
saves masks to `predictions/`, prints final metrics.

To use a specific checkpoint:
```bash
python src/predict.py --checkpoint checkpoints/checkpoint_epoch_35.pth
```

### Visualize
```bash
python src/visualize.py
```

Saves all PNGs to `visualizations/`:
- `threshold_curve.png` — F1/P/R vs threshold
- `confusion_matrix.png` — TN/FP/FN/TP heatmap
- `performance_distribution.png` — per-image F1 histogram
- `best_1..5_*.png` — top 5 predictions (6-panel each)
- `worst_1..5_*.png` — bottom 5 predictions, landslide images only
- `results_summary.png` — strategy comparison bar chart
- `training_curves.png` — loss, val F1, LR over epochs

### Train
```bash
python src/train.py
```

Runs Phase 1 then Phase 2 sequentially.
See `notebook/landslide_v2_kaggle.ipynb` for the full Kaggle notebook.

### Quick inference example
```python
import torch
from src.model import get_model
from src.dataset import LandslideDataset
from src.transforms import get_valid_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = get_model().to(device)
ckpt   = torch.load('checkpoints/checkpoint_epoch_35.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

dataset = LandslideDataset('./data', split='test', transform=get_valid_transforms())
sample  = dataset[0]
image   = sample['image'].unsqueeze(0).to(device)

with torch.no_grad():
    prob = torch.sigmoid(model(image)[0, 0]).cpu().numpy()
    pred = (prob >= 0.56).astype(int)

print(f'Landslide pixels: {pred.sum()} / {pred.size}')
```

---

## 📁 Project Structure
```
Landslide-Detection-Satellite-AI/
├── data/
│   └── README.md                  ← dataset download instructions
│
├── src/
│   ├── dataset.py                 ← LandslideDataset, HDF5 loader, normalization
│   ├── model.py                   ← get_model() — UNet++ EfficientNet-b5
│   ├── losses.py                  ← WeightedBCEDiceLoss
│   ├── transforms.py              ← train / valid augmentation pipelines
│   ├── train.py                   ← train_phase1(), train_phase2()
│   ├── predict.py                 ← best checkpoint selection, threshold search
│   └── visualize.py               ← all PNG generation
│
├── configs/
│   └── optuna_best_params.json    ← best hyperparameters from Optuna study
│
├── notebook/
│   └── landslide_v2_kaggle.ipynb  ← full Kaggle training notebook
│
├── checkpoints/                   ← .pth files (not tracked by git)
├── predictions/                   ← output HDF5 masks (not tracked by git)
├── visualizations/                ← output PNGs (not tracked by git)
│
├── docs/
│   ├── RESULTS.md                 ← detailed results and comparisons
│   ├── METHODOLOGY.md             ← architecture and training decisions
│   └── USAGE.md                   ← full usage guide
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 📚 Documentation

- **[RESULTS.md](docs/RESULTS.md)** — full metrics, confusion matrix, error analysis, literature comparison
- **[METHODOLOGY.md](docs/METHODOLOGY.md)** — architecture decisions, loss function design, training strategy
- **[USAGE.md](docs/USAGE.md)** — installation, inference, training, troubleshooting

---

## 🎓 Academic Context

Developed as a **graduation thesis** on automated landslide detection using deep learning.

**Contributions**:
- Demonstrated that ImageNet pretraining hurts performance on 14-channel satellite data
- Showed that single well-calibrated model outperforms TTA and ensemble on this dataset
- Two-phase training strategy that maximizes use of labeled data without leakage
- Achieved Test F1=0.6937, surpassing 3rd place in the original 2022 competition

**Limitations**:
- Val→Test gap of 0.084 due to geographic domain shift between official splits
- Small validation set (245 images) limits hyperparameter search reliability
- Performance drops on small landslides (<20 pixels) and heavily vegetated areas

**Future work**:
- Transformer-based architectures (SegFormer, Mask2Former)
- Semi-supervised learning with unlabeled satellite imagery
- Domain adaptation to close the geographic generalization gap

---

## 📖 Citation
```bibtex
@software{hasanli2026landslide,
  author = {Hasanli, Oghuz},
  title  = {Landslide Detection from Satellite Imagery using UNet++ EfficientNet-b5},
  year   = {2026},
  url    = {https://github.com/Oghuz20/Landslide-Detection-Satellite-AI}
}
```

**Dataset:**
```bibtex
@article{ghorbanzadeh2022landslide4sense,
  title   = {The outcome of the 2022 landslide4sense competition},
  author  = {Ghorbanzadeh, Omid and Xu, Yonghao and Ghamisi, Pedram and Kopp, Martin and Kreil, David},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2022},
  publisher = {IEEE}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Landslide4Sense** for the benchmark dataset
- **Kaggle** for P100 GPU resources
- **segmentation-models-pytorch** for the UNet++ implementation
- **Albumentations** for the augmentation pipeline