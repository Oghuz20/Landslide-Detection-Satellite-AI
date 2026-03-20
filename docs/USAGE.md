# Usage Guide

Complete guide for installation, inference, training, and visualization.

---

## Table of Contents

- [Installation](#installation)
- [Inference](#inference)
- [Visualization](#visualization)
- [Training](#training)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended) — CPU works for inference but is slow
- 8 GB RAM minimum (16 GB recommended for training)
- 3 GB storage for code and models (add 2.4 GB for dataset)

### Setup

```bash
git clone https://github.com/Oghuz20/Landslide-Detection-Satellite-AI.git
cd Landslide-Detection-Satellite-AI
```

**Windows:**
```bash
python -m venv venv_landslide
.\venv_landslide\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / Mac:**
```bash
python3 -m venv venv_landslide
source venv_landslide/bin/activate
pip install -r requirements.txt
```

### Verify

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Download Dataset

See [data/README.md](../data/README.md) for dataset download instructions.

Place data in:
```
data/
├── TrainData/
├── ValidData/
└── TestData/
```

---

## Inference

### Run Prediction

```bash
python src/predict.py
```

What it does:
1. Scans `checkpoints/` and selects the checkpoint with highest Val F1
2. Runs single forward pass on all 800 TestData images (no TTA)
3. Searches for best threshold in steps of 0.02
4. Saves predicted masks to `predictions/`
5. Prints F1, Precision, Recall, Confusion Matrix

To use a specific checkpoint:
```bash
python src/predict.py --checkpoint checkpoints/checkpoint_epoch_35.pth
```

**Output format** — each `predictions/mask_XXXXX.h5` contains:
- `mask` — uint8 binary mask at best threshold
- `prob` — float32 raw sigmoid probability map

### Note on Ensemble and TTA

Ensemble top-3 + TTA was tested but did not outperform the single best model in this experiment. The best checkpoint (Val threshold=0.82) was well-calibrated — averaging with augmented copies destabilized this and required a much lower threshold (0.26), increasing false positives. See [RESULTS.md](RESULTS.md) for the full comparison.

### Quick Inference Example

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

## Visualization

### Generate All Visualizations

```bash
python src/visualize.py
```

Reads from `predictions/`, saves all PNGs to `visualizations/`:

| File | Description |
|------|-------------|
| `training_curves.png` | Loss, Val F1, LR over epochs |
| `threshold_curve.png` | F1/Precision/Recall vs threshold |
| `confusion_matrix.png` | TN/FP/FN/TP heatmap |
| `performance_distribution.png` | Per-image F1 histogram |
| `best_1..5_*.png` | Top 5 highest F1 images, 6-panel each |
| `worst_1..5_*.png` | Bottom 5 lowest F1, landslide images only |
| `results_summary.png` | Strategy comparison bar chart |

**Note:** `worst_*.png` excludes images with no landslide pixels. F1=0 on empty images is expected behavior — not a model error.

Each best/worst PNG is a 6-panel figure showing:
- Satellite image (RGB composite from bands 3, 2, 1)
- Ground truth mask
- Model prediction mask
- Overlay (green=GT, red=prediction)
- Error analysis (green=TP, red=FP, yellow=FN)
- Image statistics (pixel counts, F1, precision, recall)

---

## Training

### Run Training

```bash
python src/train.py
```

Runs Phase 1 then Phase 2 sequentially. Hyperparameters are loaded from `configs/optuna_best_params.json` if present, otherwise safe defaults are used.

**Phase 1 — TrainData only, early stopping (patience=15):**
Identifies the best training duration by validating on official ValidData.

**Phase 2 — Train+Valid combined, fixed 35 epochs:**
Retrains from scratch with more data. Top-3 checkpoints saved by Val F1. No early stopping.

### Default Hyperparameters

```python
lr           = 0.0002
weight_decay = 0.00005
dice_w       = 0.7
pos_weight   = 12.0
batch_size   = 16
```

### Kaggle Notebook

See `notebook/landslide-v2-kaggle.ipynb` for the complete Kaggle training notebook with all cells in order. Designed for Kaggle P100 GPU — ~45 min per phase.

### Resume from Checkpoint

To skip Phase 1 and run only Phase 2 (for example after a session loss):

```python
from src.train import train_phase2
history2, top3 = train_phase2('data', 'checkpoints')
```

---

## Troubleshooting

**Out of GPU memory:**
Reduce batch size in `train.py` — change `BATCH_SIZE = 8`.

**No .h5 files found:**
Verify data folder structure matches `data/README.md`. Files must be inside `img/` and `mask/` subfolders.

**No checkpoints found:**
Run training first, or download checkpoints and place in `checkpoints/` without renaming them.

**Slow data loading on Windows:**
Set `num_workers=0` in DataLoader calls inside `src/predict.py` and `src/train.py`.

**Import errors:**
```bash
pip install albumentations h5py segmentation-models-pytorch tqdm --upgrade
```

**Predictions all zero:**
Lower the threshold. The model may need threshold=0.3 or lower on CPU due to slightly different floating point behavior. Run `src/predict.py` which searches automatically.

**FileNotFoundError for mask file:**
Ensure mask files follow the naming convention: `image_00001.h5` corresponds to `mask_00001.h5` in the mask subfolder.