# 🔬 Training Methodology

> Detailed explanation of the model training process, architecture decisions, and optimization strategy

---

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Loss Function Design](#loss-function-design)
- [Training Configuration](#training-configuration)
- [Optimization Strategy](#optimization-strategy)
- [Evaluation Protocol](#evaluation-protocol)
- [Challenges & Solutions](#challenges--solutions)

---

## Overview

This document details the complete training methodology for the landslide detection model, from data preparation to final evaluation.

### Key Decisions Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Architecture** | Attention U-Net | Better feature selection than standard U-Net |
| **Loss Function** | 0.4×Focal + 0.6×Tversky | Handles extreme class imbalance |
| **Optimizer** | AdamW | Better weight decay than Adam |
| **Learning Rate** | 1e-4 with ReduceLROnPlateau | Stable convergence |
| **Batch Size** | 16 (Colab T4) | Balance between speed and memory |
| **Augmentation** | Extensive | Prevents overfitting |
| **Regularization** | Dropout (0.3) + Weight Decay (2e-4) | Improves generalization |

---

## Data Preparation

### Dataset Splits

**Final Configuration** (proper methodology):

```
TrainData:  3,799 images (100% of original training set)
ValidData:    245 images (official validation set)
TestData:     800 images (official test set)
```

**Important**: No overlap between splits. ValidData comes from different geographic regions than TrainData.

### Previous Mistake (Fixed)

❌ **Old (Data Leakage)**:
```
TrainSplit:  3,419 images (90% of TrainData)
ValidSplit:    380 images (10% of TrainData) ← LEAKED
TestData:      800 images
```

This caused inflated validation F1 (0.71) vs. true test F1 (0.57).

### Data Loading

**Implementation**: `src/dataset.py`

```python
class LandslideDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        # split: 'train', 'valid', or 'test'
        # Maps to TrainData/, ValidData/, TestData/
```

**Key Features**:
- Loads 14-channel HDF5 images
- Applies per-band normalization
- Handles augmentation pipeline
- Returns: `{image: [14,128,128], mask: [128,128], filename: str}`

### Normalization Strategy

**Rationale**: Different bands have different value ranges and meanings.

#### Sentinel-2 Bands (1-12): Percentile Normalization

```python
for i in range(12):
    p2, p98 = np.percentile(band, (2, 98))  # Clip outliers
    band = np.clip(band, p2, p98)
    band_normalized = (band - p2) / (p98 - p2)
```

**Why percentile?**
- Robust to outliers (clouds, shadows)
- Preserves relative relationships
- More stable than min-max across diverse scenes

#### Topographic Bands (13-14): Min-Max Normalization

```python
slope_normalized = (slope - slope.min()) / (slope.max() - slope.min())
dem_normalized = (dem - dem.min()) / (dem.max() - dem.min())
```

**Why min-max?**
- Topographic data has stable ranges
- No outliers expected
- Simpler and faster

### Data Augmentation

**Rationale**: With only 3,799 training images, augmentation is critical to prevent overfitting.

#### Training Augmentation Pipeline

```python
A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        scale=(0.85, 1.15),
        rotate=(-45, 45),
        p=0.6
    ),
    
    # Noise
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.4),
    
    # Intensity
    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.4
    ),
    
    # Deformation
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    
    ToTensorV2()
])
```

**Why these augmentations?**
- **Geometric**: Landslides occur at any orientation
- **Noise**: Simulates sensor noise and atmospheric effects
- **Intensity**: Accounts for different lighting/seasonal conditions
- **Elastic**: Simulates terrain deformation

#### Validation/Test: No Augmentation

```python
A.Compose([ToTensorV2()])
```

Only tensor conversion, no augmentation. This ensures fair evaluation.

---

## Model Architecture

### Attention U-Net

**Implementation**: `src/model.py`

#### Why Attention U-Net?

Standard U-Net uses skip connections to preserve spatial information. Attention U-Net adds **attention gates** to these connections:

```
Skip Connection (Standard U-Net):
  encoder_features → concatenate → decoder

Skip Connection (Attention U-Net):
  encoder_features → attention_gate → concatenate → decoder
                            ↑
                     decoder_features (gating signal)
```

**Benefits**:
1. **Selective Feature Propagation**: Only relevant features are passed through
2. **Background Suppression**: Reduces false positives in non-landslide areas
3. **Improved Boundaries**: Better landslide edge detection

#### Architecture Details

```
Input: [Batch, 14, 128, 128]

Encoder Path:
  ├─ DoubleConv(14 → 64)      [128×128]
  ├─ Down(64 → 128)           [64×64]
  ├─ Down(128 → 256)          [32×32]
  ├─ Down(256 → 512)          [16×16]
  └─ Down(512 → 1024)         [8×8]  ← Bottleneck

Decoder Path:
  ├─ Up(1024 → 512)           [16×16]
  │  └─ AttentionGate(512, 512)
  ├─ Up(512 → 256)            [32×32]
  │  └─ AttentionGate(256, 256)
  ├─ Up(256 → 128)            [64×64]
  │  └─ AttentionGate(128, 128)
  └─ Up(128 → 64)             [128×128]
     └─ AttentionGate(64, 64)

Output: [Batch, 2, 128, 128]  ← 2 classes
```

#### DoubleConv Block

```python
Conv2d(in, mid, 3×3) → BatchNorm → ReLU → Dropout(0.3)
→ Conv2d(mid, out, 3×3) → BatchNorm → ReLU
```

**Key Features**:
- **Batch Normalization**: Stabilizes training
- **Dropout (0.3)**: Prevents overfitting
- **3×3 Convolutions**: Standard receptive field

#### Attention Gate

```python
W_g(gating_signal) + W_x(skip_connection)
  → ReLU
  → ψ (1×1 Conv)
  → Sigmoid
  → attention_weights ∈ [0, 1]

output = skip_connection * attention_weights
```

**Interpretation**:
- `attention_weights → 1`: Important features (keep)
- `attention_weights → 0`: Irrelevant features (suppress)

### Model Size

```
Total Parameters: 34,000,000 (34M)
Trainable Parameters: 34,000,000
Memory (FP32): ~140 MB
```

---

## Loss Function Design

### The Challenge: Extreme Class Imbalance

```
Background: 98.1% of pixels
Landslide:   1.9% of pixels
```

**Problem**: Standard Cross-Entropy Loss will focus on background, ignoring landslides.

### Solution: Combined Loss Function

```python
Loss = 0.4 × Focal Loss + 0.6 × Tversky Loss
```

#### Focal Loss

**Formula**:
```
FL(p_t) = -α(1 - p_t)^γ log(p_t)

where:
  p_t = predicted probability for true class
  α = 0.25 (balance factor)
  γ = 3.0 (focusing parameter)
```

**Purpose**: Focus on hard examples

**How it works**:
- Easy examples (high confidence): Low weight
- Hard examples (low confidence): High weight

**Example**:
```
Easy pixel (p_t = 0.95): weight = (1 - 0.95)^3 = 0.000125  ← Low
Hard pixel (p_t = 0.60): weight = (1 - 0.60)^3 = 0.064000  ← High
```

#### Tversky Loss

**Formula**:
```
TL = 1 - (TP / (TP + α·FP + β·FN))

where:
  α = 0.2 (false positive weight)
  β = 0.8 (false negative weight)
```

**Purpose**: Penalize false negatives more than false positives

**Rationale**: 
- Missing a landslide (FN) is worse than a false alarm (FP)
- With β > α, we prioritize recall over precision

**Example**:
```
Scenario 1: High FP, Low FN
  TL = 1 - (100 / (100 + 0.2·50 + 0.8·5)) = 0.077  ← Low loss

Scenario 2: Low FP, High FN
  TL = 1 - (100 / (100 + 0.2·5 + 0.8·50)) = 0.292  ← High loss
```

### Why This Combination?

| Loss Component | Handles | Weight |
|----------------|---------|--------|
| Focal Loss | Hard examples, class imbalance | 0.4 |
| Tversky Loss | False negatives (recall) | 0.6 |

**Result**: Model learns to:
1. Focus on difficult landslide pixels (Focal)
2. Avoid missing landslides (Tversky)
3. Handle class imbalance (both)

---

## Training Configuration

### Hyperparameters

```python
# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=2e-4,
    betas=(0.9, 0.999)
)

# Learning Rate Schedule
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True
)

# Training
batch_size = 16
epochs = 40
early_stopping_patience = 12
gradient_clipping = 1.0
```

### Why These Choices?

#### AdamW (not Adam)
- **Better weight decay**: Decoupled weight decay improves generalization
- **Standard for vision tasks**: Proven effective for semantic segmentation

#### Learning Rate: 1e-4
- **Stable convergence**: Not too fast (avoids oscillation), not too slow
- **Colab T4 GPU**: Allows batch size 16 with stable training

#### ReduceLROnPlateau
- **Adaptive**: Reduces LR when validation F1 plateaus
- **Patience=5**: Gives model 5 epochs to improve before reducing LR
- **Factor=0.5**: Halves learning rate (1e-4 → 5e-5 → 2.5e-5)

#### Early Stopping (Patience=12)
- **Prevents overfitting**: Stops training when no improvement
- **Patience=12**: More conservative than typical (usually 5-10)
- **Reason**: Small validation set (245 images) has high variance

#### Gradient Clipping (max_norm=1.0)
- **Prevents exploding gradients**: Especially important with Focal Loss
- **Stabilizes training**: Ensures gradients don't grow too large

---

## Optimization Strategy

### Training Loop

```python
for epoch in range(epochs):
    # Training
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['image'])
        loss = combined_loss(outputs, batch['mask'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    val_f1 = evaluate(model, val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_f1)
    
    # Early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_checkpoint(model, 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

### Training Environment: Google Colab

**Why Colab?**
- **Free T4 GPU**: Much faster than local MX130 (18× speedup)
- **16 GB RAM**: Allows batch size 16
- **Training Time**: ~1 hour vs. 18 hours locally

**Colab Setup**:
```python
# Check GPU
!nvidia-smi

# Install dependencies
!pip install albumentations h5py

# Mount Google Drive (for data)
from google.colab import drive
drive.mount('/content/drive')
```

### Training Progression

**Typical Training Curve**:

```
Epoch 1-5:   Rapid improvement (F1: 0.30 → 0.55)
Epoch 6-15:  Steady improvement (F1: 0.55 → 0.62)
Epoch 16-25: Slow improvement (F1: 0.62 → 0.66)
Epoch 26-30: Plateau (F1: 0.66)
Epoch 31+:   LR reduced, minor improvement
```

**Best Checkpoint**: Epoch 25 (Validation F1: 0.6594)

---

## Evaluation Protocol

### Metrics

Primary metric: **F1 Score** (harmonic mean of precision and recall)

```python
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why F1?**
- Balances precision and recall
- More meaningful than accuracy for imbalanced data
- Standard metric in Landslide4Sense competition

### Evaluation Process

```python
model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['image'])
        predictions = torch.argmax(outputs, dim=1)
        
        # Compute metrics
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions)
```

### Ensemble Prediction

**Method**: Average predictions from 3 checkpoints

```python
checkpoints = ['epoch_25.pth', 'epoch_30.pth', 'epoch_35.pth']
predictions = []

for ckpt in checkpoints:
    model.load_state_dict(torch.load(ckpt))
    pred = model(image)
    predictions.append(pred)

# Average logits
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
final_pred = torch.argmax(ensemble_pred, dim=1)
```

**Why ensemble?**
- Reduces variance
- More robust to outliers
- **Improved F1**: 0.5746 (single) → 0.5963 (ensemble) = +3.8%

### Threshold Selection

**Default**: 0.5 (standard for binary classification)

**Optimized**: 0.7 (after threshold search on validation set)

```python
# Convert logits to probabilities
probs = torch.softmax(logits, dim=1)[:, 1]  # Class 1 probability

# Apply threshold
predictions = (probs > 0.7).long()
```

**Result**: Minor improvement (+1% F1), not significant

---

## Challenges & Solutions

### Challenge 1: Data Leakage

**Problem**: Original ValidSplit came from same data as TrainSplit

**Solution**: Use official ValidData (different regions)

**Impact**: Honest metrics (F1: 0.71 → 0.66)

### Challenge 2: Extreme Class Imbalance

**Problem**: Only 1.9% of pixels are landslides

**Solutions**:
1. Focal Loss (focus on hard examples)
2. Tversky Loss (penalize false negatives)
3. Weighted sampling (not used - augmentation sufficient)

### Challenge 3: Small Validation Set

**Problem**: ValidData has only 245 images → high variance

**Solutions**:
1. Conservative early stopping (patience=12)
2. Multiple checkpoints saved
3. Ensemble of 3 models

### Challenge 4: Geographic Domain Shift

**Problem**: ValidData from different regions than TestData

**Result**: 9.6% gap between validation and test F1

**Attempted Solutions**:
1. More augmentation ✓ (helped)
2. Ensemble ✓ (helped)
3. Domain adaptation ✗ (not implemented)

**Conclusion**: Gap is a dataset limitation, not methodology issue

### Challenge 5: Limited GPU Memory

**Problem**: MX130 has only 2GB VRAM → batch size 4

**Solution**: Train on Google Colab (T4 GPU) → batch size 16

**Impact**: 18× faster training + better convergence

---

## Reproducibility

### Random Seeds

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Training Notebook

See `notebooks/train_colab.ipynb` for complete training code.

**To reproduce**:
1. Open notebook in Google Colab
2. Upload or mount dataset
3. Run all cells
4. Download checkpoints

**Expected Results** (±2% due to GPU randomness):
- Validation F1: 0.64-0.68
- Test F1: 0.57-0.61

---

## Future Improvements

### Recommended Next Steps

1. **Collect More Validation Data**: From test regions
2. **Transformer Architecture**: Try SegFormer or Mask2Former
3. **Multi-Task Learning**: Add landslide type classification
4. **Semi-Supervised Learning**: Use unlabeled satellite data
5. **Test-Time Augmentation**: Revisit with better implementation
6. **Domain Adaptation**: Address geographic domain shift

---

## References

1. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018
2. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
3. **Tversky Loss**: Salehi et al., "Tversky loss function for image segmentation using 3D FCNN", 2017
4. **Landslide4Sense**: Ghorbanzadeh et al., "The outcome of the 2022 landslide4sense competition", 2022

---

**For implementation details, see the source code in `src/`.**
