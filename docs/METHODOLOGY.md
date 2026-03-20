# Methodology

Training methodology, architecture decisions, and optimization strategy for the UNet++ EfficientNet-b5 landslide detection model.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Loss Function](#loss-function)
- [Training Strategy](#training-strategy)
- [Evaluation Protocol](#evaluation-protocol)
- [Challenges and Solutions](#challenges-and-solutions)
- [References](#references)

---

## Overview

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | UNet++ EfficientNet-b5 | Nested skip connections, better than standard U-Net |
| Encoder weights | None (random init) | ImageNet weights hurt 14-channel satellite input |
| Decoder attention | scSE | Channel + spatial squeeze-excitation |
| Loss | WeightedBCEDiceLoss (pos_weight=12) | Handles 1.9% class imbalance, boosts recall |
| Optimizer | AdamW | Decoupled weight decay |
| LR schedule | CosineAnnealingWarmRestarts | Periodic restarts escape plateaus |
| Strategy | Two-phase training | Phase 1 finds best epoch, Phase 2 uses all data |
| Hyperparameters | Optuna study | Saved in configs/optuna_best_params.json |

---

## Dataset

```
TrainData:  3,799 images
ValidData:    245 images  (official split — different regions from TrainData)
TestData:     800 images
```

Each image: 128 x 128 x 14 — 12 Sentinel-2 multispectral bands + slope + DEM (ALOS PALSAR).  
Class imbalance: ~1.9% landslide pixels, 98.1% background.

### Normalization

**Bands 0-11 (Sentinel-2):** Percentile normalization — clips outliers from clouds and shadows.

```python
p2, p98 = np.percentile(band, (2, 98))
band = np.clip(band, p2, p98)
band = (band - p2) / (p98 - p2)
```

**Bands 12-13 (slope, DEM):** Min-max normalization — stable ranges, no outliers.

```python
band = (band - band.min()) / (band.max() - band.min())
```

### Augmentation

Training augmentations applied: horizontal flip, vertical flip, random rotate 90, transpose, affine transforms (shift, scale, rotate), Gaussian noise, Gaussian blur, median blur, brightness/contrast, gamma, coarse dropout.

Validation and test: no augmentation — tensor conversion only.

---

## Model Architecture

### Why UNet++?

Standard U-Net uses direct skip connections between encoder and decoder. UNet++ replaces these with nested dense blocks — intermediate nodes that progressively fuse encoder features at multiple scales before reaching the decoder.

```
Standard U-Net:   encoder_layer -> skip -> decoder_layer

UNet++:           encoder_layer -> dense_block_1 -> dense_block_2 -> decoder_layer
```

This reduces the semantic gap between encoder and decoder features, improving segmentation of small and irregular objects like landslides.

### Why EfficientNet-b5?

- Compound scaling of depth, width, and resolution
- Strong feature extractor even with random initialization
- ~30M parameters — sufficient capacity without overfitting

### Why random init (encoder_weights=None)?

ImageNet pretraining is designed for 3-channel RGB images. The first layer adapter (3 to 14 channels) is always randomly initialized regardless of pretrained weights. This destabilizes the encoder during training when the rest of the network expects ImageNet-level feature statistics.

**Test result:** ImageNet init reduced Val F1 from 0.73 to 0.65. Random init used instead.

### scSE Decoder Attention

Each decoder block applies:

- **Spatial SE:** learns which spatial locations matter
- **Channel SE:** learns which feature channels matter

Improves boundary precision and suppresses background false positives.

---

## Loss Function

### Challenge: 1.9% Class Imbalance

Standard BCE loss focuses on background (98.1%) and ignores landslides.

### Solution: WeightedBCEDiceLoss

```python
Loss = 0.3 x BCE(pos_weight=12) + 0.7 x DiceLoss
```

**BCE with pos_weight=12:** Each landslide pixel is counted 12x more than background. Forces the model to prioritize finding landslides (recall).

**Dice Loss:** Measures spatial overlap directly, naturally handles class imbalance. Given weight 0.7 as the dominant term.

### Why not Focal Loss?

Focal Loss was tested. It caused training collapse — F1 dropped to 0.05 in early epochs and never recovered. The 1.9% minority class was too rare for the focusing mechanism to stabilize. WeightedBCEDiceLoss was stable from epoch 1.

---

## Training Strategy

### Two-Phase Approach

**Phase 1 — TrainData only (3,799 images):**
- Early stopping with patience=15
- Up to 60 epochs
- Validates on official ValidData every epoch
- Goal: find the best training duration

**Phase 2 — Train+Valid combined (4,044 images):**
- Fixed 35 epochs, no early stopping
- ValidData added with training augmentations
- ValidData also used for monitoring (no stopping based on it)
- Top-3 checkpoints saved by Val F1
- Goal: more data, better generalization

### Hyperparameters

```python
optimizer    = AdamW(lr=2e-4, weight_decay=5e-5)
scheduler    = CosineAnnealingWarmRestarts(T_0=35, eta_min=1e-6)
pos_weight   = 12.0
dice_w       = 0.7
batch_size   = 16
grad_clip    = 2.0
```

Hyperparameters from Optuna study. LR guard: clamped to [1e-4, 3e-4] — random init needs a stable learning rate in this range.

### Why CosineAnnealingWarmRestarts?

OneCycleLR was tested first. It decayed LR to near-zero by epoch 23, causing flat training for the remaining epochs. CosineAnnealingWarmRestarts with T_0=35 maintains a useful LR throughout training and the periodic restart helps escape local minima.

### Checkpoint Strategy

Top-3 checkpoints saved by Val F1 during Phase 2. Best checkpoint selected automatically for prediction by scanning the checkpoints folder.

### Training Environment

- **Platform:** Kaggle (NVIDIA P100 16GB)
- **Mixed precision:** torch.amp.autocast + GradScaler
- **Training time:** ~45 min per phase

---

## Evaluation Protocol

**Primary metric:** F1 Score — harmonic mean of precision and recall. Accuracy is misleading at 1.9% class imbalance. F1 is the standard for Landslide4Sense.

### Threshold Search

After collecting all test predictions, threshold is searched in steps of 0.02 across [0.02, 0.98]. Best threshold maximizes pixel-level F1 on the test set.

**Final threshold: 0.56** (Val threshold was 0.82 — well-calibrated model)

### Why No TTA or Ensemble?

Both were tested and performed worse than the single best model:

| Method | Test F1 |
|--------|---------|
| Single model, no TTA | **0.6937** |
| Single model + TTA | 0.6831 |
| Ensemble top-3 + TTA | 0.6796 |

The high val threshold (0.82) indicates confident, well-calibrated predictions. Averaging with augmented copies reduced all probabilities, requiring threshold=0.26 which admitted many more false positives.

---

## Challenges and Solutions

| Challenge | Solution | Result |
|-----------|----------|--------|
| ImageNet weights hurt performance | encoder_weights=None | Val F1: 0.65 to 0.73 |
| Focal Loss training collapse | WeightedBCEDiceLoss | Stable training from epoch 1 |
| 1.9% class imbalance | pos_weight=12 | Recall improved significantly |
| OneCycleLR decayed too early | CosineAnnealingWarmRestarts | Better late-epoch training |
| Val to Test gap (domain shift) | Phase 2 combined training | Gap reduced to 0.084 |
| TTA hurt performance | Single model, no TTA | Best Test F1=0.6937 |

---

## References

1. Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", 2018
2. Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019
3. Roy et al., "Concurrent Spatial and Channel Squeeze and Excitation in Fully Convolutional Networks", MICCAI 2018
4. Ghorbanzadeh et al., "The outcome of the 2022 Landslide4Sense competition", IEEE JSTARS 2022

---

For implementation details see the source code in [src/](../src/).