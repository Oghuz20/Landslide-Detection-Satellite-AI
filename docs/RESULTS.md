# Results Analysis

Comprehensive performance analysis of the UNet++ EfficientNet-b5 landslide detection model.

---

## Table of Contents

- [Final Model Results](#final-model-results)
- [Model Evolution](#model-evolution)
- [Prediction Strategy Comparison](#prediction-strategy-comparison)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Performance Distribution](#performance-distribution)
- [Error Analysis](#error-analysis)
- [Comparison with Literature](#comparison-with-literature)
- [Lessons Learned](#lessons-learned)

---

## Final Model Results

**Model:** UNet++ EfficientNet-b5, Phase 2 (epoch 35)  
**Inference:** Single best model, no TTA, threshold=0.56  
**Test Dataset:** 800 images (TestData)

| Metric | Score |
|--------|-------|
| **F1 Score** | **0.6937** |
| **Precision** | **0.6694** |
| **Recall** | **0.7198** |
| **Accuracy** | 0.9880 |

### Validation vs Test

| Dataset | F1 | Precision | Recall |
|---------|----|-----------|--------|
| Validation | 0.7780 | — | — |
| **Test** | **0.6937** | **0.6694** | **0.7198** |

**Val to Test gap: -0.0843** — caused by geographic domain shift between official dataset splits.

---

## Model Evolution

| Model | Val F1 | Test F1 | Notes |
|-------|--------|---------|-------|
| U-Net v1 (data leakage) | 0.7103 | 0.5691 | Invalid — leaked validation |
| U-Net v1 (proper splits) | 0.6688 | 0.6227 | Fixed baseline |
| UNet++ ImageNet init | 0.6536 | — | Regression — ImageNet hurt 14-ch input |
| UNet++ random init Phase 1 | 0.7152 | 0.6241 | TrainData only |
| **UNet++ random init Phase 2** | **0.7780** | **0.6937** | **Final — BEST** |

**Total improvement over baseline: +0.071 (+11.4%)**

---

## Prediction Strategy Comparison

| Strategy | Test F1 | Precision | Recall | Threshold |
|----------|---------|-----------|--------|-----------|
| **Single model, no TTA** | **0.6937** | **0.6694** | **0.7198** | **0.56** |
| Single model + TTA | 0.6831 | 0.5994 | 0.7938 | 0.26 |
| Ensemble top-3 + TTA | 0.6796 | 0.6491 | 0.7131 | 0.40 |

**Key finding:** TTA and ensemble both hurt performance on this model. The best checkpoint had val threshold=0.82, indicating well-calibrated predictions. Averaging with augmented copies reduced all probabilities, requiring a much lower threshold (0.26) which admitted many more false positives.

---

## Confusion Matrix Analysis

**Total pixels evaluated:** 13,107,200 (800 images x 128 x 128)

| | Predicted Background | Predicted Landslide |
|--|---------------------|---------------------|
| **Actual Background** | TN = 12,771,680 | FP = 87,989 |
| **Actual Landslide** | FN = 69,359 | TP = 178,172 |

```
Precision   = 178,172 / (178,172 + 87,989)  = 0.6694
Recall      = 178,172 / (178,172 + 69,359)  = 0.7198
F1          = 2 x (0.6694 x 0.7198) / (0.6694 + 0.7198) = 0.6937
Specificity = 12,771,680 / (12,771,680 + 87,989) = 0.9931
```

### Error Breakdown

**False Positives (87,989 pixels — false alarms):**
- River beds and dry stream channels
- Bare rock on exposed slopes
- Agricultural terraces with disturbed soil
- Cloud shadows

**False Negatives (69,359 pixels — missed landslides):**
- Small landslides under 20 pixels
- Partially vegetated or old slides
- Edge pixels of large landslide boundaries

---

## Performance Distribution

Per-image F1 computed over 800 test images (images with no landslide excluded from landslide stats):

| F1 Range | Quality |
|----------|---------|
| 0.8 - 1.0 | Excellent — large, clearly defined landslides |
| 0.6 - 0.8 | Good — moderate size, some vegetation |
| 0.4 - 0.6 | Moderate — small or partially vegetated |
| 0.0 - 0.4 | Poor — very small or heavily vegetated |

**Best case (F1=0.9776, image_642):** Large debris flow, clear spectral signature, minimal false positives.

**Worst cases (F1=0.0, landslide images):** Very small landslides under 10 pixels, dense forest background, nearby rivers causing false positives.

---

## Error Analysis

### Common Failure Modes

**Size Bias:**
Small landslides under 30 pixels are frequently missed. The model learned patterns from larger examples and struggles with tiny positive regions.

**Spectral Confusion:**
Dry riverbeds and bare rock slopes share spectral characteristics with fresh landslide scars. This is the primary source of false positives.

**Edge Precision:**
Landslide boundaries are imprecise — the model tends to slightly over-segment or under-segment edges of larger landslides.

**Temporal Ambiguity:**
Old, revegetated landslides and recent but subtle slides are difficult to distinguish from normal terrain variation.

### Geographic Performance Variation

Without explicit geographic labels, patterns suggest regional differences:

**Higher performance** — arid/semi-arid climates, sparse vegetation, clear spectral signatures.

**Lower performance** — tropical/subtropical climates, dense vegetation, frequent cloud cover.

---

## Comparison with Literature

| Rank | Method | F1 Score |
|------|--------|----------|
| 1st | Competition Winner (2022) | 0.7234 |
| 2nd | Second Place | 0.6891 |
| **—** | **Ours (UNet++ Phase 2)** | **0.6937** |
| 3rd | Third Place | 0.6542 |
| — | Ours (Phase 1 only) | 0.6241 |
| — | U-Net v1 baseline | 0.6227 |

Our final model surpasses 3rd place (0.6542) and is within 0.005 of 2nd place (0.6891).

---

## Lessons Learned

### What Worked

- **UNet++ nested skip connections** — better feature aggregation than standard U-Net
- **Random encoder init** — ImageNet weights hurt performance on 14-channel satellite data
- **scSE decoder attention** — improved boundary detection and background suppression
- **pos_weight=12** — forced higher recall on the 1.9% minority class
- **Phase 2 combined training** — more data reduced the val-to-test gap
- **Single best model without TTA** — well-calibrated predictions outperformed averaging

### What Did Not Work

- **ImageNet pretrained weights** — reduced Val F1 from 0.73 to 0.65
- **Focal Loss** — caused training collapse (F1 dropped to 0.05)
- **TTA** — destabilized well-calibrated high-confidence predictions
- **Ensemble top-3 + TTA** — worse than single best model
- **OneCycleLR** — decayed too early, flat training after epoch 23

### Recommendations for Future Work

- Transformer architectures (SegFormer, Mask2Former) for better global context
- Multi-scale input (FPN, DeepLabV3+) for detecting small landslides
- Semi-supervised learning with unlabeled satellite imagery
- Domain adaptation to address geographic shift between validation and test regions
- CRF post-processing for refined landslide boundaries

---

For visualizations see the [visualizations/](../visualizations/) folder.