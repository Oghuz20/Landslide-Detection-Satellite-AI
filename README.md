# Landslide Detection Using AI Techniques and Satellite Imagery

This repository presents a deep learning–based approach for automatic landslide detection using multi-spectral satellite imagery. The task is formulated as a **binary semantic segmentation** problem, where a convolutional neural network identifies landslide regions at the pixel level.

The project ia focusing on the application of **AI techniques in remote sensing** for disaster monitoring and risk management.

---

## Overview
Landslides are among the most destructive natural hazards, often causing severe damage to infrastructure and loss of life. Accurate and timely detection of landslide-prone areas is essential for disaster prevention and mitigation.

In this work, a **U-Net architecture** is trained on multi-band satellite image patches to automatically segment landslide regions. The model learns spatial and spectral patterns associated with landslide occurrences from satellite imagery.

---

## Dataset
The experiments are conducted using the **Landslide4Sense** dataset, which consists of:
- Multi-spectral satellite image patches  
- Corresponding binary masks indicating landslide areas  

Each input image has a shape of **128 × 128 × 14**, representing multiple spectral bands.

### Dataset Usage
The Landslide4Sense dataset is **not included** in this repository due to its size and licensing constraints.

To reproduce the experiments:
1. Download the Landslide4Sense dataset from its official source.
2. Organize the data using the following structure:

data/
├── images/
│ ├── image_1.h5
│ ├── image_2.h5
│ └── ...
└── masks/
├── mask_1.h5
├── mask_2.h5
└── ...

3. Use the provided training and evaluation scripts.

All preprocessing, training, and evaluation steps are fully implemented in this repository.

---

## Methodology
- **Model**: U-Net  
- **Framework**: TensorFlow / Keras  
- **Task**: Binary semantic segmentation  
- **Input size**: 128 × 128 × 14  
- **Loss function**: Binary Cross-Entropy + Dice Loss  
- **Evaluation metrics**:
  - Dice coefficient
  - Intersection over Union (IoU)
- **Data augmentation**:
  - Horizontal and vertical flips
- **Post-processing**:
  - Probability threshold optimization

---

## Results
Model performance was evaluated on a held-out test set using an optimized probability threshold.

**Best threshold:** 0.40  

**Test performance:**
- **Loss:** 0.338  
- **Dice coefficient:** 0.721  
- **IoU score:** 0.567  

These results indicate that the proposed model can reliably segment landslide regions from satellite imagery, achieving strong pixel-level accuracy.

Sample qualitative results and training curves are available in the `results/` directory.

---


## Reproducibility
This repository provides the **complete pipeline** for:
- Data loading
- Model training
- Evaluation
- Threshold optimization
- Result visualization

By downloading the dataset separately and following the documented structure, all experiments can be reproduced.

---

## Future Work
Possible extensions of this work include:
- Incorporating transformer-based segmentation models
- Using multi-temporal satellite imagery
- Applying advanced data augmentation techniques
- Cross-dataset generalization studies

---

## License
This project is intended for **academic and research purposes**.  
The dataset license is governed by the original dataset authors.

---

## Acknowledgments
- Landslide4Sense dataset authors  
- Open-source TensorFlow and Keras communities