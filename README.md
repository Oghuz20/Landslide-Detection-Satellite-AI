# Landslide Detection Using AI Techniques and Satellite Imagery

This project presents a deep learning–based approach for automatic landslide detection using multi-spectral satellite imagery. A U-Net convolutional neural network is trained to perform pixel-level segmentation of landslide areas.

## Dataset
The experiments are conducted using the **Landslide4Sense** dataset, which contains multi-band satellite image patches and corresponding binary landslide masks.

- Image size: 128 × 128
- Channels: 14 spectral bands
- Task: Binary segmentation (landslide / non-landslide)

## Methodology
- Model: U-Net
- Framework: TensorFlow / Keras
- Loss function: Binary Cross-Entropy + Dice Loss
- Metrics: Dice Coefficient, IoU
- Data format: HDF5 (.h5)
- Post-processing: Probability thresholding

## Training Details
- Train / Validation / Test split: 70% / 15% / 15%
- Batch size: 16
- Optimizer: Adam
- Initial learning rate: 1e-3
- Best threshold (validation): **0.15**

## Results
Final evaluation on the test set:

| Metric | Value |
|------|------|
| Dice Coefficient | ~0.72 |
| IoU Score | ~0.57 |

These results demonstrate reliable landslide segmentation performance on unseen satellite data.

## Repository Structure
- `notebook/` – training and evaluation notebook  
- `configs/` – inference configuration (threshold)  
- `results/` – quantitative and qualitative outputs  

## Notes
Due to dataset size, trained model weights are not included. The model can be fully reproduced by running the provided notebook.

## Author
Project – Landslide Detection Using AI Techniques and Satellite Imagery
