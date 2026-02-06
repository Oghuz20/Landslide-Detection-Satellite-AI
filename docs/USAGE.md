# 📖 Usage Guide

> Complete guide for using the landslide detection model - from installation to deployment

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Inference](#inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Custom Dataset](#custom-dataset)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1-Minute Quick Test

```bash
# Clone repository
git clone https://github.com/Oghuz20/Landslide-Detection-Satellite-AI.git
cd Landslide-Detection-Satellite-AI

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run inference on test data
python ensemble_predict.py
```

Output: Predictions saved to `predictions/test_final/`

---

## Installation

### Prerequisites

- **Python**: 3.11+ (3.12 recommended)
- **GPU** (optional): NVIDIA with CUDA 11.8
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 3 GB for code + models (add 2.4 GB for dataset)

### Step 1: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv venv_landslide
.\venv_landslide\Scripts\Activate.ps1
```

**Linux/Mac**:
```bash
python3 -m venv venv_landslide
source venv_landslide/bin/activate
```

### Step 2: Install Dependencies

**With GPU (CUDA 11.8)**:
```bash
# Install PyTorch with CUDA
pip install torch==2.7.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt --break-system-packages
```

**CPU Only**:
```bash
# Install PyTorch CPU version
pip install torch==2.7.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt --break-system-packages
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.7.1+cu118
CUDA: True  (or False if CPU-only)
```

### Step 4: Download Dataset (Optional)

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

### Using the Ensemble Model (Recommended)

**Script**: `ensemble_predict.py`

```bash
python ensemble_predict.py
```

**What it does**:
1. Loads 3 best checkpoints (epochs 25, 30, 35)
2. Averages predictions from all 3 models
3. Applies threshold (0.7)
4. Saves predictions to `predictions/test_final/`

**Output**:
```
predictions/test_final/
├── mask_1.h5
├── mask_2.h5
├── ...
└── mask_800.h5
```

### Custom Inference

```python
import torch
from src.model import create_model
from src.dataset import LandslideDataset, get_valid_transforms

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('attention_unet', n_channels=14, n_classes=2)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.to(device)
model.eval()

# Load data
dataset = LandslideDataset(
    data_dir='./data',
    split='test',
    transform=get_valid_transforms()
)

# Predict on single image
sample = dataset[0]
image = sample['image'].unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).cpu().numpy()

print(f"Prediction shape: {prediction.shape}")  # [1, 128, 128]
print(f"Unique values: {np.unique(prediction)}")  # [0, 1]
```

### Batch Inference

```python
from torch.utils.data import DataLoader

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

# Predict on all images
all_predictions = []

with torch.no_grad():
    for batch in loader:
        images = batch['image'].to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        all_predictions.extend(predictions)

print(f"Total predictions: {len(all_predictions)}")
```

### Save Predictions

```python
import h5py
import os

output_dir = 'predictions/my_predictions'
os.makedirs(output_dir, exist_ok=True)

for idx, pred in enumerate(all_predictions):
    output_path = os.path.join(output_dir, f'mask_{idx+1}.h5')
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('mask', data=pred, compression='gzip')
    
    if idx % 100 == 0:
        print(f"Saved {idx+1} predictions...")
```

---

## Training

### Training on Google Colab (Recommended)

**Notebook**: `notebooks/train_colab.ipynb`

1. Open notebook in [Google Colab](https://colab.research.google.com/)
2. Upload dataset or mount Google Drive
3. Run all cells
4. Download trained checkpoints

**Expected Training Time**: ~1 hour on T4 GPU

### Training Locally

```python
import torch
from torch.utils.data import DataLoader
from src.dataset import LandslideDataset, get_train_transforms, get_valid_transforms
from src.model import create_model

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('attention_unet', n_channels=14, n_classes=2).to(device)

# Create datasets
train_dataset = LandslideDataset(
    data_dir='./data',
    split='train',
    transform=get_train_transforms()
)

valid_dataset = LandslideDataset(
    data_dir='./data',
    split='valid',
    transform=get_valid_transforms()
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # Reduce if out of memory
    shuffle=True,
    num_workers=4
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

# Training configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=2e-4
)

# Training loop (simplified)
for epoch in range(40):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    val_f1 = evaluate(model, valid_loader)
    print(f"Epoch {epoch+1}: Val F1 = {val_f1:.4f}")
```

### Resume Training from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/epoch_20.pth')
model.load_state_dict(checkpoint)

# Continue training
# ... (same training loop as above)
```

---

## Evaluation

### Evaluate on Test Set

**Script**: `src/evaluate.py`

```bash
python src/evaluate.py
```

**Output**:
```
================================================================================
TEST SET RESULTS
================================================================================

F1 Score:   0.5963
Precision:  0.5403
Recall:     0.6652
Accuracy:   0.9878

Confusion Matrix:
  TN:  80,317,945  |  FP:     892,134
  FN:     561,300  |  TP:   1,048,621
================================================================================
```

### Custom Evaluation

```python
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# Compute metrics
y_true = []  # Ground truth labels
y_pred = []  # Model predictions

for batch in test_loader:
    # ... (get predictions)
    y_true.extend(batch['mask'].flatten().numpy())
    y_pred.extend(predictions.flatten())

# Calculate metrics
f1 = f1_score(y_true, y_pred, pos_label=1)
precision = precision_score(y_true, y_pred, pos_label=1)
recall = recall_score(y_true, y_pred, pos_label=1)

print(f"F1: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

### Confusion Matrix Analysis

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(f"  TN: {cm[0,0]:>10,}  |  FP: {cm[0,1]:>10,}")
print(f"  FN: {cm[1,0]:>10,}  |  TP: {cm[1,1]:>10,}")

# Derived metrics
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])

print(f"Specificity: {specificity:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
```

---

## Visualization

### Generate All Visualizations

**Script**: `src/visualize.py`

```bash
python src/visualize.py
```

**Output** (saved to `visualizations/`):
1. Performance distribution plots
2. Confusion matrix grid (16 examples)
3. Best 5 predictions
4. Worst 5 predictions
5. 10 random samples

### Visualize Single Image

```python
from src.visualize import LandslideVisualizer

viz = LandslideVisualizer(
    data_dir='./data',
    pred_dir='./predictions/test_final'
)

# Visualize specific image
viz.plot_single_prediction('image_42.h5', save=True)
```

**Output**: 6-panel plot showing:
- Original RGB image
- Ground truth mask
- Model prediction
- Overlay comparison
- Error analysis
- Statistics

### Custom Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data
image_rgb = ...  # [128, 128, 3]
ground_truth = ...  # [128, 128]
prediction = ...  # [128, 128]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
axes[0].imshow(image_rgb)
axes[0].set_title('Satellite Image (RGB)')
axes[0].axis('off')

# Ground truth
axes[1].imshow(ground_truth, cmap='RdYlGn')
axes[1].set_title('Ground Truth')
axes[1].axis('off')

# Prediction
axes[2].imshow(prediction, cmap='RdYlGn')
axes[2].set_title('Prediction')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('my_visualization.png', dpi=300)
plt.show()
```

---

## Custom Dataset

### Prepare Your Own Data

Your data must match this format:

**Structure**:
```
my_data/
├── img/
│   ├── image_1.h5
│   ├── image_2.h5
│   └── ...
└── mask/
    ├── mask_1.h5
    ├── mask_2.h5
    └── ...
```

**Image Format** (.h5):
```python
import h5py
import numpy as np

# Create sample image (14 channels, 128×128)
image = np.random.rand(128, 128, 14).astype(np.float32)

with h5py.File('my_data/img/image_1.h5', 'w') as f:
    f.create_dataset('img', data=image, compression='gzip')
```

**Mask Format** (.h5):
```python
# Create sample mask (binary, 128×128)
mask = np.random.randint(0, 2, (128, 128)).astype(np.uint8)

with h5py.File('my_data/mask/mask_1.h5', 'w') as f:
    f.create_dataset('mask', data=mask, compression='gzip')
```

### Use Custom Dataset

```python
from src.dataset import LandslideDataset

# Modify dataset class to support custom path
class CustomDataset(LandslideDataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.h5')])

# Load custom dataset
dataset = CustomDataset(
    img_dir='my_data/img',
    mask_dir='my_data/mask',
    transform=get_valid_transforms()
)

# Use as normal
sample = dataset[0]
```

---

## Deployment

### Export Model to ONNX

```python
import torch
from src.model import create_model

# Load model
model = create_model('attention_unet', n_channels=14, n_classes=2)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 14, 128, 128)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'landslide_model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✓ Model exported to landslide_model.onnx")
```

### Run Inference with ONNX

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('landslide_model.onnx')

# Prepare input
input_data = np.random.randn(1, 14, 128, 128).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
prediction = np.argmax(outputs[0], axis=1)

print(f"Prediction shape: {prediction.shape}")
```

### Deploy on Edge Device

For Raspberry Pi, NVIDIA Jetson, or similar:

1. **Export to TorchScript**:
```python
model_scripted = torch.jit.script(model)
model_scripted.save('landslide_model_scripted.pt')
```

2. **Load on device**:
```python
model = torch.jit.load('landslide_model_scripted.pt')
model.eval()
```

3. **Run inference** (same as regular PyTorch)

---

## Troubleshooting

### Problem: Out of Memory (CUDA)

**Solution**: Reduce batch size

```python
# In DataLoader
batch_size=4  # Instead of 16
```

Or use gradient accumulation:

```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Problem: Slow Data Loading

**Solution**: Increase `num_workers`

```python
DataLoader(..., num_workers=8)  # More workers = faster loading
```

**Note**: Windows users may need `num_workers=0`

### Problem: Low F1 Score

**Check**:
1. Data normalization is applied
2. Using ensemble prediction (not single model)
3. Correct threshold (0.7 recommended)
4. Model loaded correctly

```python
# Verify prediction distribution
print(f"Positive pixels: {(predictions == 1).sum()} ({(predictions == 1).mean()*100:.2f}%)")
```

### Problem: Predictions All Zero

**Cause**: Model predicts everything as background

**Solutions**:
1. Lower threshold: `threshold=0.5` instead of `0.7`
2. Check loss function weights
3. Verify class balance in training data

### Problem: Import Errors

```bash
# If missing modules
pip install <module_name> --break-system-packages

# Common missing modules
pip install albumentations h5py tqdm --break-system-packages
```

### Problem: FileNotFoundError

**Check**:
1. Data directory structure matches expected format
2. File paths are correct (Windows: `\` vs Linux: `/`)
3. Files have correct extensions (`.h5`)

```python
# Debug: Check what files exist
import os
print(os.listdir('data/TestData/img')[:5])
```

---

## Advanced Usage

### Multi-GPU Training

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

model = model.to(device)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch['image'])
        loss = criterion(outputs, batch['mask'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Custom Loss Function

```python
import torch.nn as nn

class MyCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Your custom loss logic
        loss = ...
        return loss

criterion = MyCustomLoss()
```

---

## Performance Optimization

### CPU Optimization

```python
# Use Intel MKL (if available)
torch.set_num_threads(8)

# Enable NNPACK (mobile)
torch.backends.nnpack.enabled = True
```

### GPU Optimization

```python
# Enable cudnn autotuner
torch.backends.cudnn.benchmark = True

# Use TF32 on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
```

---

## Additional Resources

- **Documentation**: See `docs/` folder
- **Training Notebook**: `notebooks/train_colab.ipynb`
- **Example Scripts**: Root directory (`ensemble_predict.py`, etc.)
- **Issues**: [GitHub Issues](https://github.com/Oghuz20/Landslide-Detection-Satellite-AI/issues)

---

**Need more help?** Open an issue on GitHub with:
1. Error message (full traceback)
2. Python version
3. PyTorch version
4. GPU/CPU information
5. Minimal code to reproduce the problem
