# Landslide Detection Dataset

## ⚠️ Data Not Included in Repository

The dataset is **NOT included** in this GitHub repository due to size constraints (approximately 2.4 GB).

---

## 📊 Dataset Structure

```
data/
├── TrainData/          (3,799 images + masks)
│   ├── img/            (.h5 files - 128×128×14 multispectral images)
│   └── mask/           (.h5 files - 128×128 binary masks)
│
├── ValidData/          (245 images + masks)
│   ├── img/
│   └── mask/
│
└── TestData/           (800 images + masks)
    ├── img/
    └── mask/
```

---

## 📋 Data Specifications

### Image Format
- **Format**: HDF5 (`.h5`)
- **Size**: 128 × 128 pixels
- **Channels**: 14 bands
- **Bit Depth**: 32-bit float

### Multispectral Bands
1. **Bands 1-12**: Sentinel-2 multispectral data
   - Blue (B2), Green (B3), Red (B4), Red Edge (B5-B7)
   - NIR (B8, B8A), SWIR (B11, B12)
   - Coastal Aerosol (B1), Water Vapor (B9)
   
2. **Band 13**: Slope (derived from ALOS PALSAR)
   - Terrain slope information
   - Range: 0-90 degrees
   
3. **Band 14**: Digital Elevation Model (DEM, ALOS PALSAR)
   - Elevation information
   - Unit: meters above sea level

### Masks
- **Format**: Binary HDF5 (`.h5`)
- **Values**: 
  - `0` = Non-landslide (background)
  - `1` = Landslide
- **Class Distribution**: Highly imbalanced
  - Background: ~98.1%
  - Landslide: ~1.9%

---

## 📥 Download Instructions

### Option 1: Landslide4Sense Dataset (Official)

The dataset is from the **Landslide4Sense** competition:

1. **Visit**: [IEEE Dataport - Landslide4Sense](https://ieee-dataport.org/competitions/landslide4sense)
2. **Register**: Create a free account
3. **Download**: Get the competition dataset
4. **Extract**: Place in `data/` folder

**Direct Links**:
- Competition Page: https://www.iarai.ac.at/landslide4sense/challenge/
- IEEE Dataport: https://ieee-dataport.org/competitions/landslide4sense

### Option 2: Google Drive (If Shared)

If you have received a shared Google Drive link:

1. Download `landslide_data.zip` or individual folders
2. Extract to project root
3. Verify structure matches above

---

## 🔧 Setup After Download

### 1. Verify Data Structure

```bash
# Check directories exist
python -c "import os; print('✓ TrainData' if os.path.exists('data/TrainData') else '✗ TrainData missing')"
python -c "import os; print('✓ ValidData' if os.path.exists('data/ValidData') else '✗ ValidData missing')"
python -c "import os; print('✓ TestData' if os.path.exists('data/TestData') else '✗ TestData missing')"
```

### 2. Check File Counts

```python
import os

train_count = len([f for f in os.listdir('data/TrainData/img') if f.endswith('.h5')])
valid_count = len([f for f in os.listdir('data/ValidData/img') if f.endswith('.h5')])
test_count = len([f for f in os.listdir('data/TestData/img') if f.endswith('.h5')])

print(f"TrainData: {train_count} images (expected: 3,799)")
print(f"ValidData: {valid_count} images (expected: 245)")
print(f"TestData: {test_count} images (expected: 800)")
```

### 3. Test Data Loading

```python
from src.dataset import LandslideDataset

# Test loading
dataset = LandslideDataset(data_dir='./data', split='train')
print(f"✓ Successfully loaded {len(dataset)} training images")

# Test loading a sample
sample = dataset[0]
print(f"✓ Image shape: {sample['image'].shape}")  # Should be [14, 128, 128]
print(f"✓ Mask shape: {sample['mask'].shape}")    # Should be [128, 128]
```

---

## 📊 Dataset Statistics

### Size Information
| Split | Images | Total Size | Avg per Image |
|-------|--------|------------|---------------|
| Train | 3,799  | ~1.9 GB    | ~500 KB       |
| Valid | 245    | ~122 MB    | ~500 KB       |
| Test  | 800    | ~400 MB    | ~500 KB       |
| **Total** | **4,844** | **~2.4 GB** | **~500 KB** |

### Class Distribution (Pixel-Level)
| Split | Landslide % | Background % | Total Pixels |
|-------|-------------|--------------|--------------|
| Train | 1.93%       | 98.07%       | 62,066,176   |
| Valid | 1.87%       | 98.13%       | 4,001,280    |
| Test  | 1.96%       | 98.04%       | 13,107,200   |

### Geographic Coverage
- **Regions**: Multiple countries with landslide history
- **Climate Zones**: Diverse (tropical, subtropical, temperate)
- **Terrain Types**: Mountains, hills, coastal areas
- **Time Period**: 2016-2020 (Sentinel-2 satellite data)

---

## 🔍 Data Quality

### Preprocessing Applied
- ✅ Cloud cover filtering (<20%)
- ✅ Geometric correction
- ✅ Radiometric calibration
- ✅ Co-registration with topographic data
- ✅ Expert-labeled landslide masks

### Known Issues
- Some images may have partial cloud shadows
- Edge pixels in masks may be approximate
- Small landslides (<10 pixels) may be missed
- Temporal variability in vegetation cover

---

## 📄 Data License

### Landslide4Sense Dataset License

The dataset is provided for **research and educational purposes**.

**Permissions**:
- ✅ Use for academic research
- ✅ Use for educational purposes
- ✅ Modification and derivative works
- ✅ Non-commercial use

**Restrictions**:
- ❌ Commercial use without permission
- ❌ Redistribution of raw data
- ❌ Claiming dataset ownership

For commercial use, contact the dataset creators.

---

## 📚 Citation

If you use this dataset, please cite:

```bibtex
@article{ghorbanzadeh2022landslide4sense,
  title={The outcome of the 2022 landslide4sense competition: 
         Advanced landslide detection from multisource satellite imagery},
  author={Ghorbanzadeh, Omid and Xu, Yonghao and Ghamisi, Pedram 
          and Kopp, Martin and Kreil, David},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations 
           and Remote Sensing},
  volume={15},
  pages={5927--5943},
  year={2022},
  publisher={IEEE}
}
```

**Additional References**:
- Landslide4Sense Competition: https://www.iarai.ac.at/landslide4sense/
- IEEE Dataport: https://ieee-dataport.org/competitions/landslide4sense

---

## 🛠️ Data Preprocessing (For Your Model)

### Normalization Applied
The model applies the following normalization:

1. **Sentinel-2 Bands (1-12)**:
   ```python
   # Percentile normalization (clips outliers)
   p2, p98 = np.percentile(band, (2, 98))
   band_normalized = (band - p2) / (p98 - p2)
   ```

2. **Slope (Band 13)**:
   ```python
   # Min-max normalization
   slope_normalized = (slope - slope.min()) / (slope.max() - slope.min())
   ```

3. **DEM (Band 14)**:
   ```python
   # Min-max normalization
   dem_normalized = (dem - dem.min()) / (dem.max() - dem.min())
   ```

See `src/dataset.py` for implementation details.

---

## 🔗 Useful Links

- **Competition Page**: https://www.iarai.ac.at/landslide4sense/
- **IEEE Dataport**: https://ieee-dataport.org/competitions/landslide4sense
- **Sentinel-2 Info**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- **ALOS PALSAR Info**: https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/fnf_index.htm

---

## ❓ FAQ

### Q: Can I use a subset of the data?
**A**: Yes, the code supports loading partial data. Modify file lists in `src/dataset.py`.

### Q: Can I add my own data?
**A**: Yes, as long as it matches the format (14-channel HDF5, 128×128 pixels).

### Q: What if I don't have enough disk space?
**A**: You can download only TestData (~400 MB) for inference, skipping TrainData.

### Q: Can I convert to PNG/TIFF?
**A**: See `scripts/convert_h5_to_tiff.py` (if provided) or use `h5py` to export individual bands.

---

## 🆘 Troubleshooting

### Problem: "No .h5 files found"
**Solution**: Verify data structure matches exactly. Image files should be in `img/` subfolders.

### Problem: "FileNotFoundError: mask file not found"
**Solution**: Ensure mask files follow naming convention: `image_XXX.h5` → `mask_XXX.h5`

### Problem: "Out of memory when loading data"
**Solution**: Use smaller batch sizes in training. See `notebooks/train_colab.ipynb`.

---

**Need help?** Open an issue in the repository: [GitHub Issues](https://github.com/Oghuz20/Landslide-Detection-Satellite-AI/issues)
