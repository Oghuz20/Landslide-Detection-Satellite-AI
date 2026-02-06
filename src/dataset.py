"""
Landslide4Sense Dataset Loader
Handles multi-spectral satellite imagery with masks
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LandslideDataset(Dataset):
    """
    Dataset for landslide detection from satellite imagery
    
    Structure:
        - Image: 128x128x14 (12 Sentinel-2 bands + slope + DEM)
        - Mask: 128x128 binary (0=no landslide, 1=landslide)
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Root directory containing data/
            split (str): 'train', 'valid', or 'test'
            transform (albumentations.Compose): Augmentation pipeline
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Set paths based on split
        if split == 'train':
            self.img_dir = os.path.join(data_dir, 'TrainData', 'img')
            self.mask_dir = os.path.join(data_dir, 'TrainData', 'mask')
        elif split == 'valid':
            self.img_dir = os.path.join(data_dir, 'ValidData', 'img')
            self.mask_dir = os.path.join(data_dir, 'ValidData', 'mask')
        else:  # test
            self.img_dir = os.path.join(data_dir, 'TestData', 'img')
            self.mask_dir = os.path.join(data_dir, 'TestData', 'mask')
        
        # Verify directories exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        # Get file list
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.h5')])
        
        if len(self.img_files) == 0:
            raise ValueError(f"No .h5 files found in {self.img_dir}")
        
        print(f"✓ Loaded {len(self.img_files)} images from {split} split")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Load and return a single sample
        
        Returns:
            dict: {
                'image': torch.Tensor [14, 128, 128],
                'mask': torch.Tensor [128, 128],
                'filename': str
            }
        """
        # Load image (128x128x14)
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        with h5py.File(img_path, 'r') as f:
            image = f['img'][:].astype(np.float32)
        
        # Normalize to [0, 1]
        image = self.normalize_image(image)
        
        # Load mask (128x128)
        mask_file = self.img_files[idx].replace('image', 'mask')
        mask_path = os.path.join(self.mask_dir, mask_file)
        with h5py.File(mask_path, 'r') as f:
            mask = f['mask'][:]
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return {
            'image': image,      # [14, 128, 128]
            'mask': mask,        # [128, 128]
            'filename': self.img_files[idx]
        }
    
    def normalize_image(self, image):
        """
        Normalize multi-spectral satellite image
        
        Bands 0-11: Sentinel-2 spectral bands (percentile normalization)
        Band 12: Slope (min-max normalization)
        Band 13: DEM (min-max normalization)
        
        Args:
            image: np.array [128, 128, 14]
            
        Returns:
            Normalized image [128, 128, 14] in range [0, 1]
        """
        # Sentinel-2 bands: Use robust percentile normalization
        for i in range(12):
            band = image[:, :, i]
            p2, p98 = np.percentile(band, (2, 98))  # Clip outliers
            band = np.clip(band, p2, p98)
            if p98 - p2 > 0:
                image[:, :, i] = (band - p2) / (p98 - p2)
        
        # Slope: Min-max normalization
        slope = image[:, :, 12]
        if slope.max() > slope.min():
            image[:, :, 12] = (slope - slope.min()) / (slope.max() - slope.min())
        
        # DEM: Min-max normalization
        dem = image[:, :, 13]
        if dem.max() > dem.min():
            image[:, :, 13] = (dem - dem.min()) / (dem.max() - dem.min())
        
        return image


def get_train_transforms():
    """
    Training augmentation pipeline
    
    Includes:
        - Geometric: flips, rotations, translations
        - Noise: gaussian noise, blur
        - Intensity: brightness/contrast changes
        - Deformation: elastic transforms
    """
    return A.Compose([
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-45, 45),
            p=0.6
        ),
        
        # Noise augmentations
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.4),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.4
        ),
        
        # Elastic deformation
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.3
        ),
        
        ToTensorV2()
    ])


def get_valid_transforms():
    """
    Validation/test transforms (no augmentation)
    Just convert to tensor
    """
    return A.Compose([
        ToTensorV2()
    ])