"""
Train: geometric + mild intensity augmentations.
Valid/Test: no augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.4
        ),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            p=0.2
        ),
        ToTensorV2()
    ])


def get_valid_transforms():
    return A.Compose([ToTensorV2()])