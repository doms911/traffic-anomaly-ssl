"""MoCo v2 augmentation pipeline for SSL pretraining."""
import torchvision.transforms as T

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


def build_moco_v2_transforms(crop_size: int = 224):
    return T.Compose([
        T.RandomResizedCrop(crop_size, scale=(0.08, 1.0)),   # bilo (0.2, 1.0) — agresivniji crop
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])