from albumentations import Compose, Resize, RandomResizedCrop, HorizontalFlip, VerticalFlip, Rotate, ColorJitter, GaussianBlur, Normalize
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size: int, train: bool):
    if train:
        return Compose([
            RandomResizedCrop(size=(image_size, image_size), scale=(0.80, 1.0)),
            HorizontalFlip(p=0.5), VerticalFlip(p=0.5), Rotate(limit=20, p=0.5),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05, p=0.5),
            GaussianBlur(blur_limit=(3, 5), p=0.2),
            Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
        ])
    return Compose([Resize(image_size, image_size), Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
