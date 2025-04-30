import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

def get_transforms(is_train=True):
    """
    Get transforms for 3D video data (supervised training or validation).

    Args:
        is_train (bool): If True, applies training augmentations; else, validation augmentations.

    Returns:
        callable: Function that transforms a video frame tensor [C, H, W] to [C, H, W].
    """
    if is_train:
        transform = A.Compose([
            A.RandomCrop(height=224, width=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform_fn(frame):
        # frame: [C, H, W]
        frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        augmented = transform(image=frame)
        return torch.from_numpy(augmented['image']).permute(2, 0, 1)  # [C, H, W]

    return transform_fn

def get_ssl_transforms(pretrain_method):
    """
    Get augmentations for SSL pretraining based on the specified method.

    Args:
        pretrain_method (str): Pretraining method ('contrastive', 'moco', 'mae').

    Returns:
        callable: Function that transforms a video frame tensor [C, H, W] to [C, H, W].
    """
    if pretrain_method in ['contrastive', 'moco']:
        # Strong augmentations for contrastive learning (SimCLR, MoCo)
        transform = A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif pretrain_method == 'mae':
        # Minimal augmentations for MAE to preserve reconstruction target
        transform = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),  # Light augmentation, applied consistently across frames
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unknown pretrain_method: {pretrain_method}")
    
    # elif pretrain_method == 'byol':
    #     transform = A.Compose([...])  # Similar to contrastive/MoCo
    # elif pretrain_method == 'temporal_contrastive':
    #     transform = A.Compose([...])  # Add temporal augmentations like frame jittering

    def transform_fn(frame):
        # frame: [C, H, W]
        frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        augmented = transform(image=frame)
        return torch.from_numpy(augmented['image']).permute(2, 0, 1)  # [C, H, W]

    return transform_fn