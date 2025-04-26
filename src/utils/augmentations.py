import albumentations as A
import torch

def get_transforms(is_train=True):
    if is_train:
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2(),
        ])
    
    def transform_fn(video):
        # video: [C, T, H, W]
        video = video.permute(1, 2, 3, 0)  # [T, H, W, C]
        transformed = []
        for frame in video:
            frame = frame.numpy()
            augmented = transform(image=frame)['image']
            transformed.append(augmented)
        return torch.stack(transformed).permute(1, 0, 2, 3)  # [C, T, H, W]
    
    return transform_fn

def get_ssl_transforms():
    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.8, brightness_limit=0.4, contrast_limit=0.4),
        A.GaussNoise(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2(),
    ])
    
    def transform_fn(video):
        # video: [C, T, H, W]
        video = video.permute(1, 2, 3, 0)  # [T, H, W, C]
        transformed = []
        for frame in video:
            frame = frame.numpy()
            augmented = transform(image=frame)['image']
            transformed.append(augmented)
        return torch.stack(transformed).permute(1, 0, 2, 3)  # [C, T, H, W]
    
    return transform_fn

# ----------------------------------------
# original before ssl

# import albumentations as A
# import torch

# def get_transforms(is_train=True):
#     if is_train:
#         transform = A.Compose([
#             A.Resize(224, 224),
#             A.HorizontalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.2),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             A.pytorch.ToTensorV2(),
#         ])
#     else:
#         transform = A.Compose([
#             A.Resize(224, 224),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             A.pytorch.ToTensorV2(),
#         ])
    
#     def transform_fn(video):
#         # video: [C, T, H, W]
#         video = video.permute(1, 2, 3, 0)  # [T, H, W, C]
#         transformed = []
#         for frame in video:
#             frame = frame.numpy()
#             augmented = transform(image=frame)['image']
#             transformed.append(augmented)
#         return torch.stack(transformed).permute(1, 0, 2, 3)  # [C, T, H, W]
    
#     return transform_fn