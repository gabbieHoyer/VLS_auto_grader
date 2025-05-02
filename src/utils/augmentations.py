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
            # A.RandomCrop(height=224, width=224, p=1.0),
            A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.07, p=0.7),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.6),
            ], p=1),

            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            A.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225],
                max_pixel_value=1.0
            )
        ])
    else:
        transform = A.Compose([
            A.Resize(height=224, width=224),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            A.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225],
                max_pixel_value=1.0
            )
        ])

    def transform_fn(frame):
        # frame: [C, H, W]
        frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        augmented = transform(image=frame)
        return torch.from_numpy(augmented['image']).permute(2, 0, 1)  # [C, H, W]

    return transform_fn

# ---------------------------------------------------------------------------- #

def get_ssl_transforms(pretrain_method):
    # define your augment list...
    if pretrain_method in ['contrastive', 'moco']:
        aug_list = [
            A.RandomResizedCrop(224, 224, scale=(0.5,1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.07, p=0.7),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.6),
            ], p=1),
            # A.GaussNoise(var_limit=(1.0,3.0), p=0.05),
            A.GaussNoise(var_limit=(0.1, 0.3), p=0.05),
            A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225], max_pixel_value=1.0),
        ]
    elif pretrain_method == 'mae':
        aug_list = [
            A.Resize(224,224),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.07, p=0.7),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.6),
            ], p=1),

            A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225], max_pixel_value=1.0),
        ]
    else:
        raise ValueError(pretrain_method)

    # build one ReplayCompose and reuse it
    rc = A.ReplayCompose(aug_list)

    def transform_fn(frames: torch.Tensor):
        # frames: [T,C,H,W]
        npv = frames.permute(0,2,3,1).numpy()  # [T,H,W,C]
        # record params on first frame
        out0 = rc(image=npv[0])
        replay_dict = out0['replay']
        if replay_dict is None:
            raise RuntimeError("ReplayCompose failed to record augmentations")

        # apply same aug to all
        augmented = []
        for f in npv:
            res = rc.replay(replay_dict, image=f)  # <— positional arg here!
            augmented.append(res['image'])

        # back to [T,C,H,W]
        return torch.stack([torch.from_numpy(x).permute(2,0,1) for x in augmented])

    return transform_fn


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

            # # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),  # not sure
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5), # purple ish 

            # A.HueSaturationValue(hue_shift_limit=(-15,15), sat_shift_limit=0, val_shift_limit=0, p=0.5),
            # A.RGBShift(r_shift_limit=(10,25), g_shift_limit=0, b_shift_limit=0, p=0.3),  # Gentle pink→red boost without yanking blues too far

            # very mild red shift on pinks, tiny saturation bump
            # A.ColorJitter(
            #     brightness=0.1, 
            #     contrast=0.1, 
            #     saturation=0.15, 
            #     hue=0.05,   # only ±0.05 on hue (≈±18°)
            #     p=0.5
            # ),


            # # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),  # purple ish
            
            # A.HueSaturationValue(hue_shift_limit=(-15,15), sat_shift_limit=0, val_shift_limit=0, p=0.5),
            # # Gentle pink→red boost without yanking blues too far
            # A.RGBShift(r_shift_limit=(10,25), g_shift_limit=0, b_shift_limit=0),
            
            # # very mild red shift on pinks, tiny saturation bump
            # A.ColorJitter(
            #     brightness=0.1, 
            #     contrast=0.1, 
            #     saturation=0.15, 
            #     hue=0.05,   # only ±0.05 on hue (≈±18°)
            #     p=0.5
            # ),

# Not temporal consistency:
# def get_ssl_transforms(pretrain_method):
#     """
#     Get augmentations for SSL pretraining based on the specified method.

#     Args:
#         pretrain_method (str): Pretraining method ('contrastive', 'moco', 'mae').

#     Returns:
#         callable: Function that transforms a video frame tensor [C, H, W] to [C, H, W].
#     """
#     if pretrain_method in ['contrastive', 'moco']:
#         # Strong augmentations for contrastive learning (SimCLR, MoCo)
#         transform = A.Compose([
#             # A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), p=1.0),
#             # A.HorizontalFlip(p=0.5),
#             # A.Rotate(limit=30, p=0.5),
#             # A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
#             # A.GaussNoise(var_limit=(5.0, 10.0), p=0.1),  # Reduced noise
#             # # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
#             # A.RandomGamma(gamma_limit=(80, 120), p=0.2),
#             # # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), p=1.0),  # Slightly larger crops
#             A.HorizontalFlip(p=0.5),
#             A.Rotate(limit=15, p=0.5),  # Reduced rotation angle
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),  # Reduced magnitude
#             A.GaussNoise(var_limit=(2.0, 5.0), p=0.05),  # Further reduced noise and probability
#             A.RandomGamma(gamma_limit=(90, 110), p=0.1),  # Narrower gamma range, lower probability

#             A.Normalize(
#                     mean=[0.485,0.456,0.406],
#                     std=[0.229,0.224,0.225],
#                     max_pixel_value=1.0
#             )
#         ])
#     elif pretrain_method == 'mae':
#         # Minimal augmentations for MAE to preserve reconstruction target
#         transform = A.Compose([
#             A.Resize(height=224, width=224),
#             A.HorizontalFlip(p=0.5),  # Light augmentation, applied consistently across frames
#             A.Normalize(
#                     mean=[0.485,0.456,0.406],
#                     std=[0.229,0.224,0.225],
#                     max_pixel_value=1.0
#                 )
#             # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#     else:
#         raise ValueError(f"Unknown pretrain_method: {pretrain_method}")
    
#     # elif pretrain_method == 'byol':
#     #     transform = A.Compose([...])  # Similar to contrastive/MoCo
#     # elif pretrain_method == 'temporal_contrastive':
#     #     transform = A.Compose([...])  # Add temporal augmentations like frame jittering

#     def transform_fn(frame):
#         # frame: [C, H, W]
#         frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
#         augmented = transform(image=frame)
#         return torch.from_numpy(augmented['image']).permute(2, 0, 1)  # [C, H, W]

#     return transform_fn



# Temoral consistency - apply it to the whole video/stack of frames:
# def get_ssl_transforms(pretrain_method):
#     if pretrain_method in ['contrastive', 'moco']:
#         transform = A.Compose([
#             A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), p=1.0),  # Wider crop range
#             A.HorizontalFlip(p=0.5),
#             A.Rotate(limit=30, p=0.5),  # Increased rotation
#             A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),  # Stronger color jitter
#             A.GaussNoise(var_limit=(1.0, 3.0), p=0.05), #0.03  # Reduced noise intensity and probability
#             # A.RandomGamma(gamma_limit=(95, 105), p=0.05),  # Narrower gamma range, lower probability
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)
#         ])

        # def transform_fn(frames):
        #     # frames: [T, C, H, W] -> [T, H, W, C] for Albumentations
        #     frames_np = frames.permute(0, 2, 3, 1).numpy()
        #     # Apply the same augmentation to all frames
        #     augmented = [transform(image=frame)['image'] for frame in frames_np]
        #     return torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented])

        # return transform_fn

        # def transform_fn(frames):
        #     # frames: [T, C, H, W] -> [T, H, W, C] for Albumentations
        #     frames_np = frames.permute(0, 2, 3, 1).numpy()
        #     # Apply transform to the first frame to determine the parameters
        #     first_frame = frames_np[0]  # [H, W, C]
        #     augmented_first = transform(image=first_frame)
        #     # Extract the transformation parameters
        #     transform_params = augmented_first.get('replay', None)
        #     if transform_params is None:
        #         # If replay is not available, apply the same transform to all frames manually
        #         augmented = [transform(image=frame)['image'] for frame in frames_np]
        #     else:
        #         # Replay the same transformation on all frames
        #         replay_transform = A.ReplayCompose(transform.transforms)
        #         augmented = []
        #         for frame in frames_np:
        #             result = replay_transform(image=frame, replay=transform_params)
        #             augmented.append(result['image'])
        #     return torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented])

        # return transform_fn

    #     def transform_fn(frames):
    #         # frames: [T, C, H, W] -> [T, H, W, C] for Albumentations
    #         frames_np = frames.permute(0, 2, 3, 1).numpy()
    #         # Apply transform to the first frame to determine the parameters
    #         first_frame = frames_np[0]  # [H, W, C]
    #         augmented_first = transform(image=first_frame)
    #         # Extract the transformation parameters
    #         transform_params = augmented_first.get('replay', None)
    #         print(f"Debug: Replay params available={transform_params is not None}")
    #         if transform_params is None:
    #             print("Warning: Replay not supported, falling back to independent transforms (inconsistent)")
    #             # Fallback: Apply transform independently (not temporally consistent)
    #             augmented = [transform(image=frame)['image'] for frame in frames_np]
    #         else:
    #             # Replay the same transformation on all frames
    #             replay_transform = A.ReplayCompose(transform.transforms)
    #             augmented = []
    #             for frame in frames_np:
    #                 result = replay_transform(image=frame, replay=transform_params)
    #                 augmented.append(result['image'])
    #                 # Debug: Verify the first and a sample frame
    #                 if frame is frames_np[0]:
    #                     print(f"Debug: First frame transform params={transform_params}")
    #                 elif frame is frames_np[1]:
    #                     print(f"Debug: Second frame transform params={result.get('replay', 'Not recorded')}")
    #         return torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented])

    #     return transform_fn
    
    # elif pretrain_method == 'mae':
    #     # Minimal augmentations for MAE
    #     transform = A.Compose([
    #         A.Resize(height=224, width=224),
    #         A.HorizontalFlip(p=0.5),
    #         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)
    #     ])
    #     def transform_fn(frame):
    #         # frame: [C, H, W] -> [H, W, C] for Albumentations
    #         frame_np = frame.permute(1, 2, 0).numpy()
    #         augmented = transform(image=frame_np)
    #         return torch.from_numpy(augmented['image']).permute(2, 0, 1)

    #     return transform_fn
    # else:
    #     raise ValueError(f"Unknown pretrain_method: {pretrain_method}")
    

# def get_ssl_transforms(pretrain_method):
#     if pretrain_method in ['contrastive', 'moco']:
#         # transform = A.Compose([
#         #     A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), p=1.0),
#         #     A.HorizontalFlip(p=0.5),
#         #     A.Rotate(limit=15, p=0.5),
#         #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
#         #     A.GaussNoise(var_limit=(2.0, 5.0), p=0.05),
#         #     A.RandomGamma(gamma_limit=(90, 110), p=0.1),
#         #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)
#         # ])
#         transform = A.Compose([
#             A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), p=1.0),  # Wider crop range
#             A.HorizontalFlip(p=0.5),
#             A.Rotate(limit=30, p=0.5),  # Increased rotation
#             A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),  # Stronger color jitter
#             A.GaussNoise(var_limit=(1.0, 3.0), p=0.03),  # Reduced noise intensity and probability
#             A.RandomGamma(gamma_limit=(95, 105), p=0.05),  # Narrower gamma range, lower probability
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)
#         ])

#         def transform_fn(frames):
#             # frames: [T, C, H, W] -> [T, H, W, C] for Albumentations
#             frames_np = frames.permute(0, 2, 3, 1).numpy()
#             # Apply the same augmentation to all frames
#             augmented = [transform(image=frame)['image'] for frame in frames_np]
#             return torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented])

#         return transform_fn
#     elif pretrain_method == 'mae':
#         # Minimal augmentations for MAE
#         transform = A.Compose([
#             A.Resize(height=224, width=224),
#             A.HorizontalFlip(p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0)
#         ])

#         # def transform_fn(frames):
#         #     frames_np = frames.permute(0, 2, 3, 1).numpy()
#         #     augmented = [transform(image=frame)['image'] for frame in frames_np]
#         #     return torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in augmented])
        
#         def transform_fn(frame):
#             # frame: [C, H, W] -> [H, W, C] for Albumentations
#             frame_np = frame.permute(1, 2, 0).numpy()
#             augmented = transform(image=frame_np)
#             return torch.from_numpy(augmented['image']).permute(2, 0, 1)

#         return transform_fn
#     else:
#         raise ValueError(f"Unknown pretrain_method: {pretrain_method}")
# ---------------------------------------------------------------------------- #
