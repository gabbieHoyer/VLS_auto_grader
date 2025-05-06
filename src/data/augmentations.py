# src/data/dataset.py
import torch
import albumentations as A

def get_transforms(is_train=True):
    if is_train:
        aug_list = [
            A.RandomResizedCrop(224, 224, scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.OneOf([
              A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.07, p=0.7),
              A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,  p=0.6),
            ], p=1.0),
            A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225], max_pixel_value=1.0),
        ]
    else:
        aug_list = [
            A.Resize(224, 224),
            A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225], max_pixel_value=1.0),
        ]

    rc = A.ReplayCompose(aug_list)

    def transform_fn(clip: torch.Tensor):
        # clip: [T, C, H, W]
        np_clip = clip.permute(0,2,3,1).numpy()  # [T, H, W, C]
        # record the random params on the first frame
        out0 = rc(image=np_clip[0])
        replay = out0['replay']
        augmented = [ out0['image'] ]
        # replay exactly the same ops on all other frames
        for frame in np_clip[1:]:
            res = rc.replay(replay, image=frame)
            augmented.append(res['image'])
        # back to torch [T,C,H,W]
        return torch.stack([
            torch.from_numpy(f).permute(2,0,1)
            for f in augmented
        ])

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



