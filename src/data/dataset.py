# src/data/dataset.py
import cv2
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import GPUSetup
from src.ssl import apply_masking

logger = logging.getLogger(__name__)

class MultiGraderDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, is_ssl=False, num_frames=16, hierarchical=False):
        """
        Dataset for 3D video classification with single/multi-grader support.

        Args:
            video_paths (list): List of video file paths.
            labels (list): List of labels for each grader [[grader1_label], ...] or [[grader1_label, grader2_label], ...].
                           For SSL, a dummy list (e.g., [["0"]] * len(video_paths)).
            transform (callable, optional): Transform to apply to video frames.
            is_ssl (bool): If True, returns raw video for SSL pretraining (no label processing).
            num_frames (int): Number of frames to sample per video.
            hierarchical (bool): If True, returns base/subclass labels; if False, returns mapped labels.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.is_ssl = is_ssl
        self.num_frames = num_frames
        self.hierarchical = hierarchical

        # Determine if single-grader mode
        self.single_grader = len(labels[0]) == 1 if labels else False

        # For hierarchical mode
        self.base_classes = {'1': 1, '2': 2, '3': 3, '4': 4}
        self.subclasses = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'none': 4}
        self.valid_subclasses = {
            1: ['none'],  # '1' has no subclasses
            2: ['none', 'b', 'c', 'd'],  # '2', '2b', '2c', '2d'
            3: ['none', 'b', 'c'],  # '3', '3b', '3c'
            4: ['a', 'b', 'c', 'd']  # '4a', '4b', '4c', '4d'
        }
        # For non-hierarchical mode
        self.class_names = ['1', '2', '2b', '2c', '3', '3b', '3c', '4b']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.subclasses_inverse = {v: k for k, v in self.subclasses.items()}  # For reverse mapping

        if self.hierarchical:
            self.initialize_valid_subclasses()

    def initialize_valid_subclasses(self):
        observed_subclasses = {1: set(), 2: set(), 3: set(), 4: set()}
        # Skip label parsing if labels are dummy labels (e.g., [["0"]])
        if not self.labels or self.labels[0] == ["0"]:
            return  # Dummy labels for SSL, no need to parse
        if not self.is_ssl:
            for label_pair in self.labels:
                for label in label_pair:
                    base, subclass = self.parse_label(label)
                    if subclass != 'none':
                        observed_subclasses[base].add(subclass)

    def parse_label(self, label):
        base_str = ''.join(c for c in label if c.isdigit())
        subclass_str = ''.join(c for c in label if c.isalpha()) or 'none'
        base = self.base_classes[base_str]
        subclass = self.subclasses[subclass_str]
        return base, subclass

    def map_label(self, label):
        if label == '2d':
            label = '2'
        return self.class_to_idx.get(label, 0)

    def aggregate_labels(self, label1, label2=None):
        base1, subclass1 = self.parse_label(label1)
        if label2 is None:
            base2, subclass2 = base1, subclass1
        else:
            base2, subclass2 = self.parse_label(label2)
        base_label = max(base1, base2)
        if base1 > base2:
            subclass_label = subclass1
        elif base2 > base1:
            subclass_label = subclass2
        else:
            if subclass1 == self.subclasses['none'] and subclass2 != self.subclasses['none']:
                subclass_label = subclass2
            elif subclass2 == self.subclasses['none'] and subclass1 != self.subclasses['none']:
                subclass_label = subclass1
            else:
                subclass_label = max(subclass1, subclass2)
        valid_subclass_names = self.valid_subclasses[base_label]
        if self.subclasses_inverse[subclass_label] not in valid_subclass_names:
            subclass_label = self.subclasses['none']
        valid_subclasses = [0] * len(self.subclasses)
        for subclass_name in valid_subclass_names:
            valid_subclasses[self.subclasses[subclass_name]] = 1
        return base_label, subclass_label, valid_subclasses

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        labels = self.labels[idx]

        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Sample frames
        if len(frames) < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        frames = [frames[i] for i in indices]

        # Convert to tensor
        video = np.stack(frames, axis=0)  # Shape: [T, H, W, C]
        video = torch.from_numpy(video).float() / 255.0  # [T, H, W, C]
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

        # For SSL pretraining, return raw video and labels without processing
        if self.is_ssl:
            if self.transform:
                video = torch.stack([self.transform(frame) for frame in video])
                video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
            return video, labels  # Return raw video and labels for PretrainingDataset to handle
        else:
            if self.transform:
                # video = torch.stack([self.transform(frame) for frame in video])
                # video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
                video = self.transform(video)      # [T,C,H,W]
                video = video.permute(1,0,2,3)     # [C,T,H,W]

            if self.hierarchical:
                if self.single_grader:
                    base_label, subclass_label, valid_subclasses = self.aggregate_labels(labels[0])
                else:
                    base_label, subclass_label, valid_subclasses = self.aggregate_labels(labels[0], labels[1])
                return {
                    'video': video,
                    'base_label': torch.tensor(base_label - 1),
                    'subclass_label': torch.tensor(subclass_label),
                    'valid_subclasses': torch.tensor(valid_subclasses)
                }
            else:
                if self.single_grader:
                    label = self.map_label(labels[0])
                    return video, [torch.tensor(label), torch.tensor(label)]
                else:
                    sean_label = self.map_label(labels[0])
                    santiago_label = self.map_label(labels[1])
                    return video, [torch.tensor(sean_label), torch.tensor(santiago_label)]


class PretrainingDataset(Dataset):
    def __init__(self, base_dataset, pretrain_method, cfg, ssl_transform=None):
        """
        Dataset for pretraining methods like contrastive learning or masked autoencoders.

        Args:
            base_dataset (Dataset): Base dataset (e.g., MultiGraderDataset) providing raw video data.
            pretrain_method (str): Pretraining method ('contrastive', 'moco', 'mae').
            cfg (dict): Configuration dictionary.
            ssl_transform (callable, optional): Transform function for SSL pretraining.
        """
        self.base_dataset = base_dataset
        self.pretrain_method = pretrain_method
        self.transform = ssl_transform  # Use provided transform
        self.cfg = cfg

        # MAE-specific parameters
        self.mask_ratio = cfg.get('training', {}).get('mask_ratio', 0.5)
        self.end_mask_ratio = cfg.get('training', {}).get('end_mask_ratio', 0.75)
        self.patch_size = cfg.get('training', {}).get('patch_size', 16)
        self.temporal_consistency = cfg.get('training', {}).get('temporal_consistency', 'full')
        self.change_interval = cfg.get('training', {}).get('change_interval', 5)
        self.total_epochs = cfg.get('training', {}).get('ssl_epochs', 100)
        self.current_epoch = 0
        self.current_mask_ratio = self.mask_ratio  # Initialize with initial mask_ratio

        # Validate MAE parameters
        if self.pretrain_method == 'mae':
            if not (0 <= self.mask_ratio <= 1):
                raise ValueError(f"mask_ratio must be in [0, 1], got {self.mask_ratio}")
            if not (0 <= self.end_mask_ratio <= 1):
                raise ValueError(f"end_mask_ratio must be in [0, 1], got {self.end_mask_ratio}")
            if not isinstance(self.patch_size, int) or self.patch_size <= 0:
                raise ValueError(f"patch_size must be a positive integer, got {self.patch_size}")
            if self.total_epochs <= 0:
                raise ValueError(f"total_epochs must be positive, got {self.total_epochs}")
            if not isinstance(self.change_interval, int) or self.change_interval <= 0:
                raise ValueError(f"change_interval must be a positive integer, got {self.change_interval}")
            if self.temporal_consistency not in ['full', 'partial', 'none']:
                raise ValueError(f"temporal_consistency must be 'full', 'partial', or 'none', got {self.temporal_consistency}")

    def set_epoch(self, epoch):
        """Set the current epoch and update current_mask_ratio for curriculum learning."""
        self.current_epoch = epoch
        if self.pretrain_method == 'mae':
            progress = min(self.current_epoch / (self.total_epochs - 1), 1.0) if self.total_epochs > 1 else 0.0
            self.current_mask_ratio = self.mask_ratio + progress * (self.end_mask_ratio - self.mask_ratio)
            self.current_mask_ratio = max(0.0, min(1.0, self.current_mask_ratio))
            if GPUSetup.is_main_process():
                logger.info(f"Dataset: Set epoch {epoch}, current_mask_ratio={self.current_mask_ratio:.3f}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get raw video from base dataset (no transform applied yet)
        video, _ = self.base_dataset[idx]  # Ignore labels, video shape: [T, C, H, W]
        
        # full temporal consistency:
        if self.pretrain_method == 'mae':
            # 1) apply the same augment to every frame
            #    transform_fn returns a Tensor [T, C, H, W]
            aug_clip = self.transform(video)

            # 2) reshape to [C, T, H, W] for masking
            orig = aug_clip.permute(1, 0, 2, 3)
            logger.debug(f"Video shape after permute: {orig.shape}")

            # 3) apply your masking core
            masked, mask, midxs, total = apply_masking(
                orig,
                mask_ratio=self.current_mask_ratio,  # Use precomputed value
                patch_size=self.patch_size,
                temporal_consistency=self.temporal_consistency,
                change_interval=self.change_interval,
            )
            return {
                'masked_video':    masked,   # [C,T,H,W]
                'original_video':  orig,
                'mask':            mask,
                'mask_indices':    midxs,
                'total_patches':   total
            }
        # temporal consistency:
        elif self.pretrain_method in ['contrastive', 'moco']:
            # Apply transform to the entire video stack directly
            # print(f"Debug: video shape before transform={video.shape}")
            video1 = self.transform(video)  # [T, C, H, W]
            # print(f"Debug: video1 shape after transform={video1.shape}")
            video2 = self.transform(video)  # [T, C, H, W]
            video1 = video1.permute(1, 0, 2, 3)  # [C, T, H, W]
            video2 = video2.permute(1, 0, 2, 3)  # [C, T, H, W]
            return {'video1': video1, 'video2': video2}
        
        else:
            raise ValueError(f"Unknown pretrain_method: {self.pretrain_method}")


