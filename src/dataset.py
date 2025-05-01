import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from utils.augmentations import get_ssl_transforms
from utils.ssl_helpers import apply_masking

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
                video = torch.stack([self.transform(frame) for frame in video])
                video = video.permute(1, 0, 2, 3)  # [C, T, H, W]

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
        self.mask_ratio = cfg.get('training', {}).get('mask_ratio', 0.75)  # Default mask ratio for MAE

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get raw video from base dataset (no transform applied yet)
        video, _ = self.base_dataset[idx]  # Ignore labels, video shape: [T, C, H, W]

        if self.pretrain_method == 'mae':
            # transforms → [C,T,H,W]
            orig = torch.stack([self.transform(f) for f in video]).permute(1,0,2,3)
            masked, mask, midxs, total = apply_masking(orig, self.mask_ratio, patch_size=16)
            return {
                'masked_video':   masked,     # [C,T,H,W]
                'original_video': orig,       # [C,T,H,W]
                'mask':           mask,       # [T,H,W]
                'mask_indices':   midxs,      # [num_masked]
                'total_patches':  total       # int
            }
        
        elif self.pretrain_method in ['contrastive', 'moco']:
            # Apply transforms to create two augmented views
            video1 = torch.stack([self.transform(frame) for frame in video])
            video2 = torch.stack([self.transform(frame) for frame in video])
            video1 = video1.permute(1, 0, 2, 3)  # [C, T, H, W]
            video2 = video2.permute(1, 0, 2, 3)  # [C, T, H, W]
            return {'video1': video1, 'video2': video2}
        
        else:
            raise ValueError(f"Unknown pretrain_method: {self.pretrain_method}")



# ---------------------------------------------------------------------
        # elif self.pretrain_method == 'mae':
        #     # Apply transform to the original video
        #     original_video = torch.stack([self.transform(frame) for frame in video])
        #     original_video = original_video.permute(1, 0, 2, 3)  # [C, T, H, W]
            
        #     # Apply masking to create the masked video
        #     masked_video, mask = apply_masking(original_video, mask_ratio=self.mask_ratio, patch_size=16)
        #     return {
        #         'masked_video': masked_video,
        #         'original_video': original_video,  # Needed for reconstruction loss
        #         'mask': mask
        #     }
# ----------------------------------------------------------------

# import torch
# from torch.utils.data import Dataset
# import cv2
# import numpy as np
# import albumentations as A

# # from utils.augmentations import get_contrastive_transforms, get_mae_transforms, apply_masking

# class MultiGraderDataset(Dataset):
#     def __init__(self, video_paths, labels, transform=None, is_ssl=False, num_frames=16, hierarchical=False):
#         """
#         Dataset for 3D video classification with single/multi-grader support.

#         Args:
#             video_paths (list): List of video file paths.
#             labels (list): List of labels for each grader [[grader1_label], ...] or [[grader1_label, grader2_label], ...].
#                            For SSL, a dummy list (e.g., [[0]] * len(video_paths)).
#             transform (callable, optional): Transform to apply to video frames.
#             is_ssl (bool): If True, returns video pairs for SSL pretraining.
#             num_frames (int): Number of frames to sample per video.
#             hierarchical (bool): If True, returns base/subclass labels; if False, returns mapped labels.
#         """
#         self.video_paths = video_paths
#         self.labels = labels
#         self.transform = transform
#         self.is_ssl = is_ssl
#         self.num_frames = num_frames
#         self.hierarchical = hierarchical

#         # Determine if single-grader mode
#         self.single_grader = len(labels[0]) == 1 if labels else False

#         # For hierarchical mode
#         self.base_classes = {'1': 1, '2': 2, '3': 3, '4': 4}
#         self.subclasses = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'none': 4}
#         self.valid_subclasses = {
#             1: ['none'],  # '1' has no subclasses
#             2: ['none', 'b', 'c', 'd'],  # '2', '2b', '2c', '2d'
#             3: ['none', 'b', 'c'],  # '3', '3b', '3c'
#             4: ['a', 'b', 'c', 'd']  # '4a', '4b', '4c', '4d'
#         }
#         # For non-hierarchical mode
#         self.class_names = ['1', '2', '2b', '2c', '3', '3b', '3c', '4b']
#         self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
#         self.subclasses_inverse = {v: k for k, v in self.subclasses.items()}  # For reverse mapping

#         if self.hierarchical:
#             self.initialize_valid_subclasses()

#     # def initialize_valid_subclasses(self):
#     #     observed_subclasses = {1: set(), 2: set(), 3: set(), 4: set()}
#     #     if not self.is_ssl:
#     #         for label_pair in self.labels:
#     #             for label in label_pair:
#     #                 base, subclass = self.parse_label(label)
#     #                 if subclass != 'none':
#     #                     observed_subclasses[base].add(subclass)

#     def initialize_valid_subclasses(self):
#         observed_subclasses = {1: set(), 2: set(), 3: set(), 4: set()}
#         # Skip label parsing if labels are dummy labels (e.g., [["0"]])
#         if not self.labels or self.labels[0] == ["0"]:
#             return  # Dummy labels for SSL, no need to parse
#         if not self.is_ssl:
#             for label_pair in self.labels:
#                 for label in label_pair:
#                     base, subclass = self.parse_label(label)
#                     if subclass != 'none':
#                         observed_subclasses[base].add(subclass)

#     def parse_label(self, label):
#         base_str = ''.join(c for c in label if c.isdigit())
#         subclass_str = ''.join(c for c in label if c.isalpha()) or 'none'
#         base = self.base_classes[base_str]
#         subclass = self.subclasses[subclass_str]
#         return base, subclass

#     def map_label(self, label):
#         if label == '2d':
#             label = '2'
#         return self.class_to_idx.get(label, 0)

#     def aggregate_labels(self, label1, label2=None):
#         base1, subclass1 = self.parse_label(label1)
#         if label2 is None:
#             base2, subclass2 = base1, subclass1
#         else:
#             base2, subclass2 = self.parse_label(label2)
#         base_label = max(base1, base2)
#         if base1 > base2:
#             subclass_label = subclass1
#         elif base2 > base1:
#             subclass_label = subclass2
#         else:
#             if subclass1 == self.subclasses['none'] and subclass2 != self.subclasses['none']:
#                 subclass_label = subclass2
#             elif subclass2 == self.subclasses['none'] and subclass1 != self.subclasses['none']:
#                 subclass_label = subclass1
#             else:
#                 subclass_label = max(subclass1, subclass2)
#         valid_subclass_names = self.valid_subclasses[base_label]
#         if self.subclasses_inverse[subclass_label] not in valid_subclass_names:
#             subclass_label = self.subclasses['none']
#         valid_subclasses = [0] * len(self.subclasses)
#         for subclass_name in valid_subclass_names:
#             valid_subclasses[self.subclasses[subclass_name]] = 1
#         return base_label, subclass_label, valid_subclasses

#     def __len__(self):
#         return len(self.video_paths)

#     def __getitem__(self, idx):
#         video_path = self.video_paths[idx]
#         labels = self.labels[idx]

#         # Load video
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         cap.release()

#         # Sample frames
#         if len(frames) < self.num_frames:
#             frames = frames + [frames[-1]] * (self.num_frames - len(frames))
#         indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
#         frames = [frames[i] for i in indices]

#         # Convert to tensor
#         video = np.stack(frames, axis=0)  # Shape: [T, H, W, C]
#         video = torch.from_numpy(video).float() / 255.0  # [T, H, W, C]
#         video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

#         if self.is_ssl:
#             if self.transform:
#                 video1 = torch.stack([self.transform(frame) for frame in video])
#                 video2 = torch.stack([self.transform(frame) for frame in video])
#                 video1 = video1.permute(1, 0, 2, 3)  # [C, T, H, W]
#                 video2 = video2.permute(1, 0, 2, 3)  # [C, T, H, W]
#             return video1, video2, labels
#         else:
#             if self.transform:
#                 video = torch.stack([self.transform(frame) for frame in video])
#                 video = video.permute(1, 0, 2, 3)  # [C, T, H, W]

#             if self.hierarchical:
#                 if self.single_grader:
#                     base_label, subclass_label, valid_subclasses = self.aggregate_labels(labels[0])
#                 else:
#                     base_label, subclass_label, valid_subclasses = self.aggregate_labels(labels[0], labels[1])
#                 return {
#                     'video': video,
#                     'base_label': torch.tensor(base_label - 1),
#                     'subclass_label': torch.tensor(subclass_label),
#                     'valid_subclasses': torch.tensor(valid_subclasses)
#                 }
#             else:
#                 if self.single_grader:
#                     label = self.map_label(labels[0])
#                     return video, [torch.tensor(label), torch.tensor(label)]
#                 else:
#                     sean_label = self.map_label(labels[0])
#                     santiago_label = self.map_label(labels[1])
#                     return video, [torch.tensor(sean_label), torch.tensor(santiago_label)]


# def apply_masking(video, mask_ratio=0.75, patch_size=16):
#     """
#     Apply tube masking to a video tensor for MAE pretraining.

#     Args:
#         video (torch.Tensor): Video tensor of shape [C, T, H, W].
#         mask_ratio (float): Fraction of patches to mask.
#         patch_size (int): Size of spatial patches (assumes square patches).

#     Returns:
#         tuple: (masked_video, mask)
#             - masked_video (torch.Tensor): Video with masked patches set to 0, shape [C, T, H, W].
#             - mask (torch.Tensor): Binary mask, 1 for masked patches, 0 for kept patches, shape [T, H, W].
#     """
#     C, T, H, W = video.shape
#     assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"

#     # Compute number of patches
#     num_patches_t = T  # One patch per frame (temporal dimension)
#     num_patches_h = H // patch_size
#     num_patches_w = W // patch_size
#     total_patches = num_patches_t * num_patches_h * num_patches_w

#     # Determine number of patches to mask
#     num_mask = int(mask_ratio * total_patches)

#     # Create patch indices and shuffle
#     patch_indices = np.arange(total_patches)
#     np.random.shuffle(patch_indices)

#     # Select patches to mask
#     mask_indices = patch_indices[:num_mask]
#     keep_indices = patch_indices[num_mask:]

#     # Create mask: 1 for masked patches, 0 for kept patches
#     mask = torch.zeros(T, H, W, dtype=torch.float32)
#     for idx in mask_indices:
#         t = idx // (num_patches_h * num_patches_w)
#         h_idx = (idx % (num_patches_h * num_patches_w)) // num_patches_w
#         w_idx = (idx % (num_patches_h * num_patches_w)) % num_patches_w
#         h_start = h_idx * patch_size
#         w_start = w_idx * patch_size
#         mask[t, h_start:h_start + patch_size, w_start:w_start + patch_size] = 1

#     # Apply mask to video
#     masked_video = video.clone()
#     mask_expanded = mask.unsqueeze(0)  # [1, T, H, W]
#     masked_video = masked_video * (1 - mask_expanded)  # Set masked patches to 0

#     return masked_video, mask

# class PretrainingDataset(Dataset):
#     def __init__(self, base_dataset, pretrain_method, cfg, ssl_transform=None):
#         """
#         Dataset for pretraining methods like contrastive learning or masked autoencoders.

#         Args:
#             base_dataset (Dataset): Base dataset (e.g., MultiGraderDataset) providing raw video data.
#             pretrain_method (str): Pretraining method ('contrastive', 'moco', 'mae').
#             cfg (dict): Configuration dictionary.
#         """
#         self.base_dataset = base_dataset
#         self.pretrain_method = pretrain_method
#         self.transform = ssl_transform #get_ssl_transforms(pretrain_method)
#         self.mask_ratio = cfg.get('training', {}).get('mask_ratio', 0.75)  # Default mask ratio for MAE

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         # Get raw video from base dataset (no transform applied yet)
#         video, _ = self.base_dataset[idx]  # Ignore labels, video shape: [T, C, H, W]

#         if self.pretrain_method in ['contrastive', 'moco']:
#             # Apply transforms to create two augmented views
#             video1 = torch.stack([self.transform(frame) for frame in video])
#             video2 = torch.stack([self.transform(frame) for frame in video])
#             video1 = video1.permute(1, 0, 2, 3)  # [C, T, H, W]
#             video2 = video2.permute(1, 0, 2, 3)  # [C, T, H, W]
#             return {'video1': video1, 'video2': video2}
#         elif self.pretrain_method == 'mae':
#             # Apply transform to the original video
#             original_video = torch.stack([self.transform(frame) for frame in video])
#             original_video = original_video.permute(1, 0, 2, 3)  # [C, T, H, W]
            
#             # Apply masking to create the masked video
#             masked_video, mask = apply_masking(original_video, mask_ratio=self.mask_ratio, patch_size=16)
#             return {
#                 'masked_video': masked_video,
#                 'original_video': original_video,  # Needed for reconstruction loss
#                 'mask': mask
#             }
#         else:
#             raise ValueError(f"Unknown pretrain_method: {self.pretrain_method}")



# --------------------------------------------------------------

# class PretrainingDataset(Dataset):
#     def __init__(self, base_dataset, pretrain_method, cfg):
#         """
#         Dataset for pretraining methods like contrastive learning or masked autoencoders.

#         Args:
#             base_dataset (Dataset): Base dataset (e.g., MultiGraderDataset) providing raw video data.
#             pretrain_method (str): Pretraining method ('contrastive' or 'mae').
#             cfg (dict): Configuration dictionary.
#         """
#         self.base_dataset = base_dataset
#         self.pretrain_method = pretrain_method
#         if pretrain_method == 'contrastive':
#             self.transform = get_contrastive_transforms()
#         elif pretrain_method == 'mae':
#             self.transform = get_mae_transforms()
#         else:
#             raise ValueError(f"Unknown pretrain_method: {pretrain_method}")

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         video, _ = self.base_dataset[idx]  # Ignore labels
#         if self.pretrain_method == 'contrastive':
#             video1 = torch.stack([self.transform(frame) for frame in video])
#             video2 = torch.stack([self.transform(frame) for frame in video])
#             video1 = video1.permute(1, 0, 2, 3)  # [C, T, H, W]
#             video2 = video2.permute(1, 0, 2, 3)  # [C, T, H, W]
#             return {'video1': video1, 'video2': video2}
#         elif self.pretrain_method == 'mae':
#             video = torch.stack([self.transform(frame) for frame in video])
#             video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
#             masked_video, mask = apply_masking(video)
#             return {'masked_video': masked_video, 'mask': mask}