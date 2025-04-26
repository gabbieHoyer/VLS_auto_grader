import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from utils.io import load_video_frames

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=16, is_ssl=False):
        """
        Args:
            video_paths (list): List of paths to video files.
            labels (list): List of corresponding labels.
            transform: Augmentation transforms (from albumentations).
            num_frames (int): Number of frames to sample per video.
            is_ssl (bool): If True, return two augmented views for SSL.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        self.is_ssl = is_ssl

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Load video frames
        frames = load_video_frames(self.video_paths[idx], num_frames=self.num_frames)
        
        # Convert frames to tensor: [T, H, W, C] -> [C, T, H, W]
        frames = np.transpose(frames, (3, 0, 1, 2)).astype(np.float32)
        frames = torch.from_numpy(frames)
        
        if self.is_ssl:
            # Return two augmented views for SSL
            view1 = self.transform(frames)
            view2 = self.transform(frames)
            return view1, view2
        else:
            # Apply augmentations
            if self.transform:
                frames = self.transform(frames)
            
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return frames, label

def get_dataloader(video_paths, labels, transform, batch_size, num_workers, shuffle=True, is_ssl=False):
    dataset = VideoDataset(video_paths, labels, transform, is_ssl=is_ssl)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# -------------------------------------------------
#  original before ssl

# import torch
# from torch.utils.data import Dataset
# import cv2
# import numpy as np
# from src.utils.io import load_video_frames

# class VideoDataset(Dataset):
#     def __init__(self, video_paths, labels, transform=None, num_frames=16):
#         """
#         Args:
#             video_paths (list): List of paths to video files.
#             labels (list): List of corresponding labels.
#             transform: Augmentation transforms (from albumentations).
#             num_frames (int): Number of frames to sample per video.
#         """
#         self.video_paths = video_paths
#         self.labels = labels
#         self.transform = transform
#         self.num_frames = num_frames

#     def __len__(self):
#         return len(self.video_paths)

#     def __getitem__(self, idx):
#         # Load video frames
#         frames = load_video_frames(self.video_paths[idx], num_frames=self.num_frames)
        
#         # Convert frames to tensor: [T, H, W, C] -> [C, T, H, W]
#         frames = np.transpose(frames, (3, 0, 1, 2)).astype(np.float32)
#         frames = torch.from_numpy(frames)
        
#         # Apply augmentations
#         if self.transform:
#             frames = self.transform(frames)
        
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
        
#         return frames, label

# def get_dataloader(video_paths, labels, transform, batch_size, num_workers, shuffle=True):
#     dataset = VideoDataset(video_paths, labels, transform)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#     return dataloader