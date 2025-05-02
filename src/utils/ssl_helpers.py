import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from collections import deque
import logging

logger = logging.getLogger(__name__)


# def apply_masking(video, mask_ratio=0.75, patch_size=16):
#     """
#     Now returns:
#       masked_video: [C,T,H,W]
#       mask:         [T,H,W]
#       mask_indices: long tensor of shape [num_masked]
#       total_patches (int)
#     """
#     C, T, H, W = video.shape
#     p = patch_size
#     n_h, n_w = H // p, W // p
#     total_patches = T * n_h * n_w

#     # pick which patches to mask
#     all_idxs = np.arange(total_patches)
#     np.random.shuffle(all_idxs)
#     num_mask = int(mask_ratio * total_patches)
#     mask_indices = all_idxs[:num_mask]

#     # build the binary mask map
#     mask = torch.zeros(T, H, W, dtype=torch.float32)
#     for idx in mask_indices:
#         t = idx // (n_h * n_w)
#         rem = idx % (n_h * n_w)
#         h_idx, w_idx = divmod(rem, n_w)
#         mask[t,
#              h_idx*p:(h_idx+1)*p,
#              w_idx*p:(w_idx+1)*p] = 1

#     # apply it
#     masked_video = video * (1.0 - mask.unsqueeze(0))  # broadcast C dim

#     return masked_video, mask, torch.tensor(mask_indices, dtype=torch.long), total_patches


import numpy as np
import torch

def apply_masking(video, mask_ratio=0.5, patch_size=16, temporal_consistency='full', change_interval=5, current_epoch=0, total_epochs=100, end_mask_ratio=0.75):
    """
    Apply masking for MAE pretraining with temporal consistency and curriculum learning.

    Args:
        video (torch.Tensor): Input video of shape [C, T, H, W].
        mask_ratio (float): Initial ratio of patches to mask (default: 0.5).
        patch_size (int): Size of each patch (default: 16).
        temporal_consistency (str): Masking strategy:
            - 'full': Same mask for all frames.
            - 'partial': Change mask every `change_interval` frames.
            - 'none': Independent mask for each frame (original behavior).
        change_interval (int): Number of frames after which to change the mask (for 'partial' mode).
        current_epoch (int): Current epoch for curriculum learning.
        total_epochs (int): Total epochs for curriculum learning.
        end_mask_ratio (float): Final mask ratio for curriculum learning.

    Returns:
        masked_video (torch.Tensor): Video with masked patches [C, T, H, W].
        mask (torch.Tensor): Binary mask [T, H, W].
        mask_indices (torch.Tensor): Indices of masked patches [num_masked].
        total_patches (int): Total number of patches.
    """
    C, T, H, W = video.shape
    p = patch_size
    n_h, n_w = H // p, W // p

    # Curriculum learning: increase mask ratio from mask_ratio to end_mask_ratio
    progress = min(current_epoch / total_epochs, 1.0) if total_epochs > 0 else 0.0
    current_mask_ratio = mask_ratio + progress * (end_mask_ratio - mask_ratio)

    # Total patches per frame
    patches_per_frame = n_h * n_w
    total_patches = T * patches_per_frame

    # Create the binary mask
    mask = torch.zeros(T, H, W, dtype=torch.float32)

    if temporal_consistency == 'full':
        # Same mask for all frames
        num_mask = int(current_mask_ratio * patches_per_frame)
        all_idxs = np.arange(patches_per_frame)
        np.random.shuffle(all_idxs)
        mask_indices = all_idxs[:num_mask]

        # Apply the same mask to all frames
        frame_mask = torch.zeros(H, W, dtype=torch.float32)
        for idx in mask_indices:
            h_idx, w_idx = divmod(idx, n_w)
            frame_mask[h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = 1
        mask[:] = frame_mask  # Broadcast to all frames

        # Adjust mask indices to account for all frames
        mask_indices = np.concatenate([mask_indices + t * patches_per_frame for t in range(T)])
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    elif temporal_consistency == 'partial':
        # Change mask every `change_interval` frames
        num_mask = int(current_mask_ratio * patches_per_frame)
        mask_indices = []
        for t in range(0, T, change_interval):
            all_idxs = np.arange(patches_per_frame)
            np.random.shuffle(all_idxs)
            frame_mask_indices = all_idxs[:num_mask]
            frame_mask = torch.zeros(H, W, dtype=torch.float32)
            for idx in frame_mask_indices:
                h_idx, w_idx = divmod(idx, n_w)
                frame_mask[h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = 1
            for i in range(t, min(t + change_interval, T)):
                mask[i] = frame_mask
            # Add indices for this block of frames
            frame_indices = np.array([idx + i * patches_per_frame for i in range(min(t + change_interval, T) - t) for idx in frame_mask_indices])
            mask_indices.extend(frame_indices.flatten())
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    else:
        # Original behavior: independent mask for each frame
        total_patches = T * n_h * n_w
        all_idxs = np.arange(total_patches)
        np.random.shuffle(all_idxs)
        num_mask = int(current_mask_ratio * total_patches)
        mask_indices = all_idxs[:num_mask]

        for idx in mask_indices:
            t = idx // (n_h * n_w)
            rem = idx % (n_h * n_w)
            h_idx, w_idx = divmod(rem, n_w)
            mask[t, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = 1
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    # Apply the mask to the video
    masked_video = video * (1.0 - mask.unsqueeze(0))  # broadcast C dim

    return masked_video, mask, mask_indices, total_patches


def patchify(video, patch_size):
    """
    Splits a video tensor into patches.

    Args:
        video: [B, C, T, H, W]
        patch_size: spatial patch size (int)

    Returns:
        patches: [B, num_patches, patch_dim]
    """
    B, C, T, H, W = video.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    # flatten spatial dims into patches
    video = video.reshape(B, C, T, num_patches_h, patch_size, num_patches_w, patch_size)
    video = video.permute(0, 2, 3, 5, 1, 4, 6)  # B, T, h, w, C, p, p
    patches = video.reshape(B, T * num_patches_h * num_patches_w, C * patch_size * patch_size)
    return patches


def unpatchify(patches, patch_size, T, H, W):
    """
    Reconstructs full video from patches.

    Args:
        patches: [B, num_patches, patch_dim]
        patch_size: spatial patch size
        T, H, W: original temporal and spatial dims

    Returns:
        video: [B, C, T, H, W]
    """
    B, num_patches, patch_dim = patches.shape
    C = patch_dim // (patch_size * patch_size)
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    # reshape back to grid
    x = patches.reshape(B, T, num_patches_h, num_patches_w, C, patch_size, patch_size)
    x = x.permute(0, 4, 1, 2, 5, 3, 6)  # B, C, T, h, p, w, p
    video = x.reshape(B, C, T, H, W)
    return video
