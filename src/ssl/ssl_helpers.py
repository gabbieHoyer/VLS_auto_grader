import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from collections import deque
import logging

logger = logging.getLogger(__name__)

def apply_masking(video, mask_ratio, patch_size, temporal_consistency, change_interval):
    """
    Apply masking for MAE pretraining with temporal consistency.

    Args:
        video (torch.Tensor): Input video of shape [C, T, H, W].
        mask_ratio (float): Ratio of patches to mask.
        patch_size (int): Size of each patch.
        temporal_consistency (str): Masking strategy ('full', 'partial', 'none').
        change_interval (int): Number of frames after which to change the mask (for 'partial').

    Returns:
        masked_video (torch.Tensor): Video with masked patches [C, T, H, W].
        mask (torch.Tensor): Binary mask [T, H, W].
        mask_indices (torch.Tensor): Indices of masked patches [num_masked].
        total_patches (int): Total number of patches.
    """
    # Validate inputs
    if not (0 <= mask_ratio <= 1):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size}")
    if temporal_consistency not in ['full', 'partial', 'none']:
        raise ValueError(f"temporal_consistency must be 'full', 'partial', or 'none', got {temporal_consistency}")
    if temporal_consistency == 'partial' and (not isinstance(change_interval, int) or change_interval <= 0):
        raise ValueError(f"change_interval must be a positive integer, got {change_interval}")

    C, T, H, W = video.shape
    p = patch_size
    n_h, n_w = H // p, W // p

    # Total patches per frame
    patches_per_frame = n_h * n_w
    total_patches = T * patches_per_frame

    # Create the binary mask
    mask = torch.zeros(T, H, W, dtype=torch.float32)

    if temporal_consistency == 'full':
        num_mask = int(mask_ratio * patches_per_frame)
        all_idxs = np.arange(patches_per_frame)
        np.random.shuffle(all_idxs)
        mask_indices = all_idxs[:num_mask]

        frame_mask = torch.zeros(H, W, dtype=torch.float32)
        for idx in mask_indices:
            h_idx, w_idx = divmod(idx, n_w)
            frame_mask[h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = 1
        mask[:] = frame_mask
        mask_indices = np.concatenate([mask_indices + t * patches_per_frame for t in range(T)])
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    elif temporal_consistency == 'partial':
        num_mask = int(mask_ratio * patches_per_frame)
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
            frame_indices = np.array([idx + i * patches_per_frame for i in range(min(t + change_interval, T) - t) for idx in frame_mask_indices])
            mask_indices.extend(frame_indices.flatten())
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    else:  # 'none'
        total_patches = T * n_h * n_w
        all_idxs = np.arange(total_patches)
        np.random.shuffle(all_idxs)
        num_mask = int(mask_ratio * total_patches)
        mask_indices = all_idxs[:num_mask]

        for idx in mask_indices:
            t = idx // (n_h * n_w)
            rem = idx % (n_h * n_w)
            h_idx, w_idx = divmod(rem, n_w)
            mask[t, h_idx*p:(h_idx+1)*p, w_idx*p:(w_idx+1)*p] = 1
        mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    masked_video = video * (1.0 - mask.unsqueeze(0))
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

    # Validate dimensions
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H ({H}) and W ({W}) must be divisible by patch_size ({patch_size})")
    expected_patches = T * num_patches_h * num_patches_w
    if num_patches != expected_patches:
        raise ValueError(f"num_patches ({num_patches}) does not match T ({T}) * num_patches_h ({num_patches_h}) * num_patches_w ({num_patches_w}) = {expected_patches}")
    
    # Reshape back to grid
    x = patches.reshape(B, T, num_patches_h, num_patches_w, C, patch_size, patch_size)
    x = x.permute(0, 4, 1, 2, 5, 3, 6)  # B, C, T, h, p, w, p
    video = x.reshape(B, C, T, H, W)
    return video
