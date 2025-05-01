import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from collections import deque
import logging

logger = logging.getLogger(__name__)

def apply_masking(video, mask_ratio=0.75, patch_size=16):
    """
    Now returns:
      masked_video: [C,T,H,W]
      mask:         [T,H,W]
      mask_indices: long tensor of shape [num_masked]
      total_patches (int)
    """
    C, T, H, W = video.shape
    p = patch_size
    n_h, n_w = H // p, W // p
    total_patches = T * n_h * n_w

    # pick which patches to mask
    all_idxs = np.arange(total_patches)
    np.random.shuffle(all_idxs)
    num_mask = int(mask_ratio * total_patches)
    mask_indices = all_idxs[:num_mask]

    # build the binary mask map
    mask = torch.zeros(T, H, W, dtype=torch.float32)
    for idx in mask_indices:
        t = idx // (n_h * n_w)
        rem = idx % (n_h * n_w)
        h_idx, w_idx = divmod(rem, n_w)
        mask[t,
             h_idx*p:(h_idx+1)*p,
             w_idx*p:(w_idx+1)*p] = 1

    # apply it
    masked_video = video * (1.0 - mask.unsqueeze(0))  # broadcast C dim

    return masked_video, mask, torch.tensor(mask_indices, dtype=torch.long), total_patches




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
