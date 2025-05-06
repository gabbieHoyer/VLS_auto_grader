# src/models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from collections import deque
import logging

from src.ssl import unpatchify

# Set up logging
logger = logging.getLogger(__name__)


class I3D(nn.Module):
    def __init__(
        self,
        num_base_classes: int,
        num_subclasses:   int    = 0,
        pretrained:       bool   = True,
        pretrain_method:  str|None = None,
        patch_size: int | None = None,
        mask_ratio: float | None = None,
        end_mask_ratio: float | None = None,
    ):
        super().__init__()
        self.model = video_models.r3d_18(pretrained=pretrained)
        self.in_features = self.model.fc.in_features
        self.pretrain_method = pretrain_method.lower() if pretrain_method else None
        self.hierarchical = num_subclasses > 0

        logger.info(f"Initializing I3D with pretrain_method: {self.pretrain_method}")

        # Disable in-place ReLUs in backbone
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        # Remove final fc for SSL modes
        if self.pretrain_method in ('contrastive', 'moco', 'mae'):
            self.model.fc = nn.Identity()

        if self.pretrain_method == 'mae':
            # Set MAE-specific defaults if None, or validate provided values
            self.patch_size = patch_size if patch_size is not None else 16
            self._mask_ratio = mask_ratio if mask_ratio is not None else 0.75
            self._end_mask_ratio = (
                end_mask_ratio if end_mask_ratio is not None else self._mask_ratio
            )

            # Validate parameters
            if not isinstance(self.patch_size, int) or self.patch_size <= 0:
                raise ValueError(f"patch_size must be a positive integer, got {self.patch_size}")
            if not (0 <= self._mask_ratio <= 1):
                raise ValueError(f"mask_ratio must be in [0, 1], got {self._mask_ratio}")
            if not (0 <= self._end_mask_ratio <= 1):
                raise ValueError(f"end_mask_ratio must be in [0, 1], got {self._end_mask_ratio}")

            self.frame_size = 224
            self.T = 16
            self.num_patches_per_frame = (self.frame_size // patch_size) ** 2
            self.total_patches = self.T * self.num_patches_per_frame

            logger.info(
                f"MAE setup: patch_size={self.patch_size}, "
                f"initial mask_ratio={self._mask_ratio:.3f}, "
                f"end_mask_ratio={self._end_mask_ratio:.3f}, "
            )
            # Defer patch_dim calculation to forward, define decoder without fixed output
            self.decoder = nn.Sequential(
                nn.Linear(self.in_features, 512),
                nn.ReLU(inplace=False),
                nn.Linear(512, 512),  # Temporary output, adjust in forward
            )
            self.decoder_output = nn.Linear(512, 512)  # Initial placeholder

        # common projection head (used by contrastive & MoCo)
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout with rate 0.3
            nn.Linear(512, 128),
        )

        # MoCo momentum encoder & queue
        if self.pretrain_method == 'moco':
            self.momentum_encoder = video_models.r3d_18(pretrained=pretrained)
            self.momentum_encoder.fc = nn.Identity()
            # freeze momentum parameters
            for p in self.momentum_encoder.parameters():
                p.requires_grad = False
            # projection head for keys
            self.momentum_projection_head = nn.Sequential(
                nn.Linear(self.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.1),  # Added dropout with rate 0.3
                nn.Linear(512, 128),
            )
            self.momentum = 0.999
            self.queue_size = 4096
            self.register_buffer('queue', torch.randn(self.queue_size, 128))  # placeholder
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Supervised classification heads with dropout
        if self.pretrain_method not in ('contrastive', 'moco', 'mae'):
            if self.hierarchical:
                self.model.fc = nn.Identity()
                self.base_head = nn.Sequential(
                    nn.Linear(self.in_features, num_base_classes),
                    nn.Dropout(0.2)  # Added dropout with rate 0.5
                )
                self.subclass_head = nn.Sequential(
                    nn.Linear(self.in_features, num_subclasses),
                    nn.Dropout(0.2)  # Added dropout with rate 0.5
                )
            else:
                self.model.fc = nn.Sequential(
                    nn.Linear(self.in_features, num_base_classes),
                    nn.Dropout(0.2)  # Added dropout with rate 0.5
                )
                logger.info(
                    f"Set supervised fc to Linear({self.in_features}, {num_base_classes}) with dropout"
                )

    def set_mask_ratio(self, mask_ratio):
        """Update mask_ratio and recompute num_masked."""
        if self.pretrain_method == 'mae':
            if not (0 <= mask_ratio <= 1):
                raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
            self._mask_ratio = mask_ratio
            logger.info(f"Updated mask_ratio to {self._mask_ratio:.3f}")
        else:
            logger.warning("set_mask_ratio called but pretrain_method is not 'mae'")

    @property
    def mask_ratio(self):
        return self._mask_ratio

    @property
    def end_mask_ratio(self):
        return self._end_mask_ratio
    
    @property
    def num_masked(self):
        # This will be overridden by forward based on mask_indices
        return int(self._mask_ratio * self.total_patches) if hasattr(self, 'total_patches') else 0

    def _update_momentum_encoder(self):
        # Momentum update: key_encoder = m * key_encoder + (1-m) * query_encoder
        for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        for qh, kh in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            kh.data = kh.data * self.momentum + qh.data * (1.0 - self.momentum)

    def _enqueue_and_dequeue(self, keys):
        # keys: [B, 128]
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        # replace the oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr+batch_size] = keys.detach()
        else:
            # wrap-around
            end = self.queue_size - ptr
            self.queue[ptr:] = keys[:end].detach()
            self.queue[: batch_size - end] = keys[end:].detach()
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x, x2=None, mask_indices=None):
        if self.pretrain_method == 'mae':
            # x: [B, 3, T, H, W]  -> torch.Size([8, 3, 16, 224, 224])
            B, C, T, H, W = x.shape  # Derive dimensions from input
            self.T = T
            self.frame_size = H  # Update frame_size dynamically
            self.num_patches_per_frame = (H // self.patch_size) * (W // self.patch_size) # 196
            self.total_patches = T * self.num_patches_per_frame  # 3136
            self.patch_dim = C * self.patch_size * self.patch_size # 768

            # import pdb; pdb.set_trace()
            # Update decoder output layer to match patch_dim
            if self.decoder_output.out_features != self.patch_dim:
                self.decoder_output = nn.Linear(512, self.patch_dim).to(x.device)
                logger.info(f"Updated decoder output layer to match patch_dim={self.patch_dim}")

            feat = self.model(x)  # [B, in_features]  # [8, 512]
            out = self.decoder(feat)  # [B, 512]
            out = self.decoder_output(out)  # [B, patch_dim]  # [8, 768]
            num_masked = mask_indices.size(1) if mask_indices.dim() > 1 else mask_indices.size(0)   # 624 and 8, respectively
            out = out.view(B, 1, self.patch_dim).repeat(1, num_masked, 1)  # [B, num_masked, patch_dim]   # torch.Size([8, 624, 768])

            # Scatter into full patch grid
            device, dtype = out.device, out.dtype
            full = torch.zeros(
                (B, self.total_patches, self.patch_dim),
                device=device, dtype=dtype
            )
            if mask_indices.dim() == 1:
                mask_indices = mask_indices.unsqueeze(0).expand(B, -1)
            for i in range(B):
                full[i, mask_indices[i]] = out[i]

                # video = unpatchify(full, self.patch_size, self.T, self.frame_size, self.frame_size)
                video = unpatchify(full, self.patch_size, self.T, H, W)
                return video

        if self.pretrain_method == 'contrastive':
            # require two views
            assert x2 is not None, "Contrastive mode needs both x and x2"
            q = self.projection_head(self.model(x))   # queries
            k = self.projection_head(self.model(x2))  # keys for contrastive loss
            return q, k

        if self.pretrain_method == 'moco':
            # query
            q = self.projection_head(self.model(x))
            # update key encoder and build key
            with torch.no_grad():
                self._update_momentum_encoder()
                k = self.momentum_projection_head(self.momentum_encoder(x))
            # enqueue & dequeue
            self._enqueue_and_dequeue(k)
            return q, k

        # supervised forward
        feat = self.model(x)
        if self.hierarchical:
            return self.base_head(feat), self.subclass_head(feat)
        return feat


# class I3D(nn.Module):
#     def __init__(
#         self,
#         num_base_classes: int,
#         num_subclasses:   int    = 0,
#         pretrained:       bool   = True,
#         pretrain_method:  str|None = None,
#         patch_size: int | None = None,
#         mask_ratio: float | None = None,
#         end_mask_ratio: float | None = None,
#     ):
#         super().__init__()
#         self.model = video_models.r3d_18(pretrained=pretrained)
#         self.in_features = self.model.fc.in_features
#         self.pretrain_method = pretrain_method.lower() if pretrain_method else None
#         self.hierarchical = num_subclasses > 0

#         logger.info(f"Initializing I3D with pretrain_method: {self.pretrain_method}")

#         # Disable in-place ReLUs in backbone
#         for m in self.model.modules():
#             if isinstance(m, nn.ReLU):
#                 m.inplace = False

#         # Remove final fc for SSL modes
#         if self.pretrain_method in ('contrastive', 'moco', 'mae'):
#             self.model.fc = nn.Identity()

#         if self.pretrain_method == 'mae':
#             # Set MAE-specific defaults if None, or validate provided values
#             self.patch_size = patch_size if patch_size is not None else 16
#             self._mask_ratio = mask_ratio if mask_ratio is not None else 0.75
#             self._end_mask_ratio = (
#                 end_mask_ratio if end_mask_ratio is not None else self._mask_ratio
#             )

#             # Validate parameters
#             if not isinstance(self.patch_size, int) or self.patch_size <= 0:
#                 raise ValueError(f"patch_size must be a positive integer, got {self.patch_size}")
#             if not (0 <= self._mask_ratio <= 1):
#                 raise ValueError(f"mask_ratio must be in [0, 1], got {self._mask_ratio}")
#             if not (0 <= self._end_mask_ratio <= 1):
#                 raise ValueError(f"end_mask_ratio must be in [0, 1], got {self._end_mask_ratio}")

#             self.frame_size = 224
#             self.T = 16
#             self.num_patches_per_frame = (self.frame_size // patch_size) ** 2
#             self.total_patches = self.T * self.num_patches_per_frame

#             logger.info(
#                 f"MAE setup: patch_size={self.patch_size}, "
#                 f"initial mask_ratio={self._mask_ratio:.3f}, "
#                 f"end_mask_ratio={self._end_mask_ratio:.3f}, "
#             )
#             # Defer patch_dim calculation to forward, define decoder without fixed output
#             self.decoder = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(inplace=False),
#                 nn.Linear(512, 512),  # Temporary output, adjust in forward
#             )
#             self.decoder_output = nn.Linear(512, 512)  # Initial placeholder

#         # common projection head (used by contrastive & MoCo)
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.in_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),  # Added dropout with rate 0.3
#             nn.Linear(512, 128),
#         )

#         # MoCo momentum encoder & queue
#         if self.pretrain_method == 'moco':
#             self.momentum_encoder = video_models.r3d_18(pretrained=pretrained)
#             self.momentum_encoder.fc = nn.Identity()
#             # freeze momentum parameters
#             for p in self.momentum_encoder.parameters():
#                 p.requires_grad = False
#             # projection head for keys
#             self.momentum_projection_head = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),  # Added dropout with rate 0.3
#                 nn.Linear(512, 128),
#             )
#             self.momentum = 0.999
#             self.queue_size = 4096
#             self.register_buffer('queue', torch.randn(self.queue_size, 128))  # placeholder
#             self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

#         # Supervised classification heads with dropout
#         if self.pretrain_method not in ('contrastive', 'moco', 'mae'):
#             if self.hierarchical:
#                 self.model.fc = nn.Identity()
#                 self.base_head = nn.Sequential(
#                     nn.Linear(self.in_features, num_base_classes),
#                     nn.Dropout(0.5)  # Added dropout with rate 0.5
#                 )
#                 self.subclass_head = nn.Sequential(
#                     nn.Linear(self.in_features, num_subclasses),
#                     nn.Dropout(0.5)  # Added dropout with rate 0.5
#                 )
#             else:
#                 self.model.fc = nn.Sequential(
#                     nn.Linear(self.in_features, num_base_classes),
#                     nn.Dropout(0.5)  # Added dropout with rate 0.5
#                 )
#                 logger.info(
#                     f"Set supervised fc to Linear({self.in_features}, {num_base_classes}) with dropout"
#                 )

#     def set_mask_ratio(self, mask_ratio):
#         """Update mask_ratio and recompute num_masked."""
#         if self.pretrain_method == 'mae':
#             if not (0 <= mask_ratio <= 1):
#                 raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
#             self._mask_ratio = mask_ratio
#             logger.info(f"Updated mask_ratio to {self._mask_ratio:.3f}")
#         else:
#             logger.warning("set_mask_ratio called but pretrain_method is not 'mae'")

#     @property
#     def mask_ratio(self):
#         return self._mask_ratio

#     @property
#     def end_mask_ratio(self):
#         return self._end_mask_ratio
    
#     @property
#     def num_masked(self):
#         # This will be overridden by forward based on mask_indices
#         return int(self._mask_ratio * self.total_patches) if hasattr(self, 'total_patches') else 0

#     def _update_momentum_encoder(self):
#         # Momentum update: key_encoder = m * key_encoder + (1-m) * query_encoder
#         for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
#             param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
#         for qh, kh in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
#             kh.data = kh.data * self.momentum + qh.data * (1.0 - self.momentum)

#     def _enqueue_and_dequeue(self, keys):
#         # keys: [B, 128]
#         batch_size = keys.size(0)
#         ptr = int(self.queue_ptr)
#         # replace the oldest entries
#         if ptr + batch_size <= self.queue_size:
#             self.queue[ptr:ptr+batch_size] = keys.detach()
#         else:
#             # wrap-around
#             end = self.queue_size - ptr
#             self.queue[ptr:] = keys[:end].detach()
#             self.queue[: batch_size - end] = keys[end:].detach()
#         self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

#     def forward(self, x, x2=None, mask_indices=None):
#         if self.pretrain_method == 'mae':
#             # x: [B, 3, T, H, W]  -> torch.Size([8, 3, 16, 224, 224])
#             B, C, T, H, W = x.shape  # Derive dimensions from input
#             self.T = T
#             self.frame_size = H  # Update frame_size dynamically
#             self.num_patches_per_frame = (H // self.patch_size) * (W // self.patch_size) # 196
#             self.total_patches = T * self.num_patches_per_frame  # 3136
#             self.patch_dim = C * self.patch_size * self.patch_size # 768

#             # import pdb; pdb.set_trace()
#             # Update decoder output layer to match patch_dim
#             if self.decoder_output.out_features != self.patch_dim:
#                 self.decoder_output = nn.Linear(512, self.patch_dim).to(x.device)
#                 logger.info(f"Updated decoder output layer to match patch_dim={self.patch_dim}")

#             feat = self.model(x)  # [B, in_features]  # [8, 512]
#             out = self.decoder(feat)  # [B, 512]
#             out = self.decoder_output(out)  # [B, patch_dim]  # [8, 768]
#             num_masked = mask_indices.size(1) if mask_indices.dim() > 1 else mask_indices.size(0)   # 624 and 8, respectively
#             out = out.view(B, 1, self.patch_dim).repeat(1, num_masked, 1)  # [B, num_masked, patch_dim]   # torch.Size([8, 624, 768])

#             # Scatter into full patch grid
#             device, dtype = out.device, out.dtype
#             full = torch.zeros(
#                 (B, self.total_patches, self.patch_dim),
#                 device=device, dtype=dtype
#             )
#             if mask_indices.dim() == 1:
#                 mask_indices = mask_indices.unsqueeze(0).expand(B, -1)
#             for i in range(B):
#                 full[i, mask_indices[i]] = out[i]

#                 # video = unpatchify(full, self.patch_size, self.T, self.frame_size, self.frame_size)
#                 video = unpatchify(full, self.patch_size, self.T, H, W)
#                 return video

#         if self.pretrain_method == 'contrastive':
#             # require two views
#             assert x2 is not None, "Contrastive mode needs both x and x2"
#             q = self.projection_head(self.model(x))   # queries
#             k = self.projection_head(self.model(x2))  # keys for contrastive loss
#             return q, k

#         if self.pretrain_method == 'moco':
#             # query
#             q = self.projection_head(self.model(x))
#             # update key encoder and build key
#             with torch.no_grad():
#                 self._update_momentum_encoder()
#                 k = self.momentum_projection_head(self.momentum_encoder(x))
#             # enqueue & dequeue
#             self._enqueue_and_dequeue(k)
#             return q, k

#         # supervised forward
#         feat = self.model(x)
#         if self.hierarchical:
#             return self.base_head(feat), self.subclass_head(feat)
#         return feat
    

# class I3D(nn.Module):
#     def __init__(
#         self,
#         num_base_classes: int,
#         num_subclasses:   int    = 0,
#         pretrained:       bool   = True,
#         pretrain_method:  str|None = None,
#         patch_size: int | None = None,
#         mask_ratio: float | None = None,
#         end_mask_ratio: float | None = None,
#     ):
#         super().__init__()
#         self.model = video_models.r3d_18(pretrained=pretrained)
#         self.in_features = self.model.fc.in_features
#         self.pretrain_method = pretrain_method.lower() if pretrain_method else None
#         self.hierarchical = num_subclasses > 0

#         logger.info(f"Initializing I3D with pretrain_method: {self.pretrain_method}")

#         # Disable in-place ReLUs in backbone
#         for m in self.model.modules():
#             if isinstance(m, nn.ReLU):
#                 m.inplace = False

#         # Remove final fc for SSL modes
#         if self.pretrain_method in ('contrastive', 'moco', 'mae'):
#             self.model.fc = nn.Identity()

#         if self.pretrain_method == 'mae':
#             # Set MAE-specific defaults if None, or validate provided values
#             self.patch_size = patch_size if patch_size is not None else 16
#             self._mask_ratio = mask_ratio if mask_ratio is not None else 0.75
#             self._end_mask_ratio = (
#                 end_mask_ratio if end_mask_ratio is not None else self._mask_ratio
#             )

#             # Validate parameters
#             if not isinstance(self.patch_size, int) or self.patch_size <= 0:
#                 raise ValueError(f"patch_size must be a positive integer, got {self.patch_size}")
#             if not (0 <= self._mask_ratio <= 1):
#                 raise ValueError(f"mask_ratio must be in [0, 1], got {self._mask_ratio}")
#             if not (0 <= self._end_mask_ratio <= 1):
#                 raise ValueError(f"end_mask_ratio must be in [0, 1], got {self._end_mask_ratio}")

#             self.frame_size = 224
#             self.T = 16
#             self.num_patches_per_frame = (self.frame_size // patch_size) ** 2
#             self.total_patches = self.T * self.num_patches_per_frame

#             logger.info(
#                 f"MAE setup: patch_size={self.patch_size}, "
#                 f"initial mask_ratio={self._mask_ratio:.3f}, "
#                 f"end_mask_ratio={self._end_mask_ratio:.3f}, "
#             )
#             # Defer patch_dim calculation to forward, define decoder without fixed output
#             self.decoder = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(inplace=False),
#                 nn.Linear(512, 512),  # Temporary output, adjust in forward
#             )
#             self.decoder_output = nn.Linear(512, 512)  # Initial placeholder

#         # common projection head (used by contrastive & MoCo)
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.in_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#         )

#         # MoCo momentum encoder & queue
#         if self.pretrain_method == 'moco':
#             self.momentum_encoder = video_models.r3d_18(pretrained=pretrained)
#             self.momentum_encoder.fc = nn.Identity()
#             # freeze momentum parameters
#             for p in self.momentum_encoder.parameters():
#                 p.requires_grad = False
#             # projection head for keys
#             self.momentum_projection_head = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 128),
#             )
#             self.momentum = 0.999
#             self.queue_size = 4096
#             self.register_buffer('queue', torch.randn(self.queue_size, 128))  # placeholder
#             self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

#         # supervised classification heads
#         if self.pretrain_method not in ('contrastive', 'moco', 'mae'):
#             if self.hierarchical:
#                 self.model.fc = nn.Identity()
#                 self.base_head = nn.Linear(self.in_features, num_base_classes)
#                 self.subclass_head = nn.Linear(self.in_features, num_subclasses)
#             else:
#                 self.model.fc = nn.Linear(self.in_features, num_base_classes)
#                 logger.info(
#                     f"Set supervised fc to Linear({self.in_features}, {num_base_classes})"
#                 )

#     def set_mask_ratio(self, mask_ratio):
#         """Update mask_ratio and recompute num_masked."""
#         if self.pretrain_method == 'mae':
#             if not (0 <= mask_ratio <= 1):
#                 raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
#             self._mask_ratio = mask_ratio
#             logger.info(f"Updated mask_ratio to {self._mask_ratio:.3f}")
#         else:
#             logger.warning("set_mask_ratio called but pretrain_method is not 'mae'")

#     @property
#     def mask_ratio(self):
#         return self._mask_ratio

#     @property
#     def end_mask_ratio(self):
#         return self._end_mask_ratio
    
#     @property
#     def num_masked(self):
#         # This will be overridden by forward based on mask_indices
#         return int(self._mask_ratio * self.total_patches) if hasattr(self, 'total_patches') else 0

#     def _update_momentum_encoder(self):
#         # Momentum update: key_encoder = m * key_encoder + (1-m) * query_encoder
#         for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
#             param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
#         for qh, kh in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
#             kh.data = kh.data * self.momentum + qh.data * (1.0 - self.momentum)

#     def _enqueue_and_dequeue(self, keys):
#         # keys: [B, 128]
#         batch_size = keys.size(0)
#         ptr = int(self.queue_ptr)
#         # replace the oldest entries
#         if ptr + batch_size <= self.queue_size:
#             self.queue[ptr:ptr+batch_size] = keys.detach()
#         else:
#             # wrap-around
#             end = self.queue_size - ptr
#             self.queue[ptr:] = keys[:end].detach()
#             self.queue[: batch_size - end] = keys[end:].detach()
#         self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

#     def forward(self, x, x2=None, mask_indices=None):
#         if self.pretrain_method == 'mae':
#             # x: [B, 3, T, H, W]  -> torch.Size([8, 3, 16, 224, 224])
#             B, C, T, H, W = x.shape  # Derive dimensions from input
#             self.T = T
#             self.frame_size = H  # Update frame_size dynamically
#             self.num_patches_per_frame = (H // self.patch_size) * (W // self.patch_size) # 196
#             self.total_patches = T * self.num_patches_per_frame  # 3136
#             self.patch_dim = C * self.patch_size * self.patch_size # 768

#             # import pdb; pdb.set_trace()
#             # Update decoder output layer to match patch_dim
#             if self.decoder_output.out_features != self.patch_dim:
#                 self.decoder_output = nn.Linear(512, self.patch_dim).to(x.device)
#                 logger.info(f"Updated decoder output layer to match patch_dim={self.patch_dim}")

#             feat = self.model(x)  # [B, in_features]  # [8, 512]
#             out = self.decoder(feat)  # [B, 512]
#             out = self.decoder_output(out)  # [B, patch_dim]  # [8, 768]
#             num_masked = mask_indices.size(1) if mask_indices.dim() > 1 else mask_indices.size(0)   # 624 and 8, respectively
#             out = out.view(B, 1, self.patch_dim).repeat(1, num_masked, 1)  # [B, num_masked, patch_dim]   # torch.Size([8, 624, 768])

#             # Scatter into full patch grid
#             device, dtype = out.device, out.dtype
#             full = torch.zeros(
#                 (B, self.total_patches, self.patch_dim),
#                 device=device, dtype=dtype
#             )
#             if mask_indices.dim() == 1:
#                 mask_indices = mask_indices.unsqueeze(0).expand(B, -1)
#             for i in range(B):
#                 full[i, mask_indices[i]] = out[i]

#                 # video = unpatchify(full, self.patch_size, self.T, self.frame_size, self.frame_size)
#                 video = unpatchify(full, self.patch_size, self.T, H, W)
#                 return video

#         if self.pretrain_method == 'contrastive':
#             # require two views
#             assert x2 is not None, "Contrastive mode needs both x and x2"
#             q = self.projection_head(self.model(x))   # queries
#             k = self.projection_head(self.model(x2))  # keys for contrastive loss
#             return q, k

#         if self.pretrain_method == 'moco':
#             # query
#             q = self.projection_head(self.model(x))
#             # update key encoder and build key
#             with torch.no_grad():
#                 self._update_momentum_encoder()
#                 k = self.momentum_projection_head(self.momentum_encoder(x))
#             # enqueue & dequeue
#             self._enqueue_and_dequeue(k)
#             return q, k

#         # supervised forward
#         feat = self.model(x)
#         if self.hierarchical:
#             return self.base_head(feat), self.subclass_head(feat)
#         return feat


class ViViT(nn.Module):
    def __init__(self, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None):
        super(ViViT, self).__init__()
        self.pretrain_method = pretrain_method
        self.hierarchical = num_subclasses > 0
        self.num_frames = 16  # Temporal dimension (T)
        self.frame_size = 224  # Spatial dimension (H, W)
        self.patch_size = 16  # Patch size for spatial dimension
        self.tubelet_size = 2  # Temporal patch size
        self.num_spatial_patches = (self.frame_size // self.patch_size) ** 2  # (224/16)^2 = 196
        self.num_temporal_patches = self.num_frames // self.tubelet_size  # 16/2 = 8
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches  # 196 * 8 = 1568
        self.embed_dim = 768  # Embedding dimension (like ViT-base)
        self.in_features = self.embed_dim  # Features after Transformer (CLS token)

        # Patch embedding: [C, T, H, W] -> [num_patches, embed_dim]
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            stride=(self.tubelet_size, self.patch_size, self.patch_size)
        )

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)  # +1 for CLS token
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # For contrastive or MoCo pretraining
        pretrain_method_lower = pretrain_method.lower() if pretrain_method else None
        if pretrain_method_lower in ['contrastive', 'moco']:
            self.projection_head = nn.Sequential(
                nn.Linear(self.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )
            if pretrain_method_lower == 'moco':
                # Momentum encoder
                self.momentum_encoder = nn.ModuleList([
                    nn.Conv3d(
                        in_channels=3,
                        out_channels=self.embed_dim,
                        kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
                        stride=(self.tubelet_size, self.patch_size, self.patch_size)
                    ),
                    nn.TransformerEncoder(encoder_layer, num_layers=12)
                ])
                for param in self.momentum_encoder.parameters():
                    param.requires_grad = False
                self.momentum_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.momentum_pos_embed = nn.Parameter(
                    torch.zeros(1, self.num_patches + 1, self.embed_dim)
                )
                self.momentum_projection_head = nn.Sequential(
                    nn.Linear(self.in_features, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)
                )
                self.momentum = 0.999  # Momentum update factor
                self.queue_size = 4096
                self.queue = deque(maxlen=self.queue_size)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        elif pretrain_method_lower == 'mae':
            self.decoder = nn.Sequential(
                nn.Linear(self.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 3 * self.num_frames * self.frame_size * self.frame_size // (self.tubelet_size * self.patch_size * self.patch_size)),
                nn.ReLU()
            )
        else:
            # Supervised learning setup
            if self.hierarchical:
                self.base_head = nn.Linear(self.in_features, num_base_classes)
                self.subclass_head = nn.Linear(self.in_features, num_subclasses)
            else:
                self.head = nn.Linear(self.in_features, num_base_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if pretrain_method_lower == 'moco':
            nn.init.trunc_normal_(self.momentum_cls_token, std=0.02)
            nn.init.trunc_normal_(self.momentum_pos_embed, std=0.02)

    def _update_momentum_encoder(self):
        with torch.no_grad():
            for param, param_m in zip(self.patch_embed.parameters(), self.momentum_encoder[0].parameters()):
                new_data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                param_m.copy_(new_data)
            for param, param_m in zip(self.encoder.parameters(), self.momentum_encoder[1].parameters()):
                new_data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                param_m.copy_(new_data)
            for param, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
                new_data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                param_m.copy_(new_data)
            self.momentum_cls_token.copy_(self.momentum_cls_token.data * self.momentum + self.cls_token.data * (1. - self.momentum))
            self.momentum_pos_embed.copy_(self.momentum_pos_embed.data * self.momentum + self.pos_embed.data * (1. - self.momentum))

    def _enqueue_and_dequeue(self, keys):
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        self.queue.extend(keys.detach().cpu().numpy())
        if len(self.queue) >= self.queue_size:
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x, x2=None):
        pretrain_method_lower = self.pretrain_method.lower() if self.pretrain_method else None
        logger.debug(f"Forward pass with pretrain_method: {pretrain_method_lower}")

        batch_size = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        feat = self.encoder(x)[:, 0]

        if pretrain_method_lower in ['contrastive', 'moco']:
            feat = self.projection_head(feat)
            if pretrain_method_lower == 'moco':
                with torch.no_grad():
                    self._update_momentum_encoder()
                    keys = self.momentum_encoder[0](x)
                    keys = keys.flatten(2).transpose(1, 2)
                    cls_tokens_m = self.momentum_cls_token.expand(batch_size, -1, -1)
                    keys = torch.cat((cls_tokens_m, keys), dim=1)
                    keys = keys + self.momentum_pos_embed
                    keys = self.momentum_encoder[1](keys)[:, 0]
                    keys = self.momentum_projection_head(keys)
                    self._enqueue_and_dequeue(keys)
                return feat, keys
            if x2 is not None:
                x2 = self.patch_embed(x2)
                x2 = x2.flatten(2).transpose(1, 2)
                cls_tokens2 = self.cls_token.expand(batch_size, -1, -1)
                x2 = torch.cat((cls_tokens2, x2), dim=1)
                x2 = x2 + self.pos_embed
                feat2 = self.encoder(x2)[:, 0]
                feat2 = self.projection_head(feat2)
                return feat, feat2
            return feat
        elif pretrain_method_lower == 'mae':
            reconstructed = self.decoder(feat)
            reconstructed = reconstructed.view(
                batch_size, 3, self.num_frames, self.frame_size, self.frame_size
            )
            return reconstructed
        else:
            if self.hierarchical:
                base_logits = self.base_head(feat)
                subclass_logits = self.subclass_head(feat)
                return base_logits, subclass_logits
            return self.head(feat)

    def get_attention_maps(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        attn_maps = []
        h = x
        for layer in self.encoder.layers:
            h, attn = layer.self_attn(h, h, h, need_weights=True)
            attn_maps.append(attn)
        return attn_maps


def get_model(
    model_name: str,
    num_base_classes: int,
    num_subclasses: int = 0,
    pretrained: bool = True,
    pretrain_method: str | None = None,
    patch_size: int | None = None,  # Optional, only for MAE
    mask_ratio: float | None = None,  # Optional, only for MAE
    end_mask_ratio: float | None = None  # Optional, only for MAE
) -> torch.nn.Module:
    """
    Initialize and return a model based on the specified name.

    Args:
        model_name (str): 'i3d' or 'vivit'.
        num_base_classes (int): Number of base classes for classification.
        num_subclasses (int): Number of subclasses for hierarchical classification.
        pretrained (bool): Whether to use pretrained weights.
        pretrain_method (str | None): 'contrastive', 'moco', 'mae', or None.
        patch_size (int | None): Patch side length (only used when pretrain_method=='mae').
        mask_ratio (float | None): Initial fraction of patches to mask (only for MAE).
        end_mask_ratio (float | None): Final fraction of patches to mask for MAE curriculum learning.

    Returns:
        nn.Module: Initialized model.
    """
    if model_name.lower() == 'i3d':
        return I3D(
            num_base_classes=num_base_classes,
            num_subclasses=num_subclasses,
            pretrained=pretrained,
            pretrain_method=pretrain_method,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            end_mask_ratio=end_mask_ratio,
        )
    elif model_name.lower() == 'vivit':
        return ViViT(
            num_base_classes=num_base_classes,
            num_subclasses=num_subclasses,
            pretrained=pretrained,
            pretrain_method=pretrain_method,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")







# class I3D(nn.Module):
#     def __init__(
#         self,
#         num_base_classes: int,
#         num_subclasses:   int    = 0,
#         pretrained:       bool   = True,
#         pretrain_method:  str|None = None,
#         patch_size: int | None = None,
#         mask_ratio: float | None = None,
#         end_mask_ratio: float | None = None,
#     ):
#         super().__init__()
#         self.model = video_models.r3d_18(pretrained=pretrained)
#         self.in_features = self.model.fc.in_features
#         self.pretrain_method = pretrain_method.lower() if pretrain_method else None
#         self.hierarchical = num_subclasses > 0

#         logger.info(f"Initializing I3D with pretrain_method: {self.pretrain_method}")

#         # Disable in-place ReLUs in backbone
#         for m in self.model.modules():
#             if isinstance(m, nn.ReLU):
#                 m.inplace = False

#         # Remove final fc for SSL modes
#         if self.pretrain_method in ('contrastive', 'moco', 'mae'):
#             self.model.fc = nn.Identity()

#         if self.pretrain_method == 'mae':
#             # Set MAE-specific defaults if None, or validate provided values
#             self.patch_size = patch_size if patch_size is not None else 16
#             self._mask_ratio = mask_ratio if mask_ratio is not None else 0.75
#             self._end_mask_ratio = (
#                 end_mask_ratio if end_mask_ratio is not None else self._mask_ratio
#             )

#             # Validate parameters
#             if not isinstance(self.patch_size, int) or self.patch_size <= 0:
#                 raise ValueError(f"patch_size must be a positive integer, got {self.patch_size}")
#             if not (0 <= self._mask_ratio <= 1):
#                 raise ValueError(f"mask_ratio must be in [0, 1], got {self._mask_ratio}")
#             if not (0 <= self._end_mask_ratio <= 1):
#                 raise ValueError(f"end_mask_ratio must be in [0, 1], got {self._end_mask_ratio}")

#             self.frame_size = 224
#             self.T = 16
#             self.num_patches_per_frame = (self.frame_size // patch_size) ** 2
#             self.total_patches = self.T * self.num_patches_per_frame

#             logger.info(
#                 f"MAE setup: patch_size={self.patch_size}, "
#                 f"initial mask_ratio={self._mask_ratio:.3f}, "
#                 f"end_mask_ratio={self._end_mask_ratio:.3f}, "
#             )
#             # Defer patch_dim calculation to forward, define decoder without fixed output
#             self.decoder = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(inplace=False),
#                 nn.Linear(512, 512),  # Temporary output, adjust in forward
#             )
#             self.decoder_output = nn.Linear(512, 512)  # Initial placeholder

#         # common projection head (used by contrastive & MoCo)
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.in_features, 512),
#             nn.BatchNorm1d(512),  # Added batch norm
#             nn.ReLU(),
#             nn.Dropout(0.3),  # Added dropout with rate 0.3
#             nn.Linear(512, 128),
#         )

#         # MoCo momentum encoder & queue
#         if self.pretrain_method == 'moco':
#             self.momentum_encoder = video_models.r3d_18(pretrained=pretrained)
#             self.momentum_encoder.fc = nn.Identity()
#             # freeze momentum parameters
#             for p in self.momentum_encoder.parameters():
#                 p.requires_grad = False
#             # projection head for keys
#             self.momentum_projection_head = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.BatchNorm1d(512),  # Added batch norm
#                 nn.ReLU(),
#                 nn.Dropout(0.3),  # Added dropout with rate 0.3
#                 nn.Linear(512, 128),
#             )
#             self.momentum = 0.999
#             self.queue_size = 4096
#             self.register_buffer('queue', torch.randn(self.queue_size, 128))  # placeholder
#             self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

#         # Supervised classification heads with dropout
#         if self.pretrain_method not in ('contrastive', 'moco', 'mae'):
#             if self.hierarchical:
#                 self.model.fc = nn.Identity()
#                 self.base_head = nn.Sequential(
#                     nn.BatchNorm1d(self.in_features),  # Added batch norm
#                     nn.Dropout(0.5),
#                     nn.Linear(self.in_features, num_base_classes)
#                 )
#                 self.subclass_head = nn.Sequential(
#                     nn.BatchNorm1d(self.in_features),  # Added batch norm
#                     nn.Dropout(0.5),
#                     nn.Linear(self.in_features, num_subclasses)
#                 )
#             else:
#                 self.model.fc = nn.Sequential(
#                     nn.BatchNorm1d(self.in_features),  # Added batch norm
#                     nn.Dropout(0.5),
#                     nn.Linear(self.in_features, num_base_classes)
#                 )
#                 logger.info(
#                     f"Set supervised fc to Linear({self.in_features}, {num_base_classes}) with batch norm and dropout"
#                 )

#     def set_mask_ratio(self, mask_ratio):
#         """Update mask_ratio and recompute num_masked."""
#         if self.pretrain_method == 'mae':
#             if not (0 <= mask_ratio <= 1):
#                 raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
#             self._mask_ratio = mask_ratio
#             logger.info(f"Updated mask_ratio to {self._mask_ratio:.3f}")
#         else:
#             logger.warning("set_mask_ratio called but pretrain_method is not 'mae'")

#     @property
#     def mask_ratio(self):
#         return self._mask_ratio

#     @property
#     def end_mask_ratio(self):
#         return self._end_mask_ratio
    
#     @property
#     def num_masked(self):
#         # This will be overridden by forward based on mask_indices
#         return int(self._mask_ratio * self.total_patches) if hasattr(self, 'total_patches') else 0

#     def _update_momentum_encoder(self):
#         # Momentum update: key_encoder = m * key_encoder + (1-m) * query_encoder
#         for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):
#             param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
#         for qh, kh in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
#             kh.data = kh.data * self.momentum + qh.data * (1.0 - self.momentum)

#     def _enqueue_and_dequeue(self, keys):
#         # keys: [B, 128]
#         batch_size = keys.size(0)
#         ptr = int(self.queue_ptr)
#         # replace the oldest entries
#         if ptr + batch_size <= self.queue_size:
#             self.queue[ptr:ptr+batch_size] = keys.detach()
#         else:
#             # wrap-around
#             end = self.queue_size - ptr
#             self.queue[ptr:] = keys[:end].detach()
#             self.queue[: batch_size - end] = keys[end:].detach()
#         self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

#     def forward(self, x, x2=None, mask_indices=None):
#         if self.pretrain_method == 'mae':
#             # x: [B, 3, T, H, W]  -> torch.Size([8, 3, 16, 224, 224])
#             B, C, T, H, W = x.shape  # Derive dimensions from input
#             self.T = T
#             self.frame_size = H  # Update frame_size dynamically
#             self.num_patches_per_frame = (H // self.patch_size) * (W // self.patch_size) # 196
#             self.total_patches = T * self.num_patches_per_frame  # 3136
#             self.patch_dim = C * self.patch_size * self.patch_size # 768

#             # import pdb; pdb.set_trace()
#             # Update decoder output layer to match patch_dim
#             if self.decoder_output.out_features != self.patch_dim:
#                 self.decoder_output = nn.Linear(512, self.patch_dim).to(x.device)
#                 logger.info(f"Updated decoder output layer to match patch_dim={self.patch_dim}")

#             feat = self.model(x)  # [B, in_features]  # [8, 512]
#             out = self.decoder(feat)  # [B, 512]
#             out = self.decoder_output(out)  # [B, patch_dim]  # [8, 768]
#             num_masked = mask_indices.size(1) if mask_indices.dim() > 1 else mask_indices.size(0)   # 624 and 8, respectively
#             out = out.view(B, 1, self.patch_dim).repeat(1, num_masked, 1)  # [B, num_masked, patch_dim]   # torch.Size([8, 624, 768])

#             # Scatter into full patch grid
#             device, dtype = out.device, out.dtype
#             full = torch.zeros(
#                 (B, self.total_patches, self.patch_dim),
#                 device=device, dtype=dtype
#             )
#             if mask_indices.dim() == 1:
#                 mask_indices = mask_indices.unsqueeze(0).expand(B, -1)
#             for i in range(B):
#                 full[i, mask_indices[i]] = out[i]

#                 # video = unpatchify(full, self.patch_size, self.T, self.frame_size, self.frame_size)
#                 video = unpatchify(full, self.patch_size, self.T, H, W)
#                 return video

#         if self.pretrain_method == 'contrastive':
#             # require two views
#             assert x2 is not None, "Contrastive mode needs both x and x2"
#             q = self.projection_head(self.model(x))   # queries
#             k = self.projection_head(self.model(x2))  # keys for contrastive loss
#             return q, k

#         if self.pretrain_method == 'moco':
#             # query
#             q = self.projection_head(self.model(x))
#             # update key encoder and build key
#             with torch.no_grad():
#                 self._update_momentum_encoder()
#                 k = self.momentum_projection_head(self.momentum_encoder(x))
#             # enqueue & dequeue
#             self._enqueue_and_dequeue(k)
#             return q, k

#         # supervised forward
#         feat = self.model(x)
#         if self.hierarchical:
#             return self.base_head(feat), self.subclass_head(feat)
#         return feat