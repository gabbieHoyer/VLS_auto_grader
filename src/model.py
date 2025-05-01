import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from collections import deque
import logging

from utils.ssl_helpers import patchify, unpatchify

# Set up logging
logger = logging.getLogger(__name__)

class I3D(nn.Module):
    def __init__(
        self,
        num_base_classes,
        num_subclasses=0,
        pretrained=True,
        pretrain_method=None,
        patch_size=16,
        mask_ratio=0.75,
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

        if pretrain_method == 'mae':
            self.frame_size = 224
            self.patch_size = patch_size
            self.T = 16
            self.num_patches_per_frame = (224 // patch_size) ** 2
            self.total_patches = self.T * self.num_patches_per_frame
            self.mask_ratio = mask_ratio       # pass this into your model ctor
            self.num_masked = int(self.mask_ratio * self.total_patches)
            self.patch_dim = 3 * patch_size * patch_size

            self.decoder = nn.Sequential(
                nn.Linear(self.in_features, 512),
                nn.ReLU(inplace=False),
                nn.Linear(512, self.num_masked * self.patch_dim),
            )

        # common projection head (used by contrastive & MoCo)
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
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
                nn.Linear(512, 128),
            )
            self.momentum = 0.999
            self.queue_size = 4096
            self.register_buffer('queue', torch.randn(self.queue_size, 128))  # placeholder
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # supervised classification heads
        if self.pretrain_method not in ('contrastive', 'moco', 'mae'):
            if self.hierarchical:
                self.model.fc = nn.Identity()
                self.base_head = nn.Linear(self.in_features, num_base_classes)
                self.subclass_head = nn.Linear(self.in_features, num_subclasses)
            else:
                self.model.fc = nn.Linear(self.in_features, num_base_classes)
                logger.info(
                    f"Set supervised fc to Linear({self.in_features}, {num_base_classes})"
                )

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
        # x: [B, 3, T, H, W]
        if self.pretrain_method == 'mae':
            B = x.size(0)
            feat = self.model(x)                              # [B, in_features]
            out  = self.decoder(feat)                         # [B, num_masked*patch_dim]
            out  = out.view(B, self.num_masked, self.patch_dim)

            # Scatter into full patch grid:
            device, dtype = out.device, out.dtype
            full = torch.zeros(
                (B, self.total_patches, self.patch_dim),
                device=device, dtype=dtype
            )
            if mask_indices.dim() == 1:
                mask_indices = mask_indices.unsqueeze(0).expand(B, -1)
            for i in range(B):
                full[i, mask_indices[i]] = out[i]

            # Only unpatchify is needed here:
            video = unpatchify(full, self.patch_size, self.T, self.frame_size, self.frame_size)
            # use the *full* frame height & width (224), not patch_size
            # video = unpatchify(full, self.patch_size, self.T, 224, 224)
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

def get_model(model_name, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None):
    """
    Initialize and return a model based on the specified name.

    Args:
        model_name (str): Name of the model ('i3d' or 'vivit').
        num_base_classes (int): Number of base classes (e.g., 8 for single-grader, 4 for hierarchical).
        num_subclasses (int): Number of subclasses (0 for single-grader, 5 for hierarchical).
        pretrained (bool): Whether to load pretrained weights (for I3D or ViViT backbone).
        pretrain_method (str, optional): Pretraining method ('contrastive', 'moco', 'mae') or None for supervised.

    Returns:
        nn.Module: Initialized model.
    """
    if model_name == 'i3d':
        return I3D(num_base_classes, num_subclasses, pretrained, pretrain_method)
    elif model_name == 'vivit':
        return ViViT(num_base_classes, num_subclasses, pretrained, pretrain_method)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

# ----------------------------------------------------------------------------

        # MAE-specific decoder setup
        # if self.pretrain_method == 'mae':
        #     # compute number of patches
        #     num_patches_per_frame = (224 // patch_size) ** 2
        #     total_patches = num_patches_per_frame * 16  # 16 frames
        #     patch_dim = 3 * patch_size * patch_size
        #     self.decoder = nn.Sequential(
        #         nn.Linear(self.in_features, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, total_patches * patch_dim),
        #     )

        # if self.pretrain_method == 'mae':
        #     patches = patchify(x, self.patch_size)  # [B, P, patch_dim]
        #     feat = self.model(x)                    # [B, in_features]
        #     out = self.decoder(feat)                # [B, P * patch_dim]
        #     out = out.view(x.size(0), patches.size(1), -1)
        #     recon = unpatchify(out, self.patch_size, 16, 224, 224)
        #     return recon


# class I3D(nn.Module):
#     def __init__(self, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None):
#         super(I3D, self).__init__()
#         self.model = video_models.r3d_18(pretrained=pretrained)
#         self.in_features = self.model.fc.in_features  # Should be 512 for r3d_18
#         self.pretrain_method = pretrain_method
#         self.hierarchical = num_subclasses > 0

#         # Debug log to check pretrain_method
#         logger.info(f"Initializing I3D with pretrain_method: {pretrain_method}")

#         # For contrastive or MoCo, we need raw features, not class logits
#         # Handle case-sensitivity by converting to lowercase
#         pretrain_method_lower = pretrain_method.lower() if pretrain_method else None
#         if pretrain_method_lower in ['contrastive', 'moco']:
#             self.model.fc = nn.Identity()  # Remove the fc layer to get raw features
#             logger.info("Set self.model.fc to nn.Identity() for contrastive/MoCo pretraining")
#         elif pretrain_method_lower == 'mae':
#             # Encoder-decoder for MAE
#             self.model.fc = nn.Identity()
#             self.decoder = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 3 * 16 * 224 * 224 // (16 * 16)), ## <-- 9408 dims # Reconstruct to [C, T, H, W] after patching
#                 nn.ReLU()
#             )
#         else:
#             # Supervised learning setup
#             if self.hierarchical:
#                 self.model.fc = nn.Identity()
#                 self.base_head = nn.Linear(self.in_features, num_base_classes)
#                 self.subclass_head = nn.Linear(self.in_features, num_subclasses)
#             else:
#                 self.model.fc = nn.Linear(self.in_features, num_base_classes)
#                 logger.info(f"Set self.model.fc to nn.Linear({self.in_features}, {num_base_classes}) for supervised learning")

#         # Projection head for contrastive learning (SimCLR or MoCo)
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.in_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128)
#         )

#         if pretrain_method_lower == 'moco':
#             # Momentum encoder for MoCo
#             self.momentum_encoder = video_models.r3d_18(pretrained=pretrained)
#             self.momentum_encoder.fc = nn.Identity()
#             for param in self.momentum_encoder.parameters():
#                 param.requires_grad = False
#             self.momentum_projection_head = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 128)
#             )
#             self.momentum = 0.999  # Momentum update factor
#             self.queue_size = 4096
#             self.queue = deque(maxlen=self.queue_size)
#             self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     def _update_momentum_encoder(self):
#         with torch.no_grad():
#             for param, param_m in zip(self.model.parameters(), self.momentum_encoder.parameters()):
#                 new_data = param_m.data * self.momentum + param.data * (1. - self.momentum)
#                 param_m.copy_(new_data)
#             for param, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
#                 new_data = param_m.data * self.momentum + param.data * (1. - self.momentum)
#                 param_m.copy_(new_data)

#     def _enqueue_and_dequeue(self, keys):
#         batch_size = keys.size(0)
#         ptr = int(self.queue_ptr)
#         self.queue.extend(keys.detach().cpu().numpy())
#         if len(self.queue) >= self.queue_size:
#             self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

#     def forward(self, x, x2=None):
#         # Handle case-sensitivity in pretrain_method
#         pretrain_method_lower = self.pretrain_method.lower() if self.pretrain_method else None
#         logger.debug(f"Forward pass with pretrain_method: {pretrain_method_lower}")

#         if pretrain_method_lower in ['contrastive', 'moco']:
#             feat = self.model(x)
#             logger.debug(f"Shape of feat after self.model(x): {feat.shape}")
#             feat = self.projection_head(feat)
#             if pretrain_method_lower == 'moco':
#                 with torch.no_grad():
#                     self._update_momentum_encoder()
#                     keys = self.momentum_encoder(x)
#                     keys = self.momentum_projection_head(keys)
#                     self._enqueue_and_dequeue(keys)
#                 return feat, keys
#             if x2 is not None:
#                 feat2 = self.model(x2)
#                 feat2 = self.projection_head(feat2)
#                 return feat, feat2
#             return feat
#         elif pretrain_method_lower == 'mae':
#             feat = self.model(x)
#             reconstructed = self.decoder(feat)
#             reconstructed = reconstructed.view(-1, 3, 16, 224, 224)  # 2 408 448 elements  # Reshape to [batch, C, T, H, W]
#             return reconstructed
#         else:
#             feat = self.model(x)
#             if self.hierarchical:
#                 base_logits = self.base_head(feat)
#                 subclass_logits = self.subclass_head(feat)
#                 return base_logits, subclass_logits
#             return feat


# -----------------------------------------------------
    
# worked before ssl:

# class I3D(nn.Module):
#     def __init__(self, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None):
#         super(I3D, self).__init__()
#         self.model = video_models.r3d_18(pretrained=pretrained)
#         self.in_features = self.model.fc.in_features
#         self.pretrain_method = pretrain_method
#         self.hierarchical = num_subclasses > 0

#         # Projection head for contrastive learning (SimCLR or MoCo)
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.in_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128)
#         )

#         if pretrain_method == 'moco':
#             # Momentum encoder for MoCo
#             self.momentum_encoder = video_models.r3d_18(pretrained=pretrained)
#             self.momentum_encoder.fc = nn.Identity()
#             for param in self.momentum_encoder.parameters():
#                 param.requires_grad = False
#             self.momentum_projection_head = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 128)
#             )
#             self.momentum = 0.999  # Momentum update factor
#             self.queue_size = 4096
#             self.queue = deque(maxlen=self.queue_size)
#             self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#         elif pretrain_method == 'mae':
#             # Encoder-decoder for MAE
#             self.model.fc = nn.Identity()
#             self.decoder = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 3 * 16 * 224 * 224 // (16 * 16)),  # Reconstruct to [C, T, H, W] after patching
#                 nn.ReLU()
#             )
#         else:
#             # Supervised learning setup
#             if self.hierarchical:
#                 self.model.fc = nn.Identity()
#                 self.base_head = nn.Linear(self.in_features, num_base_classes)
#                 self.subclass_head = nn.Linear(self.in_features, num_subclasses)
#             else:
#                 self.model.fc = nn.Linear(self.in_features, num_base_classes)

#     def _update_momentum_encoder(self):
#         for param, param_m in zip(self.model.parameters(), self.momentum_encoder.parameters()):
#             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
#         for param, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
#             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

#     def _enqueue_and_dequeue(self, keys):
#         batch_size = keys.size(0)
#         ptr = int(self.queue_ptr)
#         self.queue.extend(keys.detach().cpu().numpy())
#         if len(self.queue) >= self.queue_size:
#             self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

#     def forward(self, x, x2=None):
#         if self.pretrain_method in ['contrastive', 'moco']:
#             feat = self.model(x)
#             feat = self.projection_head(feat)
#             if self.pretrain_method == 'moco':
#                 with torch.no_grad():
#                     self._update_momentum_encoder()
#                     keys = self.momentum_encoder(x)
#                     keys = self.momentum_projection_head(keys)
#                     self._enqueue_and_dequeue(keys)
#                 return feat, keys
#             if x2 is not None:
#                 feat2 = self.model(x2)
#                 feat2 = self.projection_head(feat2)
#                 return feat, feat2
#             return feat
#         elif self.pretrain_method == 'mae':
#             feat = self.model(x)
#             reconstructed = self.decoder(feat)
#             reconstructed = reconstructed.view(-1, 3, 16, 224, 224)  # Reshape to [batch, C, T, H, W]
#             return reconstructed
#         else:
#             feat = self.model(x)
#             if self.hierarchical:
#                 base_logits = self.base_head(feat)
#                 subclass_logits = self.subclass_head(feat)
#                 return base_logits, subclass_logits
#             return feat

#     def get_attention_maps(self, x):
#         return None  # I3D doesn't use attention



# class ViViT(nn.Module):
#     def __init__(self, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None):
#         super(ViViT, self).__init__()
#         self.patch_size = 16
#         self.num_patches = (16 // self.patch_size) * (224 // self.patch_size) * (224 // self.patch_size)
#         self.embed_dim = 768
#         self.encoder = nn.Sequential(
#             nn.Conv3d(3, self.embed_dim, kernel_size=(2, self.patch_size, self.patch_size), stride=(2, self.patch_size, self.patch_size)),
#             nn.Flatten(start_dim=2),
#             nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, dim_feedforward=3072),
#                 num_layers=12
#             )
#         )
#         self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
#         self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_dim))
#         self.in_features = self.embed_dim
#         self.pretrain_method = pretrain_method
#         self.hierarchical = num_subclasses > 0

#         # Projection head for contrastive learning
#         self.projection_head = nn.Sequential(
#             nn.Linear(self.in_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128)
#         )

#         if pretrain_method == 'moco':
#             # Momentum encoder for MoCo
#             self.momentum_encoder = nn.Sequential(
#                 nn.Conv3d(3, self.embed_dim, kernel_size=(2, self.patch_size, self.patch_size), stride=(2, self.patch_size, self.patch_size)),
#                 nn.Flatten(start_dim=2),
#                 nn.TransformerEncoder(
#                     nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12, dim_feedforward=3072),
#                     num_layers=12
#                 )
#             )
#             for param in self.momentum_encoder.parameters():
#                 param.requires_grad = False
#             self.momentum_projection_head = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 128)
#             )
#             self.momentum = 0.999
#             self.queue_size = 4096
#             self.queue = deque(maxlen=self.queue_size)
#             self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#         elif pretrain_method == 'mae':
#             # Decoder for MAE
#             self.decoder = nn.Sequential(
#                 nn.TransformerDecoder(
#                     nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=12, dim_feedforward=3072),
#                     num_layers=4
#                 ),
#                 nn.Linear(self.embed_dim, 3 * 2 * self.patch_size * self.patch_size),
#                 nn.ReLU()
#             )
#         else:
#             # Supervised learning setup
#             if self.hierarchical:
#                 self.base_head = nn.Linear(self.in_features, num_base_classes)
#                 self.subclass_head = nn.Linear(self.in_features, num_subclasses)
#             else:
#                 self.head = nn.Linear(self.in_features, num_base_classes)

#     def _update_momentum_encoder(self):
#         for param, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
#             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
#         for param, param_m in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
#             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

#     def _enqueue_and_dequeue(self, keys):
#         batch_size = keys.size(0)
#         ptr = int(self.queue_ptr)
#         self.queue.extend(keys.detach().cpu().numpy())
#         if len(self.queue) >= self.queue_size:
#             self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

#     def forward(self, x, x2=None):
#         batch_size = x.size(0)
#         # Extract features
#         x = self.encoder[0](x)  # [batch, embed_dim, T', H', W']
#         x = self.encoder[1](x)  # [batch, embed_dim, num_patches]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_tokens, x.permute(0, 2, 1)), dim=1)
#         x = x + self.pos_embed
#         feat = self.encoder[2](x)[:, 0]  # Take CLS token

#         if self.pretrain_method in ['contrastive', 'moco']:
#             feat = self.projection_head(feat)
#             if self.pretrain_method == 'moco':
#                 with torch.no_grad():
#                     self._update_momentum_encoder()
#                     keys = self.momentum_encoder[0](x)
#                     keys = self.momentum_encoder[1](keys)
#                     cls_tokens_m = self.cls_token.expand(batch_size, -1, -1)
#                     keys = torch.cat((cls_tokens_m, keys.permute(0, 2, 1)), dim=1)
#                     keys = keys + self.pos_embed
#                     keys = self.momentum_encoder[2](keys)[:, 0]
#                     keys = self.momentum_projection_head(keys)
#                     self._enqueue_and_dequeue(keys)
#                 return feat, keys
#             if x2 is not None:
#                 x2 = self.encoder[0](x2)
#                 x2 = self.encoder[1](x2)
#                 cls_tokens2 = self.cls_token.expand(batch_size, -1, -1)
#                 x2 = torch.cat((cls_tokens2, x2.permute(0, 2, 1)), dim=1)
#                 x2 = x2 + self.pos_embed
#                 feat2 = self.encoder[2](x2)[:, 0]
#                 feat2 = self.projection_head(feat2)
#                 return feat, feat2
#             return feat
#         elif self.pretrain_method == 'mae':
#             # For MAE, assume input x is already masked
#             # Decoder reconstructs the patches
#             reconstructed = self.decoder(feat.unsqueeze(1), feat.unsqueeze(1))
#             reconstructed = reconstructed.view(batch_size, -1, 3, 2, self.patch_size, self.patch_size)
#             reconstructed = reconstructed.permute(0, 2, 3, 4, 5, 1)  # [batch, C, T', H', W', num_patches]
#             reconstructed = reconstructed.reshape(batch_size, 3, 16, 224, 224)  # Upscale to original size
#             return reconstructed
#         else:
#             if self.hierarchical:
#                 base_logits = self.base_head(feat)
#                 subclass_logits = self.subclass_head(feat)
#                 return base_logits, subclass_logits
#             return self.head(feat)

#     def get_attention_maps(self, x):
#         # Simplified attention extraction for ViViT
#         x = self.encoder[0](x)
#         x = self.encoder[1](x)
#         cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
#         x = torch.cat((cls_tokens, x.permute(0, 2, 1)), dim=1)
#         x = x + self.pos_embed
#         attn_layers = self.encoder[2].layers
#         attn_maps = []
#         for layer in attn_layers:
#             attn = layer.self_attn(x, x, x)[1]  # Attention weights
#             attn_maps.append(attn)
#         return attn_maps
    

# def get_model(model_name, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None):
#     """
#     Initialize and return a model based on the specified name.

#     Args:
#         model_name (str): Name of the model ('i3d' or 'vivit').
#         num_base_classes (int): Number of base classes (e.g., 8 for single-grader, 4 for hierarchical).
#         num_subclasses (int): Number of subclasses (0 for single-grader, 5 for hierarchical).
#         pretrained (bool): Whether to load pretrained weights (for I3D or ViViT backbone).
#         pretrain_method (str, optional): Pretraining method ('contrastive', 'moco', 'mae') or None for supervised.

#     Returns:
#         nn.Module: Initialized model.
#     """
#     if model_name == 'i3d':
#         return I3D(num_base_classes, num_subclasses, pretrained, pretrain_method)
#     elif model_name == 'vivit':
#         return ViViT(num_base_classes, num_subclasses, pretrained, pretrain_method)
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
    
# ----------------------------------------------------


# class I3D(nn.Module):
#     def __init__(self, num_base_classes, num_subclasses=0, pretrained=True, ssl_mode=False):
#         super(I3D, self).__init__()
#         self.model = video_models.r3d_18(pretrained=pretrained)
#         self.in_features = self.model.fc.in_features
#         self.ssl_mode = ssl_mode
#         self.hierarchical = num_subclasses > 0

#         if self.ssl_mode:
#             # Two-layer projection head for SSL
#             self.model.fc = nn.Sequential(
#                 nn.Linear(self.in_features, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 128)
#             )
#         else:
#             if self.hierarchical:
#                 # Two heads for hierarchical classification
#                 self.model.fc = nn.Identity()  # Remove the default FC layer
#                 self.base_head = nn.Linear(self.in_features, num_base_classes)
#                 self.subclass_head = nn.Linear(self.in_features, num_subclasses)
#             else:
#                 # Single head for supervised learning (single/multi-grader)
#                 self.model.fc = nn.Linear(self.in_features, num_base_classes)

#     def forward(self, x, x2=None):
#         if self.ssl_mode:
#             if x2 is not None:
#                 # SSL forward pass with two views
#                 feat1 = self.model(x)
#                 feat2 = self.model(x2)
#                 return feat1, feat2  # For contrastive loss
#             else:
#                 # Single view (e.g., for validation)
#                 return self.model(x)
#         else:
#             feat = self.model(x)
#             if self.hierarchical:
#                 # Return base and subclass logits
#                 base_logits = self.base_head(feat)
#                 subclass_logits = self.subclass_head(feat)
#                 return base_logits, subclass_logits
#             else:
#                 return feat  # Already passed through fc layer in non-hierarchical mode

#     def get_attention_maps(self, x):
#         return None  # I3D doesn't use attention


# class ViViT(nn.Module):
#     def __init__(self, num_base_classes, num_subclasses=0, image_size=(224, 224), num_frames=16, patch_size=16, dim=192, depth=12, heads=3, ssl_mode=False):
#         super(ViViT, self).__init__()
#         self.num_frames = num_frames
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.ssl_mode = ssl_mode
#         self.hierarchical = num_subclasses > 0

#         # Patch embedding
#         num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size) * num_frames
#         self.patch_embed = nn.Conv3d(3, dim, kernel_size=(3, patch_size, patch_size), stride=(1, patch_size, patch_size))

#         # Positional embedding
#         self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))

#         # Transformer
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4),
#             num_layers=depth
#         )

#         if self.ssl_mode:
#             # Projection head for SSL
#             self.head = nn.Sequential(
#                 nn.Linear(dim, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 128)
#             )
#         elif self.hierarchical:
#             # Two heads for hierarchical classification
#             self.base_head = nn.Sequential(
#                 nn.LayerNorm(dim),
#                 nn.Linear(dim, num_base_classes)
#             )
#             self.subclass_head = nn.Sequential(
#                 nn.LayerNorm(dim),
#                 nn.Linear(dim, num_subclasses)
#             )
#         else:
#             # Single head for supervised learning (single/multi-grader)
#             self.head = nn.Sequential(
#                 nn.LayerNorm(dim),
#                 nn.Linear(dim, num_base_classes)
#             )

#         # For attention visualization
#         self.attention_weights = []

#     def forward(self, x, x2=None):
#         # x: [B, C, T, H, W]
#         x = self.patch_embed(x)  # [B, dim, T', H', W']
#         B, C, T, H, W = x.shape
#         x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)  # [B, num_patches, dim]

#         x = x + self.pos_embed
#         x = self.transformer(x)  # [B, num_patches, dim]

#         # Pooling
#         x = x.mean(dim=1)  # [B, dim]

#         if self.ssl_mode:
#             if x2 is not None:
#                 # SSL forward pass with two views
#                 x2 = self.patch_embed(x2)
#                 B2, C2, T2, H2, W2 = x2.shape
#                 x2 = x2.permute(0, 2, 3, 4, 1).reshape(B2, T2*H2*W2, C2)
#                 x2 = x2 + self.pos_embed
#                 x2 = self.transformer(x2)
#                 x2 = x2.mean(dim=1)
#                 return self.head(x), self.head(x2)
#             else:
#                 return self.head(x)
#         elif self.hierarchical:
#             # Return base and subclass logits
#             base_logits = self.base_head(x)
#             subclass_logits = self.subclass_head(x)
#             return base_logits, subclass_logits
#         else:
#             return self.head(x)

#     def get_attention_maps(self, x):
#         # Placeholder for attention visualization
#         return self.attention_weights


# def get_model(model_name, num_base_classes, num_subclasses=0, pretrained=True, pretrain_method=None, ssl_mode=False):
#     """
#     Initialize and return a model based on the specified name.

#     Args:
#         model_name (str): Name of the model ('i3d' or 'vivit').
#         num_base_classes (int): Number of base classes (e.g., 8 for single-grader, 4 for hierarchical).
#         num_subclasses (int): Number of subclasses (0 for single-grader, 5 for hierarchical).
#         pretrained (bool): Whether to load pretrained weights (for I3D).
#         ssl_mode (bool): If True, configures model for SSL pretraining.

#     Returns:
#         nn.Module: Initialized model.
#     """

#     if pretrain_method != None:
#         ssl_mode=True
#     else:
#         ssl_mode=False

#     if model_name == 'i3d':
#         return I3D(num_base_classes, num_subclasses, pretrained, ssl_mode)
#     elif model_name == 'vivit':
#         return ViViT(num_base_classes=num_base_classes, num_subclasses=num_subclasses, ssl_mode=ssl_mode)
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
