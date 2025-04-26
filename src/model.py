import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Compute contrastive loss between two views (z1, z2) of the same video.
        Args:
            z1, z2: Feature embeddings [batch_size, dim]
        """
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(z1.device)
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

class ViViT(nn.Module):
    def __init__(self, num_classes, image_size=(224, 224), num_frames=16, patch_size=16, dim=192, depth=12, heads=3, ssl_mode=False):
        super(ViViT, self).__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.ssl_mode = ssl_mode
        
        # Patch embedding
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size) * num_frames
        self.patch_embed = nn.Conv3d(3, dim, kernel_size=(3, patch_size, patch_size), stride=(1, patch_size, patch_size))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4),
            num_layers=depth
        )
        
        # Classification head (used in supervised mode)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # Projection head for SSL
        self.ssl_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # For attention visualization
        self.attention_weights = []

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.patch_embed(x)  # [B, dim, T', H', W']
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)  # [B, num_patches, dim]
        
        x = x + self.pos_embed
        x = self.transformer(x)  # [B, num_patches, dim]
        
        # Pooling
        x = x.mean(dim=1)  # [B, dim]
        
        if self.ssl_mode:
            x = self.ssl_head(x)  # [B, 128] for contrastive learning
        else:
            x = self.mlp_head(x)  # [B, num_classes]
        
        return x
    
    def get_attention_maps(self, x):
        # Placeholder for attention visualization
        # Implement based on your needs (e.g., extract attention weights from transformer)
        return self.attention_weights

def get_model(model_name, num_classes, pretrained=True, ssl_mode=False):
    if model_name == 'i3d':
        model = video_models.r3d_18(pretrained=pretrained)
        if ssl_mode:
            # Replace classification head with SSL projection head
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vivit':
        model = ViViT(num_classes=num_classes, ssl_mode=ssl_mode)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# --------------------------------------------------------
#  original without ssl

# import torch
# import torch.nn as nn
# import torchvision.models.video as video_models

# class ViViT(nn.Module):
#     def __init__(self, num_classes, image_size=(224, 224), num_frames=16, patch_size=16, dim=192, depth=12, heads=3):
#         super(ViViT, self).__init__()
#         self.num_frames = num_frames
#         self.image_size = image_size
#         self.patch_size = patch_size
        
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
        
#         # Classification head
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
        
#         # For attention visualization
#         self.attention_weights = []

#     def forward(self, x):
#         # x: [B, C, T, H, W]
#         x = self.patch_embed(x)  # [B, dim, T', H', W']
#         B, C, T, H, W = x.shape
#         x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)  # [B, num_patches, dim]
        
#         x = x + self.pos_embed
#         x = self.transformer(x)  # [B, num_patches, dim]
        
#         # Pooling
#         x = x.mean(dim=1)  # [B, dim]
#         x = self.mlp_head(x)  # [B, num_classes]
        
#         return x
    
#     def get_attention_maps(self, x):
#         # Placeholder for attention visualization
#         # Implement based on your needs (e.g., extract attention weights from transformer)
#         return self.attention_weights

# def get_model(model_name, num_classes, pretrained=True):
#     if model_name == 'i3d':
#         model = video_models.r3d_18(pretrained=pretrained)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)
#     elif model_name == 'vivit':
#         model = ViViT(num_classes=num_classes)
#     else:
#         raise ValueError(f"Unknown model: {model_name}")
#     return model