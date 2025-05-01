import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.xent = nn.CrossEntropyLoss()

    def forward(self, q, k):
        """
        q,k: [B, D] normalized embeddings of two views.
        """
        B = q.size(0)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # logits_ij = q ⋅ kᵀ / τ
        logits_12 = q @ k.t() / self.temperature    # [B,B]
        logits_21 = k @ q.t() / self.temperature    # [B,B]
        labels   = torch.arange(B, device=q.device)

        loss = (
            self.xent(logits_12, labels)
          + self.xent(logits_21, labels)
        ) * 0.5
        return loss


class MoCoLoss(nn.Module):
    def __init__(self, queue, temperature=0.07):
        """
        queue: a Tensor [N, D] of negative keys (should be the same buffer your model enqueues).
        """
        super().__init__()
        self.queue = queue
        self.temperature = temperature
        self.xent = nn.CrossEntropyLoss()

    def forward(self, q, k):
        """
        q: [B, D] queries, k: [B, D] positive keys
        queue: [N, D] negative keys
        """
        B = q.size(0)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        neg = F.normalize(self.queue, dim=1)

        # positive logits: B×1
        # l_pos = torch.einsum('bd,bd->b1', [q, k])  #wrong
        
        # approach 1: returns a vector of length B, then unsqueeze to [B,1]
        l_pos = torch.einsum('bd,bd->b', [q, k]).unsqueeze(1)

        # approach 2: elementwise multiply, sum over dim=1, keepdim to get [B,1]
        # l_pos = (q * k).sum(dim=1, keepdim=True)

        # negative logits: B×N
        l_neg = q @ neg.t()   # [B,N]

        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # [B, 1+N]
        labels = torch.zeros(B, dtype=torch.long, device=q.device)    # positives are index 0

        loss = self.xent(logits, labels)
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reconstructed, original, mask):
        mask = mask.unsqueeze(1)                                # [B,1,T,H,W]
        diff = (reconstructed - original).pow(2)
        return (diff * mask).sum() / (mask.sum().clamp_min(1.0))



# ---------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models.video as video_models
# from collections import deque
# import logging

# logger = logging.getLogger(__name__)


# class ReconstructionLoss(torch.nn.Module):
#     def __init__(self):
#         super(ReconstructionLoss, self).__init__()

#     def forward(self, reconstructed, original, mask):
#         """
#         Compute reconstruction loss for MAE, focusing on masked regions.

#         Args:
#             reconstructed: Reconstructed video tensor [batch, channels, time, height, width].
#             original: Original video tensor [batch, channels, time, height, width].
#             mask: Binary mask [batch, time, height, width], 1 for masked regions.
#         """
#         # Expand mask to match channels dimension
#         mask = mask.unsqueeze(1)  # [batch, 1, time, height, width]
#         # Compute MSE loss only on masked regions
#         diff = (reconstructed - original) ** 2
#         loss = (diff * mask).sum() / (mask.sum() + 1e-6)  # Avoid division by zero
#         return loss

# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, output1, output2):
#         """
#         Compute symmetrized contrastive loss between two views of the same video.

#         Args:
#             output1, output2: Feature embeddings [batch_size, dim]

#         Returns:
#             torch.Tensor: Contrastive loss
#         """
#         batch_size = output1.size(0)
#         # Normalize embeddings
#         output1 = F.normalize(output1, dim=1)
#         output2 = F.normalize(output2, dim=1)

#         # Compute similarity matrices (symmetric)
#         logits_12 = torch.matmul(output1, output2.T) / self.temperature  # [batch_size, batch_size]
#         logits_21 = torch.matmul(output2, output1.T) / self.temperature  # [batch_size, batch_size]

#         # Labels: positive pairs are on the diagonal
#         labels = torch.arange(batch_size, device=output1.device)

#         # Compute losses for both directions
#         loss_12 = self.criterion(logits_12, labels)
#         loss_21 = self.criterion(logits_21, labels)

#         # Symmetrize the loss
#         loss = (loss_12 + loss_21) / 2

#         # Log for debugging
#         logger.debug(f"Contrastive loss: {loss.item():.4f}, logits range: [{logits_12.min().item():.4f}, {logits_12.max().item():.4f}]")
#         logger.debug(f"Output1 norm: {output1.norm(dim=1).mean().item():.4f}, Output2 norm: {output2.norm(dim=1).mean().item():.4f}")

#         return loss
    

# class ContrastiveLoss(torch.nn.Module):
#     def __init__(self, temperature=0.5):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, feat1, feat2):
#         batch_size = feat1.size(0)
#         feat1 = torch.nn.functional.normalize(feat1, dim=1)
#         feat2 = torch.nn.functional.normalize(feat2, dim=1)
#         sim_matrix = torch.mm(feat1, feat2.t()) / self.temperature
#         labels = torch.arange(batch_size).to(feat1.device)
#         loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
#         return loss