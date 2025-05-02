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
