import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLayer(nn.Module):

    def __init__(self, dim, n_codes, commitment_cost=0.25):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.commitment_cost = commitment_cost

        self.embed = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.embed.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, x):
        # x: (B, dim)
        d = self.embed.weight
        dist = torch.cdist(x, d)
        idx = dist.argmin(dim=-1)
        z_q = self.embed(idx)
        loss_vq = F.mse_loss(z_q, x.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), x)
        z_q = x + (z_q - x).detach()
        return z_q, idx, loss_vq


class RQVAEBlock(nn.Module):

    def __init__(self, dim, n_codes):
        super().__init__()
        self.vq = VQLayer(dim, n_codes)

    def forward(self, x):
        z_q, idx, loss_vq = self.vq(x)
        residual = x - z_q
        return z_q, idx, residual, loss_vq


class RQVAE(nn.Module):

    def __init__(self, dim, n_codes, n_levels=1):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.n_levels = n_levels
        self.blocks = nn.ModuleList([RQVAEBlock(dim, n_codes) for _ in range(n_levels)])

    def forward(self, x):
        # x: (B, dim)
        recon = torch.zeros_like(x)
        indices = []
        total_vq_loss = 0.0

        r = x
        for block in self.blocks:
            z_q, idx, r, loss_vq = block(r)
            recon = recon + z_q
            indices.append(idx)
            total_vq_loss += loss_vq

        return recon, indices, total_vq_loss
