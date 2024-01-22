import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()

        self.embedding_size = embedding_size

        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_size, 2).float() / embedding_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        positions = torch.arange(0, x.shape[1], device=x.device)

        pos_emb = torch.einsum("i,j->ij", positions, self.inv_freq)
        pe = torch.zeros(x.shape[1], self.embedding_size, device=x.device)
        pe[:, 0::2] = torch.sin(pos_emb)
        pe[:, 1::2] = torch.cos(pos_emb)
        self.cache = pe

        return pe