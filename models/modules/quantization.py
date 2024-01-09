import torch
import einops
import numpy as np
import torch.nn as nn


def round_ste(z: torch.Tensor):
    z_hat = z + (torch.round(z) - z).detach()

    return z_hat


class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()

        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        _basis = torch.cumprod(torch.tensor([1.0] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis)

        codebook_size = np.prod(levels)

        implicit_codebook = self.indices_to_codes(torch.arange(codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def _scale_shift_inverse(self, z_hat: torch.Tensor):
        half_width = self._levels // 2
        # normalzing between [-1, 1]
        z_hat_normalized = (z_hat - half_width) / half_width

        return z_hat_normalized

    def _scale_shift(self, z_hat_normalized: torch.Tensor):
        half_width = self._levels // 2
        # denormalizing z_hat
        z_hat = half_width * z_hat_normalized + half_width

        return z_hat

    def indices_to_codes(self, indices: torch.Tensor):
        # shape: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        indices = einops.rearrange(indices, "... -> ... 1")
        # building hypercube representation: [batch_size, seq_len, levels]
        z_hat_non_centered = (indices // self._basis) % self._levels
        z_hat_normalized = self._scale_shift_inverse(z_hat_non_centered)

        return z_hat_normalized

    def codes_to_indices(self, z_hat: torch.Tensor):
        # denormalization
        z_hat = self._scale_shift(z_hat)
        indices = torch.sum(z_hat * self._basis, dim=-1).to(torch.int32)

        return indices

    def bound(self, z: torch.Tensor):
        half_l = (self._levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        # bounding z between [-1, 1]
        z_bounded = torch.tanh(z + shift) * half_l - offset

        return z_bounded

    def quantize(self, z: torch.Tensor):
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2

        return quantized / half_width

    def forward(self, z: torch.Tensor):
        # input shape: [batch_size, seq_len, embedding_size]
        # embedding_size should be the same as len(levels)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        return codes, indices
