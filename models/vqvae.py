import torch
import einops
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder
from models.modules.quantization import FSQ
from models.modules.features2embedding import Embedding2Feature, Feature2Embedding


class MidiVQVAE(nn.Module):
    def __init__(
        self, 
        dim: int, 
        dim_mults: list[int] = [1, 2, 4, 8], 
        fsq_levels: list[int] = [8, 8, 6, 5, 5], 
        resnet_block_groups: int = 4,
        causal: bool = False,
    ):
        super().__init__()

        # list of dimensions
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        quantization_dim = len(fsq_levels)

        # embedding heads
        self.pitch_embedding = nn.Embedding(num_embeddings=88, embedding_dim=dim)
        self.velocity_embedding = Feature2Embedding(1, dim)
        self.dstart_embedding = Feature2Embedding(1, dim)
        self.duration_embedding = Feature2Embedding(1, dim)

        # initial conv
        self.init_conv = nn.Conv1d(4 * dim, dim, kernel_size=1)

        # encoder
        self.encoder = Encoder(dims, resnet_block_groups, causal=causal)

        # fsq
        self.q_proj_in = nn.Conv1d(dims[-1], quantization_dim, kernel_size=1)
        self.quantizer = FSQ(levels=fsq_levels)
        self.q_proj_out = nn.Conv1d(quantization_dim, dims[-1], kernel_size=1)

        # decoder
        self.decoder = Decoder(dims, resnet_block_groups, causal=causal)

        # out conv
        self.out_conv = nn.Conv1d(dim, 4 * dim, kernel_size=1)

        # output heads
        self.pitch_out = Embedding2Feature(4 * dim, 88)
        self.velocity_out = Feature2Embedding(4 * dim, 1)
        self.dstart_out = Feature2Embedding(4 * dim, 1)
        self.duration_out = Feature2Embedding(4 * dim, 1)

    def _features_to_embedding(self, pitch: torch.Tensor, velocity: torch.Tensor, dstart: torch.Tensor, duration: torch.Tensor):
        pitch_emb = self.pitch_embedding(pitch)
        velocity_emb = self.velocity_embedding(velocity[:, :, None])
        dstart_emb = self.dstart_embedding(dstart[:, :, None])
        duration_emb = self.duration_embedding(duration[:, :, None])

        # shape: [batch_size, seq_len, embedding_dim]
        x = torch.cat([pitch_emb, velocity_emb, dstart_emb, duration_emb], dim=-1)
        # shape: [batch_size, embedding_dim, seq_len] for convolution
        x = x.permute(0, 2, 1)

        return x

    def _embedding_to_features(self, embedding: torch.Tensor):
        # shape: [batch_size, embedding_dim, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedding = embedding.permute(0, 2, 1)

        pitch = self.pitch_out(embedding)
        velocity = self.velocity_out(embedding).squeeze(-1)
        dstart = self.dstart_out(embedding).squeeze(-1)
        duration = self.duration_out(embedding).squeeze(-1)

        return pitch, velocity, dstart, duration

    def quantize(self, pitch: torch.Tensor, velocity: torch.Tensor, dstart: torch.Tensor, duration: torch.Tensor):
        embedding = self._features_to_embedding(pitch, velocity, dstart, duration)

        x = self.init_conv(embedding)
        x = self.encoder(x)

        x = self.q_proj_in(x)
        x = einops.rearrange(x, "b c l -> b l c")
        _, indices = self.quantizer(x)

        return indices

    def dequantize(self, indices: torch.Tensor):
        codes = self.quantizer.indices_to_codes(indices)
        x = einops.rearrange(codes, "b l c -> b c l")
        x = self.q_proj_out(x)

        x = self.decoder(x)
        x = self.out_conv(x)

        pitch_hat, velocity_hat, dstart_hat, duration_hat = self._embedding_to_features(x)

        return pitch_hat, velocity_hat, dstart_hat, duration_hat

    def forward(self, pitch: torch.Tensor, velocity: torch.Tensor, dstart: torch.Tensor, duration: torch.Tensor):
        embedding = self._features_to_embedding(pitch, velocity, dstart, duration)

        x = self.init_conv(embedding)
        x = self.encoder(x)

        x = self.q_proj_in(x)
        x = einops.rearrange(x, "b c l -> b l c")
        codes, _ = self.quantizer(x)
        x = einops.rearrange(codes, "b l c -> b c l")
        x = self.q_proj_out(x)

        x = self.decoder(x)
        x = self.out_conv(x)

        pitch_hat, velocity_hat, dstart_hat, duration_hat = self._embedding_to_features(x)

        return pitch_hat, velocity_hat, dstart_hat, duration_hat
