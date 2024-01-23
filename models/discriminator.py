import torch
import torch.nn as nn

from models.modules.attention_layers import ConvAttention
from models.modules.features2embedding import Feature2Embedding
from models.modules.conv_layers import PreNorm, Residual, Downsample, ResnetBlock


class Discriminator(nn.Module):
    def __init__(self, dim: int, dim_mults: list[int] = [1, 2, 4, 8], resnet_block_groups: int = 4, causal: bool = False):
        super().__init__()

        # get list of corresponding ins and outs
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        ins_outs = list(zip(dims[:-1], dims[1:]))

        # embedding heads
        self.pitch_embedding = nn.Embedding(num_embeddings=88, embedding_dim=dim)
        self.velocity_embedding = Feature2Embedding(1, dim)
        self.dstart_embedding = Feature2Embedding(1, dim)
        self.duration_embedding = Feature2Embedding(1, dim)

        # initial conv
        self.init_conv = nn.Conv1d(4 * dim, dim, kernel_size=1)

        layers = []

        for idx, (ins, outs) in enumerate(ins_outs):
            layers += [
                ResnetBlock(ins, ins, kernel_size=9, groups=resnet_block_groups, causal=causal),
                ResnetBlock(ins, ins, kernel_size=9, groups=resnet_block_groups, causal=causal),
                Residual(PreNorm(ins, ConvAttention(ins, causal=causal))),
                Downsample(ins, outs),
            ]

        layers += [nn.Conv1d(dims[-1], 1, kernel_size=1)]

        self.discriminator = nn.Sequential(*layers)

    def _features_to_embedding(self, pitch: torch.Tensor, velocity: torch.Tensor, dstart: torch.Tensor, duration: torch.Tensor):
        # pitch_emb = self.pitch_embedding(pitch)
        # pitch shape: [batch_size, seq_len, num_pitches]
        pitch_emb = pitch @ self.pitch_embedding.weight
        velocity_emb = self.velocity_embedding(velocity[:, :, None])
        dstart_emb = self.dstart_embedding(dstart[:, :, None])
        duration_emb = self.duration_embedding(duration[:, :, None])

        # shape: [batch_size, seq_len, embedding_dim]
        x = torch.cat([pitch_emb, velocity_emb, dstart_emb, duration_emb], dim=-1)
        # shape: [batch_size, embedding_dim, seq_len] for convolution
        x = x.permute(0, 2, 1)

        return x

    def forward(self, pitch: torch.Tensor, velocity: torch.Tensor, dstart: torch.Tensor, duration: torch.Tensor):
        embedding = self._features_to_embedding(pitch, velocity, dstart, duration)

        x = self.init_conv(embedding)
        x = self.discriminator(x)

        return x
