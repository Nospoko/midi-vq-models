import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from models.modules.cape import CAPE1d
from models.modules.quantization import FSQ
from models.modules.features2embedding import TransformerFeatureProjection
from models.modules.positional_embedding import SinusoidalPositionEmbeddingsV2
from models.modules.attention_layers import DownsampleAttention, UpsampleAttention

class MidiVQVAETransformer(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        fsq_levels: list[int] = [8, 8, 8, 6, 5],
        mlp_ratio: float = 4.0,
    ):
        super().__init__()


        # list of dimensions
        quantization_dim = len(fsq_levels)
        
        # embedding heads
        quarter_dim = dim // 4
        self.pitch_embedding = nn.Embedding(num_embeddings=88, embedding_dim=quarter_dim)
        self.velocity_embedding = nn.Embedding(num_embeddings=128, embedding_dim=quarter_dim)
        self.time_embedding = SinusoidalPositionEmbeddingsV2(quarter_dim)

        # self.cape = CAPE1d(quarter_dim, max_global_shift=0.1, max_local_shift=0.5, max_global_scaling=1.4, batch_first=True)

        # encoder
        self.encoder = nn.ModuleList(
            [
                nn.ModuleList([
                    Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                    ),
                    Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                    ),
                    nn.LayerNorm(dim),
                    DownsampleAttention(dim, downsample_factor=2, heads=num_heads),
                ])
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

        # fsq
        self.q_proj_in = nn.Linear(dim, quantization_dim)
        self.quantizer = FSQ(levels=fsq_levels)
        self.q_proj_out = nn.Linear(quantization_dim, dim)

        # decoder
        self.decoder = nn.ModuleList(
            [
                nn.ModuleList([
                    Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                    ),
                    Block(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                    ),
                    nn.LayerNorm(dim),
                    UpsampleAttention(dim, upsample_factor=2, heads=num_heads),
                ])
                for _ in range(depth)
            ]
        )

        self.out_norm = nn.LayerNorm(dim)

        self.pitch_out = TransformerFeatureProjection(dim, 88)
        self.velocity_out = TransformerFeatureProjection(dim, 128)
        self.dstart_out = TransformerFeatureProjection(dim, 1)
        self.duration_out = TransformerFeatureProjection(dim, 1)

    def _features_to_embedding(self, pitch: torch.Tensor, velocity: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
        pitch_emb = self.pitch_embedding(pitch)
        velocity_emb = self.velocity_embedding(velocity)
        start_emb = self.time_embedding(start)
        end_emb = self.time_embedding(end)

        # start_emb = self.cape(start_emb)
        # end_emb = self.cape(end_emb)

        # shape: [batch_size, seq_len, embedding_dim]
        x = torch.cat([pitch_emb, velocity_emb, start_emb, end_emb], dim=-1)

        return x

    def _embedding_to_features(self, embedding: torch.Tensor):
        pitch = self.pitch_out(embedding)
        velocity = self.velocity_out(embedding)
        dstart = self.dstart_out(embedding).squeeze(-1)
        duration = self.duration_out(embedding).squeeze(-1)

        return pitch, velocity, dstart, duration

    def quantize(self, pitch: torch.Tensor, velocity: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
        x = self._features_to_embedding(pitch, velocity, start, end)
        for block1, block2, norm, down in self.encoder:
            x = block1(x)
            x = block2(x)
            x = norm(x)
            x = down(x)

        x = self.out_norm(x)
        x = self.q_proj_in(x)
        _, indices = self.quantizer(x)

        return indices

    def dequantize(self, indices: torch.Tensor):
        codes = self.quantizer.indices_to_codes(indices)
        x = self.q_proj_out(codes)

        for block1, block2, norm, up in self.decoder:
            x = block1(x)
            x = block2(x)
            x = norm(x)
            x = up(x)

        x = self.out_norm(x)

        pitch_hat, velocity_hat, dstart_hat, duration_hat = self._embedding_to_features(x)

        return pitch_hat, velocity_hat, dstart_hat, duration_hat

    def forward(self, pitch: torch.Tensor, velocity: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
        x = self._features_to_embedding(pitch, velocity, start, end)

        for block1, block2, norm, down in self.encoder:
            x = block1(x)
            x = block2(x)
            # x = norm(x)
            # x = down(x)

        x = self.out_norm(x)

        x = self.q_proj_in(x)
        codes, _ = self.quantizer(x)
        x = self.q_proj_out(codes)

        for block1, block2, norm, up in self.decoder:
            x = block1(x)
            x = block2(x)
            # x = norm(x)
            # x = up(x)
        
        x = self.out_norm(x)

        pitch_hat, velocity_hat, dstart_hat, duration_hat = self._embedding_to_features(x)

        return pitch_hat, velocity_hat, dstart_hat, duration_hat


# if __name__ == "__main__":
#     p = torch.randint(0, 88, (1, 128)).to("cuda")
#     v = torch.randint(0, 128, (1, 128)).to("cuda")
#     s = torch.rand((1, 128)).to("cuda")
#     e = torch.rand((1, 128)).to("cuda")

#     m = MidiVQVAETransformer().to("cuda")

#     out = m(p, v, s, e)

#     print(out[0].shape)