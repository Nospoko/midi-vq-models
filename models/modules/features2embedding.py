import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    def __init__(self, in_features: int, features: int, out_features: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, features),
            nn.SiLU(),
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int, heads: int, ffn_expansion: int = 2, dropout_rate: float = 0.3):
        """
        Transformer Block

        Args:
            embedding_size (int): size of embedding dim
            heads (int): number of attention heads
            ffn_expansion (int): scaling factor for hidden dim expansion in feed forward layer
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        super().__init__()

        # expanded dimension for feed forward
        hidden_dim = embedding_size * ffn_expansion

        self.ln1 = nn.LayerNorm(embedding_size)
        self.attention = nn.MultiheadAttention(embedding_size, heads, batch_first=True)

        self.ln2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # skip connection
        skip = x

        # normalization
        x = self.ln1(x)

        # calculating attention
        x, _ = self.attention(x, x, x, mask)

        # residual connection with input x
        x = x + skip
        skip = x

        # dropout
        x = self.dropout(x)

        # normalization
        x = self.ln2(x)

        # passing to feedforward layer
        x = self.feed_forward(x)

        # residual connection
        x = x + skip

        # dropout
        x = self.dropout(x)

        return x
    
class TransformerFeatureProjection(nn.Module):
    def __init__(self, features: int, out_features: int, heads: int = 8):
        super().__init__()

        self.block = nn.Sequential(
            TransformerBlock(features, heads=heads),
            TransformerBlock(features, heads=heads),
            TransformerBlock(features, heads=heads),
        )

        self.out = nn.Linear(features, out_features)

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        x = self.out(x)

        return x