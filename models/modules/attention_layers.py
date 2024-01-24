import torch
import einops
import torch.nn as nn
import torch.nn.functional as F


class ConvAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, causal: bool = False):
        super().__init__()

        self.heads = heads
        self.head_dim = channels // heads
        self.embedding_size = channels
        self.causal = causal

        assert self.head_dim * heads == channels, "Channels needs to be dividable by heads"

        self.values_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.keys_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.queries_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # applying linear projections
        q = self.queries_proj(x)
        k = self.keys_proj(x)
        v = self.values_proj(x)

        # rearranging q, k, v from [batch_size, channels, seq_len] -> [batch_size, seq_len, heads, head_dim]
        q = einops.rearrange(q, "n (h d) l -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n (h d) l -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n (h d) l -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        if self.causal:
            L = qk.shape[-1]
            mask = torch.tril(torch.ones((L, L), device=qk.device))
            # setting masked values to -inf so that softmax will give then probability zero
            qk = qk.masked_fill(mask == 0, float("-inf"))

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=-1)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n (h d) l")

        return self.out(out)


class DownsampleAttention(nn.Module):
    def __init__(self, dim: int, downsample_factor: int = 2, heads: int = 4, causal: bool = False):
        super().__init__()

        self.heads = heads
        self.downsample_factor = downsample_factor
        self.head_dim = dim // heads
        self.embedding_size = dim
        self.causal = causal

        assert self.head_dim * heads == dim, "Channels needs to be dividable by heads"

        self.values_proj = nn.Linear(dim * downsample_factor, dim)
        self.keys_proj = nn.Linear(dim * downsample_factor, dim)
        self.queries_proj = nn.Linear(dim * downsample_factor, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "n (l p) d -> n l (d p)", p=self.downsample_factor)

        # applying linear projections
        q = self.queries_proj(x)
        k = self.keys_proj(x)
        v = self.values_proj(x)

        # rearranging q, k, v from [batch_size, channels, seq_len] -> [batch_size, seq_len, heads, head_dim]
        q = einops.rearrange(q, "n l (h d) -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n l (h d) -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n l (h d) -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        if self.causal:
            L = qk.shape[-1]
            mask = torch.tril(torch.ones((L, L), device=qk.device))
            # setting masked values to -inf so that softmax will give then probability zero
            qk = qk.masked_fill(mask == 0, float("-inf"))

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=-1)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n l (h d)")

        return self.out(out)
    

class UpsampleAttention(nn.Module):
    def __init__(self, dim: int, upsample_factor: int = 2, heads: int = 4, causal: bool = False):
        super().__init__()

        self.heads = heads
        self.upsample_factor = upsample_factor
        self.head_dim = dim // heads
        self.embedding_size = dim
        self.causal = causal

        assert self.head_dim * heads == dim, "Channels needs to be dividable by heads"

        self.values_proj = nn.Linear(dim, dim)
        self.keys_proj = nn.Linear(dim, dim)
        self.queries_proj = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upsample q using bilinear interpolation
        q = einops.rearrange(x, "b l d -> b d l")
        q = F.interpolate(q, scale_factor=self.upsample_factor, mode="linear")
        q = einops.rearrange(q, "b d l -> b l d")

        # applying linear projections
        q = self.queries_proj(q)
        k = self.keys_proj(x)
        v = self.values_proj(x)

        # rearranging q, k, v from [batch_size, channels, seq_len] -> [batch_size, seq_len, heads, head_dim]
        q = einops.rearrange(q, "n l (h d) -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n l (h d) -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n l (h d) -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        if self.causal:
            Q = qk.shape[-2]
            K = qk.shape[-1]
            mask = torch.tril(torch.ones((Q, K), device=qk.device))
            # setting masked values to -inf so that softmax will give then probability zero
            qk = qk.masked_fill(mask == 0, float("-inf"))

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size**-0.5), dim=-1)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n l (h d)")

        return self.out(out)