import torch
import torch.nn as nn

from models.modules.attention_layers import ConvAttention
from models.modules.conv_layers import PreNorm, Residual, Downsample, ResnetBlock


class Encoder(nn.Module):
    def __init__(self, dims: tuple[int], resnet_block_groups: int = 4, causal: bool = False):
        super().__init__()

        # get list of corresponding ins and outs
        ins_outs = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(ins_outs)

        layers = []

        for idx, (ins, outs) in enumerate(ins_outs):
            is_last = idx >= (num_resolutions - 1)

            layers += [
                ResnetBlock(ins, ins, kernel_size=9, groups=resnet_block_groups, causal=causal),
                ResnetBlock(ins, ins, kernel_size=9, groups=resnet_block_groups, causal=causal),
                Residual(PreNorm(ins, ConvAttention(ins, causal=causal))),
                Downsample(ins, outs) if not is_last else nn.Conv1d(ins, outs, kernel_size=1),
            ]

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)
