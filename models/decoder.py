import torch
import torch.nn as nn

from models.modules.attention_layers import ConvAttention
from models.modules.conv_layers import PreNorm, Residual, Upsample, ResnetBlock


class Decoder(nn.Module):
    def __init__(self, dims: tuple[int], resnet_block_groups: int = 4, causal: bool = False):
        super().__init__()

        # get list of corresponding ins and outs
        ins_outs = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(ins_outs)

        layers = []

        for idx, (ins, outs) in enumerate(reversed(ins_outs)):
            is_last = idx >= (num_resolutions - 1)

            layers += [
                ResnetBlock(outs, outs, groups=resnet_block_groups, causal=causal),
                ResnetBlock(outs, outs, groups=resnet_block_groups, causal=causal),
                Residual(PreNorm(outs, ConvAttention(outs, causal=causal))),
                Upsample(outs, ins) if not is_last else nn.Conv1d(outs, ins, kernel_size=1),
            ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.decoder(x)
