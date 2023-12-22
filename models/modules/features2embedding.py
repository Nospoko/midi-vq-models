import torch
import torch.nn as nn


class Feature2Embedding(nn.Module):
    def __init__(self, in_features: int, features: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, features),
            nn.SiLU(),
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class Embedding2Feature(nn.Module):
    def __init__(self, features: int, out_features: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)
