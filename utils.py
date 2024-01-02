from dataclasses import dataclass
import torch

@dataclass
class MidiFeatures:
    filename: str
    source: str
    pitch: torch.Tensor
    velocity: torch.Tensor
    dstart: torch.Tensor
    duration: torch.Tensor

    def to_(self, device: torch.device):
        self.pitch = self.pitch.to(device)
        self.velocity = self.velocity.to(device)
        self.dstart = self.dstart.to(device)
        self.duration = self.duration.to(device)