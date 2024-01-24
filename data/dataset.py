import random

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from data.augmentation import pitch_shift, change_speed


class MidiDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        pitch_shift_probability: float = 0.0,
        time_stretch_probability: float = 0.0,
    ):
        super().__init__()

        self.dataset = dataset.with_format("numpy")

        self.pitch_shift_probability = pitch_shift_probability
        self.time_stretch_probability = time_stretch_probability

    def __rich_repr__(self):
        yield "MidiDataset"
        yield "size", len(self)
        yield "pitch_shift_prob", self.pitch_shift_probability
        yield "time_stretch_prob", self.time_stretch_probability

    def __len__(self):
        return len(self.dataset)

    def apply_augmentation(self, record: dict):
        # shift pitch augmentation
        if random.random() < self.pitch_shift_probability:
            shift = 7
            record["pitch"] = pitch_shift(pitch=record["pitch"], shift_threshold=shift)

        # change tempo augmentation
        if random.random() < self.time_stretch_probability:
            record["dstart"], record["duration"] = change_speed(dstart=record["dstart"], duration=record["duration"])

        return record

    def __getitem__(self, index: int) -> dict:
        record = self.dataset[index]

        # sanity check, replace NaN with 0
        if np.any(np.isnan(record["dstart"])):
            record["dstart"] = np.nan_to_num(record["dstart"], copy=False)

        record = self.apply_augmentation(record)

        # clip outliers
        record["dstart"] = np.clip(record["dstart"], 0.0, 5.0)

        data = {
            "filename": record["midi_filename"],
            "source": record["source"],
            "pitch": torch.tensor(record["pitch"], dtype=torch.long) - 21,
            "velocity": (torch.tensor(record["velocity"], dtype=torch.float) / 64) - 1,
            "dstart": torch.tensor(record["dstart"], dtype=torch.float),
            "duration": torch.tensor(record["duration"], dtype=torch.float).clip(0.0, 5.0),
        }

        return data


class MidiDatasetWithStartEnd(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        pitch_shift_probability: float = 0.0,
        time_stretch_probability: float = 0.0,
    ):
        super().__init__()

        self.dataset = dataset.with_format("numpy")

        self.pitch_shift_probability = pitch_shift_probability
        self.time_stretch_probability = time_stretch_probability

    def __rich_repr__(self):
        yield "MidiDataset"
        yield "size", len(self)
        yield "pitch_shift_prob", self.pitch_shift_probability
        yield "time_stretch_prob", self.time_stretch_probability

    def __len__(self):
        return len(self.dataset)

    def apply_augmentation(self, record: dict):
        # shift pitch augmentation
        if random.random() < self.pitch_shift_probability:
            shift = 7
            record["pitch"] = pitch_shift(pitch=record["pitch"], shift_threshold=shift)

        # change tempo augmentation
        if random.random() < self.time_stretch_probability:
            record["dstart"], record["duration"] = change_speed(dstart=record["dstart"], duration=record["duration"])

        return record

    def __getitem__(self, index: int) -> dict:
        record = self.dataset[index]

        # sanity check, replace NaN with 0
        if np.any(np.isnan(record["dstart"])):
            record["dstart"] = np.nan_to_num(record["dstart"], copy=False)

        record = self.apply_augmentation(record)

        # clip outliers
        record["dstart"] = np.clip(record["dstart"], 0.0, 5.0)
        record["duration"] = np.clip(record["duration"], 0.0, 5.0)

        record["start"] = np.cumsum(record["dstart"])[:-1]
        record["start"] = np.insert(record["start"], obj=0, values=0.0)

        record["end"] = record["start"] + record["duration"]
        # record["sequence_duration"] = record["start"][-1] + record["duration"][-1]

        data = {
            "filename": record["midi_filename"],
            "source": record["source"],
            "pitch": torch.tensor(record["pitch"], dtype=torch.long) - 21,
            "velocity": torch.tensor(record["velocity"], dtype=torch.long),
            "dstart": torch.tensor(record["dstart"], dtype=torch.float),
            "duration": torch.tensor(record["duration"], dtype=torch.float),
            "start": torch.tensor(record["start"], dtype=torch.float),
            "end": torch.tensor(record["end"], dtype=torch.float),
        }

        return data
