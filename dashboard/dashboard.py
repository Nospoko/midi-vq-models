import json
import random
from glob import glob

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from streamlit_pianoroll import from_fortepyan

from models.vqvae import MidiVQVAE
from data.dataset import MidiDataset

device = "cpu"


def display_pianoroll(processing_result: dict):
    st.json(processing_result["source"], expanded=False)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Original")
        from_fortepyan(processing_result["original"])
    with col2:
        st.write("### Reconstructed")
        from_fortepyan(processing_result["generated"])


def denormalize_velocity(velocity: np.ndarray):
    return ((velocity + 1) * 64).astype("int")


def denormalize_time_features(time_feature: np.ndarray, mean: float, std: float):
    time_feature = std * time_feature + mean
    return 2**time_feature - 1e-8


def to_midi_piece(
    pitch: np.ndarray,
    dstart: np.ndarray,
    duration: np.ndarray,
    velocity: np.ndarray,
) -> ff.MidiPiece:
    record = {
        "pitch": pitch,
        "velocity": velocity,
        "dstart": dstart.astype("float"),
        "duration": duration.astype("float"),
    }

    df = pd.DataFrame(record)
    df["start"] = df.dstart.cumsum().shift(1).fillna(0)
    df["end"] = df.start + df.duration

    return ff.MidiPiece(df)


# @st.cache_data
def get_model(checkpoint_path: str) -> MidiVQVAE:
    checkpoint = torch.load(checkpoint_path)

    cfg = checkpoint["config"]
    st.json(OmegaConf.to_container(cfg), expanded=False)

    model = MidiVQVAE(
        dim=cfg.model.dim,
        dim_mults=cfg.model.dim_mults,
        fsq_levels=cfg.model.fsq_levels,
        resnet_block_groups=cfg.model.num_resnet_groups,
        causal=cfg.model.causal,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, cfg


def model_selection() -> MidiVQVAE:
    checkpoints = glob("checkpoints/*")
    checkpoint_path = st.selectbox("Select checkpoint", options=checkpoints)

    model, cfg = get_model(checkpoint_path)
    return model, cfg


@st.cache_data
def get_dataset(dataset_name: str) -> Dataset:
    # dataset_name = "JasiekKaczmarczyk/maestro-v1-sustain-masked"
    dataset = load_dataset(dataset_name, split="train")

    # available_splits = list(dataset.keys())
    # split = st.selectbox("Choose split", options=available_splits)

    return dataset


def slice_records(records: list[ff.MidiPiece], window_size: int):
    sliced_records = []

    for record in records:
        n_samples = 1 + (record.size - window_size) // window_size
        for it in range(n_samples):
            start = it * window_size
            finish = start + window_size
            part = record[start:finish]

            sliced_records.append(part)

    return sliced_records


def generate_midi_sequence(records: list[ff.MidiPiece]):
    sequences = []

    for record in records:
        midi_filename = "tmp"
        record.df["next_start"] = record.df.start.shift(-1)
        record.df["dstart"] = record.df.next_start - record.df.start
        record.df["dstart"] = record.df["dstart"].fillna(0)

        sequence = {
            "midi_filename": midi_filename,
            "source": json.dumps(record.source),
            "pitch": record.df.pitch.astype("int16").values,
            "start": record.df.start.astype("float32").values,
            "dstart": record.df.dstart.astype("float32").values,
            "duration": record.df.duration.astype("float32").values,
            "velocity": record.df.velocity.astype("int16").values,
        }

        sequences.append(sequence)

    return Dataset.from_list(sequences)


def main():
    # Load model
    model, _ = model_selection()

    # Prep data
    dataset_name = st.selectbox(
        label="Select dataset",
        options=["roszcz/maestro-sustain-v2", "roszcz/giant-midi-sustain-v2"],
    )
    dataset = get_dataset(dataset_name)
    source = [json.loads(source) for source in dataset["source"]]
    source_df = pd.DataFrame(source)

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
    )
    st.write(selected_title)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = dataset.select(part_df.index.values)
    records = []

    for ds in part_dataset:
        record = ff.MidiPiece.from_huggingface(ds)
        records.append(record)

    display_mode = st.toggle("Specific Window/Random Windows")

    if display_mode:
        num_displayed_records = st.number_input(label="Number of displayed records", min_value=1, value=10)
        window_size = st.number_input(label="Number of notes in each window", min_value=16, value=128)

        sliced_records = slice_records(records, window_size=window_size)
        sampled_records = random.sample(sliced_records, k=num_displayed_records)
    else:
        start_note = st.number_input("Start note", value=0)
        window_size = st.number_input(label="Number of notes in window", min_value=16, value=128)

        sampled_records = [records[0][start_note : start_note + window_size]]

    midi_sequences = generate_midi_sequence(sampled_records)

    midi_dataset = MidiDataset(midi_sequences)

    generated_pieces = generate_pieces(
        midi_dataset=midi_dataset,
        model=model,
    )

    for processing_result in generated_pieces:
        display_pianoroll(processing_result)


@torch.no_grad()
def generate_pieces(
    midi_dataset: MidiDataset,
    model: MidiVQVAE,
):
    dataloader = DataLoader(midi_dataset, batch_size=256, shuffle=True)

    generated_pieces = []
    for batch in dataloader:
        pitches = batch["pitch"].to(device)
        velocities = batch["velocity"].to(device)
        dstarts = batch["dstart"].to(device)
        durations = batch["duration"].to(device)

        gen_pitches, gen_velocities, gen_dstarts, gen_durations = model(
            pitch=pitches,
            velocity=velocities,
            dstart=dstarts,
            duration=durations,
        )

        # replace tokens that were masked with generated values
        gen_pitches = torch.argmax(gen_pitches, dim=-1)

        generated_pieces += decode_batch(
            batch=batch,
            gen_pitches=gen_pitches,
            gen_velocities=gen_velocities,
            gen_dstarts=gen_dstarts,
            gen_durations=gen_durations,
        )

    return generated_pieces


def decode_batch(
    batch: torch.Tensor,
    gen_pitches: torch.Tensor,
    gen_velocities: torch.Tensor,
    gen_dstarts: torch.Tensor,
    gen_durations: torch.Tensor,
):
    sources = batch["source"]
    pitches = batch["pitch"].to(device)
    velocities = batch["velocity"].to(device)
    dstarts = batch["dstart"].to(device)
    durations = batch["duration"].to(device)

    batch_size = pitches.shape[0]

    generated_pieces = []
    for it in range(batch_size):
        pitch = pitches[it].cpu().numpy() + 21
        velocity = velocities[it].cpu().numpy()
        dstart = dstarts[it].cpu().numpy()
        duration = durations[it].cpu().numpy()

        gen_pitch = gen_pitches[it].cpu().numpy() + 21
        gen_velocity = gen_velocities[it].cpu().numpy()
        gen_dstart = gen_dstarts[it].cpu().numpy()
        gen_duration = gen_durations[it].cpu().numpy()

        velocity = denormalize_velocity(velocity)
        gen_velocity = denormalize_velocity(gen_velocity)

        original_piece = to_midi_piece(pitch, dstart, duration, velocity)
        model_piece = to_midi_piece(gen_pitch, gen_dstart, gen_duration, gen_velocity)

        processing_result = {
            "original": original_piece,
            "generated": model_piece,
            "source": json.loads(sources[it]),
        }
        generated_pieces.append(processing_result)

    return generated_pieces


if __name__ == "__main__":
    main()
