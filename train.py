import os
from collections import Counter

import hydra
import torch
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
import torchmetrics.functional as M
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader
from datasets import load_dataset, concatenate_datasets

import wandb
from utils import MidiFeatures
from models.vqvae import MidiVQVAE
from data.dataset import MidiDataset


def print_metrics(metrics: dict):
    text = "|".join([f"{k}: {v:.2f}" for k, v in metrics.items()])

    print(text)


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(
    dataset_names: list[str],
    batch_size: int,
    num_workers: int,
    pitch_shift_probability: float,
    time_stretch_probability: float,
    *,
    overfit_single_batch: bool = False,
):
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

    train_ds = []
    val_ds = []
    test_ds = []

    for dataset_name in dataset_names:
        tr_ds = load_dataset(dataset_name, split="train", use_auth_token=hf_token)
        v_ds = load_dataset(dataset_name, split="validation", use_auth_token=hf_token)
        t_ds = load_dataset(dataset_name, split="test", use_auth_token=hf_token)

        train_ds.append(tr_ds)
        val_ds.append(v_ds)
        test_ds.append(t_ds)

    train_ds = concatenate_datasets(train_ds)
    val_ds = concatenate_datasets(val_ds)
    test_ds = concatenate_datasets(test_ds)

    train_ds = MidiDataset(
        train_ds,
        pitch_shift_probability=pitch_shift_probability,
        time_stretch_probability=time_stretch_probability,
    )
    val_ds = MidiDataset(
        val_ds,
        pitch_shift_probability=0.0,
        time_stretch_probability=0.0,
    )
    test_ds = MidiDataset(
        test_ds,
        pitch_shift_probability=0.0,
        time_stretch_probability=0.0,
    )

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def forward_step(
    model: MidiVQVAE,
    batch: MidiFeatures,
    loss_lambdas: list,
):
    pred_pitch, pred_velocity, pred_dstart, pred_duration = model(
        pitch=batch.pitch,
        velocity=batch.velocity,
        dstart=batch.dstart,
        duration=batch.duration,
    )

    pred_pitch = pred_pitch.permute(0, 2, 1)

    # calculate losses
    pitch_loss = F.cross_entropy(pred_pitch, batch.pitch)
    velocity_loss = F.mse_loss(pred_velocity, batch.velocity)
    dstart_loss = F.mse_loss(pred_dstart, batch.dstart)
    duration_loss = F.mse_loss(pred_duration, batch.duration)

    # shape: [num_losses, ]
    losses = torch.stack([pitch_loss, velocity_loss, dstart_loss, duration_loss])

    loss = loss_lambdas @ losses

    pitch_acc = (torch.argmax(pred_pitch, dim=1) == batch.pitch).float().mean()
    velocity_r2 = M.r2_score(pred_velocity, batch.velocity)
    dstart_r2 = M.r2_score(pred_dstart, batch.dstart)
    duration_r2 = M.r2_score(pred_duration, batch.duration)

    metrics = {
        "loss": loss.item(),
        "pitch_loss": pitch_loss.item(),
        "velocity_loss": velocity_loss.item(),
        "dstart_loss": dstart_loss.item(),
        "duration_loss": duration_loss.item(),
        "pitch_acc": pitch_acc.item(),
        "velocity_r2": velocity_r2.item(),
        "dstart_r2": dstart_r2.item(),
        "duration_r2": duration_r2.item(),
        "goal_weighted_sum": ((pitch_acc + velocity_r2 + dstart_r2 + duration_r2) / 4).item(),
    }

    return loss, metrics


@torch.no_grad()
def validation_epoch(
    model: MidiVQVAE,
    dataloader: DataLoader,
    loss_lambdas: torch.Tensor,
    device: torch.device,
    cfg: OmegaConf,
) -> dict:
    # val epoch
    metrics_epoch = Counter(
        {
            "loss": 0.0,
            "pitch_loss": 0.0,
            "velocity_loss": 0.0,
            "dstart_loss": 0.0,
            "duration_loss": 0.0,
            "pitch_acc": 0.0,
            "velocity_r2": 0.0,
            "dstart_r2": 0.0,
            "duration_r2": 0.0,
            "goal_weighted_sum": 0.0,
        }
    )

    for batch_idx, batch in enumerate(dataloader):
        batch = MidiFeatures(
            filename=batch["filename"],
            source=batch["source"],
            pitch=batch["pitch"],
            velocity=batch["velocity"],
            dstart=batch["dstart"],
            duration=batch["duration"],
        )

        batch.to_(device)
        # metrics returns loss and additional metrics if specified in step function
        _, metrics = forward_step(model, batch, loss_lambdas)

        if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
            print_metrics(metrics)
        metrics_epoch += Counter(metrics)

    return metrics_epoch


def save_checkpoint(model: MidiVQVAE, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


def upload_to_huggingface(ckpt_save_path: str, cfg: OmegaConf):
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # upload model to hugging face
    upload_file(ckpt_save_path, path_in_repo=f"{cfg.logger.run_name}.ckpt", repo_id=cfg.paths.hf_repo_id, token=token)


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    wandb.login()

    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    # dataset
    train_dataloader, val_dataloader, _ = preprocess_dataset(
        dataset_names=cfg.train.dataset_names,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pitch_shift_probability=cfg.train.pitch_shift_probability,
        time_stretch_probability=cfg.train.time_stretch_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # validate on quantized maestro
    # This dataset is already in cfg.train.dataset_names - is this unneccessary duplication?
    _, maestro_test, _ = preprocess_dataset(
        dataset_names=["JasiekKaczmarczyk/maestro-v1-sustain-masked"],
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pitch_shift_probability=cfg.train.pitch_shift_probability,
        time_stretch_probability=cfg.train.time_stretch_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    device = torch.device(cfg.train.device)

    # model
    model = MidiVQVAE(
        dim=cfg.model.dim,
        dim_mults=cfg.model.dim_mults,
        fsq_levels=cfg.model.fsq_levels,
        resnet_block_groups=cfg.model.num_resnet_groups,
        causal=cfg.model.causal,
    ).to(device)

    # checkpoint save path
    num_params_millions = sum([p.numel() for p in model.parameters()]) / 1_000_000
    run_name = f"{cfg.logger.run_name}-{num_params_millions:.2f}M.ckpt"
    save_path = f"{cfg.paths.save_ckpt_dir}/{run_name}"

    # logger
    wandb.init(
        project="midi-vqvae",
        name=run_name,
        dir=cfg.paths.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # setting up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # get loss lambdas as tensor
    loss_lambdas = torch.tensor(list(cfg.train.loss_lambdas.values()), dtype=torch.float, device=device)

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # step counts for logging to wandb
    step_count = 0
    notes_processed = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        model.train()
        training_metrics_epoch = Counter(
            {
                "loss": 0.0,
                "pitch_loss": 0.0,
                "velocity_loss": 0.0,
                "dstart_loss": 0.0,
                "duration_loss": 0.0,
                "pitch_acc": 0.0,
                "velocity_r2": 0.0,
                "dstart_r2": 0.0,
                "duration_r2": 0.0,
                "goal_weighted_sum": 0.0,
            }
        )

        for batch_idx, batch in enumerate(train_dataloader):
            batch = MidiFeatures(
                filename=batch["filename"],
                source=batch["source"],
                pitch=batch["pitch"],
                velocity=batch["velocity"],
                dstart=batch["dstart"],
                duration=batch["duration"],
            )

            batch.to_(device)

            # metrics returns loss and additional metrics if specified in step function
            loss, metrics = forward_step(model, batch, loss_lambdas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1
            notes_processed += torch.numel(batch.pitch)

            training_metrics_epoch += Counter(metrics)

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                metrics = {"epoch": epoch, "step": batch_idx, "notes_processed": notes_processed} | metrics
                print_metrics(metrics)

                metrics |= {"learning_rate": cfg.train.lr}
                metrics = {"train/" + key: value for key, value in metrics.items()}

                # log metrics
                wandb.log(metrics, step=step_count)

                # save model and optimizer states
                save_checkpoint(model, optimizer, cfg, save_path=save_path)

        training_metrics_epoch = {
            "train_epoch/" + key: value / len(train_dataloader) for key, value in training_metrics_epoch.items()
        }

        model.eval()

        # val epoch
        print("Validation")
        val_metrics_epoch = validation_epoch(
            model=model,
            dataloader=val_dataloader,
            loss_lambdas=loss_lambdas,
            device=device,
            cfg=cfg,
        )
        val_metrics_epoch = {"val_epoch/" + key: value / len(val_dataloader) for key, value in val_metrics_epoch.items()}

        # maestro test epoch
        print("Test (maestro)")
        test_metrics_epoch = validation_epoch(
            model=model,
            dataloader=maestro_test,
            loss_lambdas=loss_lambdas,
            device=device,
            cfg=cfg,
        )
        test_metrics_epoch = {"maestro_epoch/" + key: value / len(maestro_test) for key, value in test_metrics_epoch.items()}

        metrics_epoch = training_metrics_epoch | val_metrics_epoch | test_metrics_epoch
        wandb.log(metrics_epoch, step=step_count)

    # save model at the end of training
    save_checkpoint(model, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    train()
