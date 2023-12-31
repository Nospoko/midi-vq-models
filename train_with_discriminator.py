import os
from collections import Counter

import hydra
import torch
import wandb
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
import torchmetrics.functional as M
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader
from datasets import load_dataset, concatenate_datasets

from models.vqvae import MidiVQVAE
from models.discriminator import Discriminator
from data.dataset import MidiDataset
from utils import MidiFeatures

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(
    dataset_name: list[str],
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

    for ds_name in dataset_name:
        tr_ds = load_dataset(ds_name, split="train", use_auth_token=hf_token)
        v_ds = load_dataset(ds_name, split="validation", use_auth_token=hf_token)
        t_ds = load_dataset(ds_name, split="test", use_auth_token=hf_token)

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

    # pred_pitch = pred_pitch.permute(0, 2, 1)

    # calculate losses
    pitch_loss = F.cross_entropy(pred_pitch.permute(0, 2, 1), batch.pitch)
    velocity_loss = F.mse_loss(pred_velocity, batch.velocity)
    dstart_loss = F.mse_loss(pred_dstart, batch.dstart)
    duration_loss = F.mse_loss(pred_duration, batch.duration)

    # shape: [num_losses, ]
    losses = torch.stack([pitch_loss, velocity_loss, dstart_loss, duration_loss])

    loss = loss_lambdas @ losses

    pitch_acc = (torch.argmax(pred_pitch, dim=-1) == batch.pitch).float().mean()
    velocity_r2 = M.r2_score(pred_velocity, batch.velocity)
    dstart_r2 = M.r2_score(pred_dstart, batch.dstart)
    duration_r2 = M.r2_score(pred_duration, batch.duration)

    batch_pred = MidiFeatures(
        filename=batch.filename,
        source=batch.source,
        pitch=F.gumbel_softmax(pred_pitch, hard=True, dim=-1),
        velocity=pred_velocity,
        dstart=pred_dstart,
        duration=pred_duration,
    )

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
    }

    return loss, batch_pred, metrics

def discriminator_step(
    discriminator: Discriminator,
    batch: MidiFeatures,
    batch_pred: MidiFeatures,
):
    disc_real = discriminator(F.one_hot(batch.pitch, num_classes=88).to(torch.float32), batch.velocity, batch.dstart, batch.duration)
    disc_fake = discriminator(batch_pred.pitch, batch_pred.velocity, batch_pred.dstart, batch_pred.duration)

    loss_discriminator = 0.5 * (torch.mean(F.relu(1. - disc_real)) + torch.mean(F.relu(1. + disc_fake)))
    loss_generator = -torch.mean(disc_fake)

    return loss_discriminator, loss_generator

def calculate_alpha(model: MidiVQVAE, loss_reconstruction: torch.Tensor, loss_gan: torch.Tensor):
    last_layer_weight = model.decoder.decoder[-1].weight

    loss_reconstruction_grad = torch.autograd.grad(loss_reconstruction, last_layer_weight, retain_graph=True)[0]
    loss_gan_grad = torch.autograd.grad(loss_gan, last_layer_weight, retain_graph=True)[0]

    alpha = torch.norm(loss_reconstruction_grad) / (torch.norm(loss_gan_grad) + 1e-4)
    alpha = alpha.clamp(0, 1e4)

    return 0.8 * alpha

@torch.no_grad()
def validation_epoch(
    model: MidiVQVAE,
    dataloader: DataLoader,
    loss_lambdas: torch.Tensor,
    device: torch.device,
) -> dict:
    # val epoch
    val_loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
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
        }
    )

    for batch_idx, batch in val_loop:
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
        _, _, metrics = forward_step(model, batch, loss_lambdas)

        val_loop.set_postfix(metrics)
        metrics_epoch += Counter(metrics)

    return metrics_epoch


def save_checkpoint(model: MidiVQVAE, discriminator: Discriminator, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "disciriminator": discriminator.state_dict(),
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
        dataset_name=cfg.train.dataset_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pitch_shift_probability=cfg.train.pitch_shift_probability,
        time_stretch_probability=cfg.train.time_stretch_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # validate on quantized maestro
    _, maestro_test, _ = preprocess_dataset(
        dataset_name=["JasiekKaczmarczyk/maestro-v1-sustain-masked"],
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pitch_shift_probability=cfg.train.pitch_shift_probability,
        time_stretch_probability=cfg.train.time_stretch_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # logger
    wandb.init(
        project="midi-vqvae",
        name=cfg.logger.run_name,
        dir=cfg.paths.log_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
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

    discriminator = Discriminator(
        dim=cfg.model.dim,
        dim_mults=cfg.model.dim_mults,
        resnet_block_groups=cfg.model.num_resnet_groups,
        causal=cfg.model.causal,
    ).to(device)

    # setting up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    optimizer_discriminator = optim.AdamW(discriminator.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # get loss lambdas as tensor
    loss_lambdas = torch.tensor(list(cfg.train.loss_lambdas.values()), dtype=torch.float, device=device)

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # checkpoint save path
    num_params_millions = sum([p.numel() for p in model.parameters()]) / 1_000_000
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}-params-{num_params_millions:.2f}M.ckpt"

    # step counts for logging to wandb
    step_count = 0
    disc_factor = 0.0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        model.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
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
            }
        )

        for batch_idx, batch in train_loop:
            batch = MidiFeatures(
                filename=batch["filename"],
                source=batch["source"],
                pitch=batch["pitch"],
                velocity=batch["velocity"],
                dstart=batch["dstart"],
                duration=batch["duration"],
            )

            batch.to_(device)

            if step_count > cfg.train.discriminator_warmup:
                disc_factor = 1.0

            # metrics returns loss and additional metrics if specified in step function
            loss_reconstruction, batch_pred, metrics = forward_step(model, batch, loss_lambdas)

            loss_discriminator, loss_generator = discriminator_step(discriminator, batch, batch_pred)
            alpha = calculate_alpha(model, loss_reconstruction=loss_reconstruction, loss_gan=loss_generator)

            loss = loss_reconstruction + alpha * loss_generator
            loss_discriminator = disc_factor * loss_discriminator

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            optimizer_discriminator.zero_grad()
            loss_discriminator.backward()

            optimizer.step()
            optimizer_discriminator.step()

            train_loop.set_postfix(metrics)
            step_count += 1

            training_metrics_epoch += Counter(metrics)

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                metrics = {"train/" + key: value for key, value in metrics.items()}

                # log metrics
                wandb.log(metrics, step=step_count)

                # save model and optimizer states
                save_checkpoint(model, discriminator, optimizer, cfg, save_path=save_path)

        training_metrics_epoch = {
            "train_epoch/" + key: value / len(train_dataloader) for key, value in training_metrics_epoch.items()
        }

        model.eval()

        # val epoch
        val_metrics_epoch = validation_epoch(
            model,
            val_dataloader,
            loss_lambdas,
            device,
        )
        val_metrics_epoch = {"val_epoch/" + key: value / len(val_dataloader) for key, value in val_metrics_epoch.items()}

        # maestro test epoch
        test_metrics_epoch = validation_epoch(
            model,
            maestro_test,
            loss_lambdas,
            device,
        )
        test_metrics_epoch = {"maestro_epoch/" + key: value / len(maestro_test) for key, value in test_metrics_epoch.items()}

        metrics_epoch = training_metrics_epoch | val_metrics_epoch | test_metrics_epoch
        wandb.log(metrics_epoch, step=step_count)

    # save model at the end of training
    save_checkpoint(model, discriminator, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    train()
