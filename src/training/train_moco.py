"""MoCo v2 SSL pretraining loop."""
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.datasets.cityscapes_ssl import CityscapesSSL
from src.datasets.ssl_transforms import build_moco_v2_transforms
from src.models.moco import MoCo
from src.utils.config import Config, save_config
from src.paths import CITYSCAPES_ROOT, CHECKPOINTS_ROOT, LOGS_ROOT


def cosine_lr_scheduler(optimizer, base_lr, current_epoch, num_epochs):
    """Cosine LR schedule: lr = 0.5 * base_lr * (1 + cos(pi * t / T))."""
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * current_epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def build_dataloader(cfg: Config) -> DataLoader:
    transform = build_moco_v2_transforms(crop_size=cfg.augmentations.crop_size)
    ds = CityscapesSSL(
        root=str(CITYSCAPES_ROOT),
        split="train",
        mode=cfg.dataset.mode,
        transform=transform,
        max_frame_gap=getattr(cfg.dataset, "max_frame_gap", 15),
    )
    print(f"Dataset mode: {cfg.dataset.mode}, samples: {len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=True,
        persistent_workers=True,
    )
    return loader


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup output dirs
    run_dir = CHECKPOINTS_ROOT / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS_ROOT / cfg.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    writer = SummaryWriter(log_dir=str(log_dir))

    # Data
    train_loader = build_dataloader(cfg)
    print(f"Train batches per epoch: {len(train_loader)}")

    # Model
    model = MoCo(
        encoder_name=cfg.model.encoder,
        proj_dim=cfg.model.proj_dim,
        queue_size=cfg.model.queue_size,
        momentum=cfg.model.momentum,
        temperature=cfg.model.temperature,
        encoder_weights=getattr(cfg.model, "encoder_weights", "imagenet"),
    ).to(device)
    
    params = list(model.backbone.parameters()) + list(model.projection_head.parameters())
    
    optimizer = torch.optim.SGD(
    params,
    lr=cfg.training.learning_rate,
    momentum=cfg.training.sgd_momentum,
    weight_decay=cfg.training.weight_decay,
)

    # Mixed precision
    use_amp = cfg.training.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        lr = cosine_lr_scheduler(
            optimizer,
            base_lr=cfg.training.learning_rate,
            current_epoch=epoch,
            num_epochs=cfg.training.num_epochs,
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        for view_q, view_k in pbar:
            view_q = view_q.to(device, non_blocking=True)
            view_k = view_k.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(view_q, view_k)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.5f}",
            })

            if global_step % cfg.logging.log_every_n_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)

        # Periodic checkpoint
        if (epoch + 1) % cfg.logging.save_every_n_epochs == 0:
            ckpt_path = run_dir / f"epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    # Final checkpoint
    final_path = run_dir / "last.pth"
    torch.save({
        "epoch": cfg.training.num_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": avg_loss
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    writer.close()