"""Training loop for semantic segmentation."""
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.datasets.cityscapes import CityscapesSegmentation, NUM_CLASSES, IGNORE_INDEX
from src.datasets.transforms import SegmentationTransforms
from src.models.segmentation import build_deeplabv3plus
from src.evaluation.metrics import ConfusionMatrix
from src.utils.config import Config, save_config
from src.paths import CITYSCAPES_ROOT, CHECKPOINTS_ROOT, LOGS_ROOT


def poly_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    current_iter: int,
    max_iter: int,
    power: float = 0.9,
) -> None:
    """Poly learning rate schedule: lr = base_lr * (1 - iter/max_iter)^power."""
    lr = base_lr * (1 - current_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_dataloaders(cfg: Config):
    train_transform = SegmentationTransforms(
        split="train",
        crop_size=tuple(cfg.augmentations.crop_size),
        horizontal_flip_prob=cfg.augmentations.horizontal_flip_prob,
        color_jitter=dict(cfg.augmentations.color_jitter),
        ignore_index=IGNORE_INDEX,
    )
    val_transform = SegmentationTransforms(
        split="val",
        ignore_index=IGNORE_INDEX,
    )

    train_ds = CityscapesSegmentation(
        root=str(CITYSCAPES_ROOT), split="train", transform=train_transform
    )
    val_ds = CityscapesSegmentation(
        root=str(CITYSCAPES_ROOT), split="val", transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # val slike su pune rezolucije, batch 1 da stane
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    return train_loader, val_loader


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    cm = ConfusionMatrix(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    pbar = tqdm(val_loader, desc="Val", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        cm.update(preds, masks)

    miou = cm.compute_miou()
    return miou


def train(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directories
    run_dir = CHECKPOINTS_ROOT / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS_ROOT / cfg.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    # Tensorboard
    writer = SummaryWriter(log_dir=str(log_dir))

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    ssl_path = getattr(cfg.model, "ssl_checkpoint_path", None)
    model = build_deeplabv3plus(
        encoder_name=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        num_classes=cfg.model.num_classes,
        ssl_checkpoint_path=ssl_path,
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.training.ignore_index)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )

    # Mixed precision
    use_amp = cfg.training.mixed_precision
    scaler = amp.GradScaler("cuda", enabled=use_amp) # type: ignore[attr-defined]

    # Training loop
    max_iter = cfg.training.num_epochs * len(train_loader)
    global_step = 0
    best_miou = 0.0

    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            poly_lr_scheduler(
                optimizer,
                base_lr=cfg.training.learning_rate,
                current_iter=global_step,
                max_iter=max_iter,
                power=cfg.training.poly_power,
            )

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=use_amp): # type: ignore[attr-defined]
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
            })

            if global_step % cfg.logging.log_every_n_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)

        # Validation
        if (epoch + 1) % cfg.logging.val_every_n_epochs == 0:
            miou = validate(model, val_loader, device)
            print(f"Epoch {epoch+1}: val mIoU = {miou:.4f}")
            writer.add_scalar("val/miou", miou, epoch + 1)

            if miou > best_miou:
                best_miou = miou
                torch.save({
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "miou": miou,
                }, run_dir / "best.pth")
                print(f"  New best mIoU. Saved to {run_dir / 'best.pth'}")

        # Periodic checkpoint
        if (epoch + 1) % cfg.logging.save_every_n_epochs == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, run_dir / f"epoch_{epoch+1}.pth")

    # Final save
    torch.save({
        "epoch": cfg.training.num_epochs,
        "model_state": model.state_dict(),
    }, run_dir / "last.pth")

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    writer.close()