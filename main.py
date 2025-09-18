import os
import time
import logging
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm

from torch import amp  # PyTorch 2.1+: autocast & GradScaler live under torch.amp -> amp.autocast / amp.GradScaler
from torch.utils.tensorboard import SummaryWriter

from config import data_config
from utils.helpers import get_model, get_dataloader
from utils.datasets import MPIIGaze  # used for manual split when dataset == 'mpiigaze'

from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation training.")
    parser.add_argument("--data", type=str, default="data/MPIIFaceGaze", help="Directory path for gaze images.")
    parser.add_argument("--dataset", type=str, default="mpiigaze",
                        help="Dataset name, available `gaze360`, `mpiigaze`.")
    parser.add_argument("--output", type=str, default="output", help="Path of output models.")
    parser.add_argument("--checkpoint", type=str, default="output/checkpoint.ckpt", help="Path to checkpoint for resuming training.")
    parser.add_argument("--num-epochs", type=int, default=200, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=56, help="Batch size.")
    parser.add_argument("--arch", type=str, default="mobileone_s0",
                        help="Network architecture, currently: resnet18/34/50, mobilenetv2, mobileone_s0-s4.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regression loss coefficient.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for data loading.")

    # Train/val split controls (used for MPIIFaceGaze)
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="Validation split ratio if dataset has no predefined val split (MPIIFaceGaze).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")

    # AMP toggle
    parser.add_argument("--amp", dest="amp", action="store_true",
                        help="Enable Automatic Mixed Precision (CUDA only).")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable Automatic Mixed Precision.")
    parser.set_defaults(amp=True)

    # TensorBoard
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory (defaults to --output).")

    args = parser.parse_args()

    # Attach common bin config from data_config
    ds_key = args.dataset.lower()
    if ds_key not in data_config:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Expected one of: {list(data_config.keys())}")
    args.bins = data_config[ds_key]["bins"]
    args.binwidth = data_config[ds_key]["binwidth"]
    args.angle = data_config[ds_key]["angle"]
    return args

# ---------------------------------------------------------------------
# Model / Optimizer init
# ---------------------------------------------------------------------
def initialize_model(params, device):
    model = get_model(params.arch, bins=params.bins, pretrained=True).to(device)

    print("MODEL HEADS:", model.fc_yaw.weight.shape, model.fc_pitch.weight.shape)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    # AMP scaler (only useful on CUDA)
    scaler = amp.GradScaler('cuda', enabled=(device.type == "cuda" and params.amp))
    start_epoch = 0
    if params.checkpoint and os.path.isfile(params.checkpoint):
        ckpt = torch.load(params.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) - 1
        logging.info(f"Resumed from checkpoint: {params.checkpoint} (epoch {start_epoch+1})")
    return model, optimizer, scaler, start_epoch

# ---------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------
def train_one_epoch(
    params,
    model: nn.Module,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    data_loader: DataLoader,
    idx_tensor: torch.Tensor,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    sum_loss_pitch, sum_loss_yaw = 0.0, 0.0

    # tqdm progress bar over training batches
    pbar = tqdm(
        data_loader,
        total=len(data_loader),
        desc=f"Epoch {epoch + 1} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels_gaze, regression_labels_gaze, _ in pbar:
        images = images.to(device)
        label_pitch = labels_gaze[:, 0].to(device)
        label_yaw = labels_gaze[:, 1].to(device)
        label_pitch_regression = regression_labels_gaze[:, 0].to(device)
        label_yaw_regression = regression_labels_gaze[:, 1].to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast('cuda', enabled=(device.type == "cuda" and params.amp)):
            pitch, yaw = model(images)

            # classification loss
            loss_pitch_cls = cls_criterion(pitch, label_pitch)
            loss_yaw_cls = cls_criterion(yaw, label_yaw)

            # convert to continuous angle (regression head via soft argmax on bins)
            pitch_sm = F.softmax(pitch, dim=1)
            yaw_sm = F.softmax(yaw, dim=1)
            pitch_predicted = torch.sum(pitch_sm * idx_tensor, 1) * params.binwidth - params.angle
            yaw_predicted = torch.sum(yaw_sm * idx_tensor, 1) * params.binwidth - params.angle

            # regression loss
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_regression)
            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_regression)

            # total loss for each head
            loss_pitch = loss_pitch_cls + params.alpha * loss_reg_pitch
            loss_yaw = loss_yaw_cls + params.alpha * loss_reg_yaw
            loss = loss_pitch + loss_yaw

        if device.type == "cuda" and params.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        sum_loss_pitch += loss_pitch.item()
        sum_loss_yaw += loss_yaw.item()

        # live progress info
        seen = max(1, pbar.n)  # batches seen so far
        avg_pitch = sum_loss_pitch / seen
        avg_yaw = sum_loss_yaw / seen

        global_step = epoch * len(data_loader) + pbar.n

        writer.add_scalar("BatchLoss/train_pitch", loss_pitch.item(), global_step)
        writer.add_scalar("BatchLoss/train_yaw",   loss_yaw.item(),   global_step)
        writer.add_scalar("BatchLoss/total",       loss.item(),       global_step)


        pbar.set_postfix({
            "loss_pitch": f"{avg_pitch:.4f}",
            "loss_yaw": f"{avg_yaw:.4f}",
            "total": f"{(avg_pitch + avg_yaw):.4f}",
        })

    avg_loss_pitch = sum_loss_pitch / len(data_loader)
    avg_loss_yaw = sum_loss_yaw / len(data_loader)
    return avg_loss_pitch, avg_loss_yaw

@torch.no_grad()
def validate_one_epoch(
    params,
    model: nn.Module,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    data_loader: DataLoader,
    idx_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Returns:
        (avg_pitch_loss, avg_yaw_loss, mae_pitch_deg, mae_yaw_deg)
    """
    model.eval()
    sum_loss_pitch, sum_loss_yaw = 0.0, 0.0
    sum_abs_err_pitch, sum_abs_err_yaw = 0.0, 0.0
    n_samples = 0

    for images, labels_gaze, regression_labels_gaze, _ in data_loader:
        images = images.to(device)
        label_pitch = labels_gaze[:, 0].to(device)
        label_yaw = labels_gaze[:, 1].to(device)
        label_pitch_regression = regression_labels_gaze[:, 0].to(device)
        label_yaw_regression = regression_labels_gaze[:, 1].to(device)

        bsz = images.size(0)
        n_samples += bsz

        # autocast for faster/lower-mem eval on CUDA
        with amp.autocast('cuda', enabled=(device.type == "cuda" and params.amp)):
            pitch, yaw = model(images)

            # classification loss (mean over batch)
            loss_pitch_cls = cls_criterion(pitch, label_pitch)
            loss_yaw_cls = cls_criterion(yaw, label_yaw)

            # soft-argmax to degrees
            pitch_sm = F.softmax(pitch, dim=1)
            yaw_sm = F.softmax(yaw, dim=1)
            pitch_pred = torch.sum(pitch_sm * idx_tensor, 1) * params.binwidth - params.angle
            yaw_pred = torch.sum(yaw_sm * idx_tensor, 1) * params.binwidth - params.angle

            # regression loss (mean)
            loss_pitch_reg = reg_criterion(pitch_pred, label_pitch_regression)
            loss_yaw_reg = reg_criterion(yaw_pred, label_yaw_regression)

            # total head losses
            loss_pitch = loss_pitch_cls + params.alpha * loss_pitch_reg
            loss_yaw = loss_yaw_cls + params.alpha * loss_yaw_reg

        sum_loss_pitch += loss_pitch.item()
        sum_loss_yaw += loss_yaw.item()

        # MAE (degrees) — accumulate per-sample absolute error
        sum_abs_err_pitch += torch.abs(pitch_pred - label_pitch_regression).sum().item()
        sum_abs_err_yaw += torch.abs(yaw_pred - label_yaw_regression).sum().item()

    avg_loss_pitch = sum_loss_pitch / len(data_loader)
    avg_loss_yaw = sum_loss_yaw / len(data_loader)
    mae_pitch = sum_abs_err_pitch / max(1, n_samples)
    mae_yaw = sum_abs_err_yaw / max(1, n_samples)
    return avg_loss_pitch, avg_loss_yaw, mae_pitch, mae_yaw

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    params = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(params.output, exist_ok=True)

    # TensorBoard writer
    log_dir = params.log_dir or params.output
    writer = SummaryWriter(log_dir=log_dir)

    torch.backends.cudnn.benchmark = True
    model, optimizer, scaler, start_epoch = initialize_model(params, device)

    # ----------------- Build train/val loaders -----------------
    ds = params.dataset.lower()
    if ds == "gaze360":
        # Uses repo’s predefined split files (train/val) via helper
        train_loader = get_dataloader(params, mode="train")
        val_loader = get_dataloader(params, mode="val")
    else:
        # MPIIFaceGaze: reproducible random split
        transform = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        full_ds = MPIIGaze(
            params.data,
            transform,
            angle=params.angle,
            binwidth=params.binwidth,
        )
        val_size = max(1, int(len(full_ds) * params.val_ratio))
        train_size = len(full_ds) - val_size
        g = torch.Generator().manual_seed(params.seed)
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=g)

        train_loader = DataLoader(
            train_ds,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        )

    # Losses / helpers
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    best_val = float("inf")
    logging.info(f"Started training from epoch: {start_epoch + 1}")

    for epoch in range(start_epoch, params.num_epochs):
        t0 = time.time()
        avg_loss_pitch, avg_loss_yaw = train_one_epoch(
            params, model, cls_criterion, reg_criterion, optimizer, scaler,
            train_loader, idx_tensor, device, writer, epoch
        )
        train_total = avg_loss_pitch + avg_loss_yaw
        logging.info(
            f"Epoch [{epoch + 1}/{params.num_epochs}] "
            f"Train -> PitchLoss: {avg_loss_pitch:.4f}, YawLoss: {avg_loss_yaw:.4f}, Total: {train_total:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

        # ---- validation (+ MAE in degrees)
        v_pitch_loss, v_yaw_loss, v_mae_pitch, v_mae_yaw = validate_one_epoch(
            params, model, cls_criterion, reg_criterion, val_loader, idx_tensor, device
        )
        val_total = v_pitch_loss + v_yaw_loss
        logging.info(
            f"Validation -> PitchLoss: {v_pitch_loss:.4f}, YawLoss: {v_yaw_loss:.4f}, "
            f"TotalLoss: {val_total:.4f}, "
            f"MAE(deg) Pitch: {v_mae_pitch:.3f}, Yaw: {v_mae_yaw:.3f}, "
            f"AvgMAE: {(v_mae_pitch + v_mae_yaw) / 2:.3f}"
        )

        # checkpoint (by validation loss)
        ckpt_path = os.path.join(params.output, "checkpoint.ckpt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_total,
            "val_mae_pitch_deg": v_mae_pitch,
            "val_mae_yaw_deg": v_mae_yaw,
        }, ckpt_path)
        logging.info(f"Checkpoint saved at {ckpt_path}")

        if val_total < best_val:
            best_val = val_total
            best_model_path = os.path.join(params.output, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"Best model updated -> {best_model_path} "
                f"(val total {best_val:.4f}, MAE P:{v_mae_pitch:.3f} / Y:{v_mae_yaw:.3f})"
            )

        # ---- TensorBoard scalars (epoch-level)
        step = epoch + 1
        writer.add_scalar("Loss/train_pitch", avg_loss_pitch, step)
        writer.add_scalar("Loss/train_yaw",   avg_loss_yaw,  step)
        writer.add_scalar("Loss/val_pitch",   v_pitch_loss,  step)
        writer.add_scalar("Loss/val_yaw",     v_yaw_loss,    step)
        writer.add_scalar("Loss/val_total",   val_total,     step)
        writer.add_scalar("MAE/val_pitch_deg", v_mae_pitch,  step)
        writer.add_scalar("MAE/val_yaw_deg",   v_mae_yaw,    step)
        writer.add_scalar("LR/base", optimizer.param_groups[0]["lr"], step)

    writer.close()

if __name__ == "__main__":
    main()
