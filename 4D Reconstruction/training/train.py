"""Phase 1 pretraining loop on Epic-Kitchens100."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.epic_kitchens import EpicKitchensDataset
from models.repr_4d_model import Repr4DModel
from training.loss import L1PointLoss, TemporalConsistencyLoss
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    full = EpicKitchensDataset(
        root_dir=cfg.data.root_dir,
        num_frames=cfg.data.num_frames,
        num_points=cfg.data.num_points,
        frame_size=tuple(cfg.data.frame_size),
    )
    n = len(full)
    n_train = int(cfg.data.split[0] * n)
    n_val = max(1, int(cfg.data.split[1] * n))
    n_test = n - n_train - n_val
    train_set, val_set, _ = random_split(
        full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def _init_wandb(cfg: DictConfig) -> None:
    if not cfg.use_wandb:
        return
    try:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except ImportError:
        logger.warning("wandb not installed; skipping logging.")


def _log_step(step: int, metrics: Dict[str, float], cfg: DictConfig) -> None:
    if cfg.use_wandb:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass
    if step % cfg.log_interval == 0:
        msg = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("step %d | %s", step, msg)


def evaluate(model: Repr4DModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = L1PointLoss()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            tracks = batch["tracks"].to(device)
            pred = model(frames, tracks)
            total += loss_fn(pred, tracks).item() * frames.size(0)
            count += frames.size(0)
    model.train()
    return total / max(1, count)


@hydra.main(config_path="../configs", config_name="training", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    _init_wandb(cfg)

    train_loader, val_loader = _build_dataloaders(cfg)
    logger.info("Train batches=%d  Val batches=%d", len(train_loader), len(val_loader))

    model = Repr4DModel(cfg.model).to(device)
    l1_loss = L1PointLoss()
    tc_loss = TemporalConsistencyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader),
        eta_min=float(cfg.scheduler.min_lr),
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0

    for epoch in range(cfg.epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            frames = batch["frames"].to(device, non_blocking=True)
            tracks = batch["tracks"].to(device, non_blocking=True)

            pred = model(frames, tracks)
            loss_main = l1_loss(pred, tracks) * cfg.loss.l1_weight
            loss_tc = tc_loss(pred, tracks) * cfg.loss.temporal_consistency_weight
            loss = loss_main + loss_tc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            _log_step(step, {
                "train/loss": loss.item(),
                "train/l1": loss_main.item(),
                "train/tc": loss_tc.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }, cfg)

        if (epoch + 1) % cfg.eval_interval == 0:
            val = evaluate(model, val_loader, device)
            logger.info("epoch %d | val/l1=%.4f", epoch, val)
            if val < best_val:
                best_val = val
                torch.save({
                    "model": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch,
                    "val_loss": val,
                }, ckpt_dir / "phase1_best.pth")
                logger.info("Saved new best checkpoint (val=%.4f).", val)

    torch.save({"model": model.state_dict()}, ckpt_dir / "phase1_last.pth")
    logger.info("Training complete. Best val loss=%.4f", best_val)


if __name__ == "__main__":
    main()
