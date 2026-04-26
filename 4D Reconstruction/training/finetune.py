"""Phase 2 finetuning on robot demonstration videos."""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.epic_kitchens import EpicKitchensDataset
from models.repr_4d_model import Repr4DModel
from training.loss import L1PointLoss, TemporalConsistencyLoss
from utils.logger import get_logger

logger = get_logger(__name__)


def load_pretrained(model: Repr4DModel, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    weights = state.get("model", state)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    logger.info(
        "Loaded pretrained weights from %s (missing=%d, unexpected=%d).",
        ckpt_path, len(missing), len(unexpected),
    )


@hydra.main(config_path="../configs", config_name="training", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Phase 2 finetune on device=%s", device)

    # Robot demonstration data shares the same on-disk format; subclass for
    # any robot-specific metadata you need to load.
    dataset = EpicKitchensDataset(
        root_dir=cfg.data.root_dir,
        num_frames=cfg.data.num_frames,
        num_points=cfg.data.num_points,
        frame_size=tuple(cfg.data.frame_size),
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )

    model = Repr4DModel(cfg.model).to(device)
    pretrained_ckpt = Path(cfg.finetune.pretrained_ckpt)
    if pretrained_ckpt.exists():
        load_pretrained(model, pretrained_ckpt)
    else:
        logger.warning("Pretrained checkpoint %s not found; training from scratch.",
                       pretrained_ckpt)

    if cfg.finetune.freeze_backbone:
        for p in model.vit_encoder.parameters():
            p.requires_grad = False
        logger.info("Frozen ViT backbone for finetuning.")

    l1_loss = L1PointLoss()
    tc_loss = TemporalConsistencyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.finetune.lr, weight_decay=cfg.weight_decay,
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.finetune.epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"finetune ep{epoch}"):
            frames = batch["frames"].to(device)
            tracks = batch["tracks"].to(device)
            pred = model(frames, tracks)
            loss = (
                cfg.loss.l1_weight * l1_loss(pred, tracks)
                + cfg.loss.temporal_consistency_weight * tc_loss(pred, tracks)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += loss.item()
        logger.info("finetune epoch %d | mean_loss=%.4f", epoch, running / len(loader))

    torch.save({
        "model": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }, ckpt_dir / "phase2_best.pth")
    logger.info("Saved finetuned weights to %s", ckpt_dir / "phase2_best.pth")


if __name__ == "__main__":
    main()
