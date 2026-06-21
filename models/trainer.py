import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import logging

from models.losses import MultiTaskLoss


logger = logging.getLogger(__name__)


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def get_optimizer_with_layer_decay(model, base_lr=1e-4, weight_decay=0.01, layer_decay=0.9):
    backbone_layers = [
        ("layer4", 1),
        ("layer3", layer_decay),
        ("layer2", layer_decay ** 2),
        ("layer1", layer_decay ** 3),
        ("layer0", layer_decay ** 4),
    ]
    param_groups = []
    backbone_param_ids = set()
    for name, lr_mult in backbone_layers:
        layer = getattr(model, name, None)
        if layer is not None:
            params = list(layer.parameters())
            backbone_param_ids.update(id(p) for p in params)
            param_groups.append({"params": params, "lr": base_lr * lr_mult})

    other_params = [p for p in model.parameters()
                    if id(p) not in backbone_param_ids and p.requires_grad]
    param_groups.append({"params": other_params, "lr": base_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay, betas=(0.9, 0.999))


def get_scheduler(optimizer, num_epochs, warmup_epochs=1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_balanced_sampler(samples, num_classes=6):
    grades = np.array([s["grade"] for s in samples])
    class_counts = np.bincount(grades[grades >= 0], minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[g] if g >= 0 else 0.0 for g in grades])
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(samples),
        replacement=True,
    )


class Trainer:
    def __init__(self, model, loss_fn, device, config, output_dir):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, loader, optimizer, stage, training=True):
        self.model.train() if training else self.model.eval()
        total_losses = {"total": 0, "seg": 0, "det": 0, "cls": 0}
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                images = batch["image"].to(self.device)
                batch_device = {
                    "grade": batch["grade"].to(self.device),
                    "masks": batch["masks"].to(self.device),
                    "has_lesion_annotation": batch["has_lesion_annotation"],
                    "det_targets": batch["det_targets"],
                }

                outputs = self.model(images)
                losses = self.loss_fn(outputs, batch_device, stage=stage)

                if training:
                    optimizer.zero_grad()
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                for k in total_losses:
                    val = losses[k]
                    total_losses[k] += val.item() if isinstance(val, torch.Tensor) else val
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def train_stage(self, train_loader, val_loader, stage, max_epochs, patience=10):
        cfg = self.config
        base_lr = cfg.get("base_lr", 1e-4)

        if stage == "seg":
            freeze_module(self.model.cls_head)
            freeze_module(self.model.det_head)
            unfreeze_module(self.model.seg_head)
            unfreeze_module(self.model.fpn)
            for layer in [self.model.layer0, self.model.layer1,
                          self.model.layer2, self.model.layer3, self.model.layer4]:
                unfreeze_module(layer)
        elif stage == "det":
            unfreeze_module(self.model.det_head)
            freeze_module(self.model.cls_head)
        else:
            unfreeze_module(self.model)

        if stage == "full":
            optimizer = get_optimizer_with_layer_decay(
                self.model, base_lr=base_lr,
                weight_decay=cfg.get("weight_decay", 0.01),
                layer_decay=cfg.get("layer_decay", 0.9),
            )
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=base_lr,
                weight_decay=cfg.get("weight_decay", 0.01),
            )

        scheduler = get_scheduler(optimizer, max_epochs, warmup_epochs=1)
        best_val_loss = float("inf")
        patience_counter = 0
        best_ckpt_path = self.output_dir / f"best_{stage}.pth"

        for epoch in range(max_epochs):
            train_losses = self._run_epoch(train_loader, optimizer, stage, training=True)
            val_losses = self._run_epoch(val_loader, optimizer, stage, training=False)
            scheduler.step()

            logger.info(
                f"[{stage}] Epoch {epoch+1}/{max_epochs} | "
                f"Train: {train_losses['total']:.4f} | Val: {val_losses['total']:.4f}"
            )

            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "stage": stage,
                }, best_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (stage: {stage})")
                    break

        checkpoint = torch.load(best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Loaded best {stage} checkpoint (val_loss={best_val_loss:.4f})")
        return best_val_loss

    def train_progressive(self, train_samples, val_samples, train_dataset_cls,
                          val_dataset_cls, collate_fn):
        from data.data_utils import make_balanced_sampler as mbs
        cfg = self.config

        sampler = make_balanced_sampler(train_samples)
        train_loader = DataLoader(
            train_dataset_cls,
            batch_size=cfg.get("batch_size_lesion", 4),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset_cls,
            batch_size=cfg.get("batch_size_lesion", 4),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

        self.train_stage(train_loader, val_loader, "seg",
                         cfg.get("seg_epochs", 100),
                         patience=cfg.get("patience", 10))

        self.train_stage(train_loader, val_loader, "det",
                         cfg.get("det_epochs", 100),
                         patience=cfg.get("patience", 10))

        full_loader = DataLoader(
            train_dataset_cls,
            batch_size=cfg.get("batch_size_full", 8),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )
        full_val_loader = DataLoader(
            val_dataset_cls,
            batch_size=cfg.get("batch_size_full", 8),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

        self.train_stage(full_loader, full_val_loader, "full",
                         cfg.get("full_epochs", 100),
                         patience=cfg.get("patience", 10))

        final_path = self.output_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"Training complete. Model saved to {final_path}")
