import os
import sys
import json
import logging
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import config
from data.dataset import DRDataset, get_train_transforms, get_val_transforms, collate_fn
from data.data_utils import (
    load_ddr_samples,
    deduplicate_and_split,
    compute_class_weights,
)
from models.model import DRMultiTaskNet
from models.losses import MultiTaskLoss
from models.trainer import Trainer
from utils.metrics import run_inference, find_optimal_thresholds


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddr_root", type=str, required=True)
    parser.add_argument("--lesion_annotation_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/crossval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_folds", type=int, default=config["n_folds"])
    parser.add_argument("--n_seeds", type=int, default=config["n_seeds"])
    return parser.parse_args()


def train_one_fold(fold_idx, seed, train_samples, val_samples, device, output_dir):
    set_seed(seed)

    class_weights = compute_class_weights(train_samples, num_classes=config["num_grades"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_transforms = get_train_transforms(config["image_size"])
    val_transforms = get_val_transforms(config["image_size"])

    train_dataset = DRDataset(train_samples, config["image_size"], train_transforms, mode="train")
    val_dataset = DRDataset(val_samples, config["image_size"], val_transforms, mode="val")

    model = DRMultiTaskNet(
        num_grades=config["num_grades"],
        num_lesion_types=config["num_lesion_types"],
        fpn_channels=config["fpn_channels"],
        dropout=config["dropout"],
    )

    loss_fn = MultiTaskLoss(class_weights=class_weights_tensor)

    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}_seed_{seed}"
    trainer = Trainer(model, loss_fn, device, config, str(fold_output_dir))

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size_lesion"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    annotated_train = [s for s in train_samples if s.get("has_lesion_annotation")]
    train_seg_dataset = DRDataset(annotated_train if annotated_train else train_samples,
                                  config["image_size"], train_transforms, mode="train")
    full_train_dataset = DRDataset(train_samples, config["image_size"], train_transforms, mode="train")

    from models.trainer import make_balanced_sampler

    seg_sampler = make_balanced_sampler(annotated_train if annotated_train else train_samples)
    full_sampler = make_balanced_sampler(train_samples)

    seg_train_loader = DataLoader(
        train_seg_dataset,
        batch_size=config["batch_size_lesion"],
        sampler=seg_sampler,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=config["batch_size_full"],
        sampler=full_sampler,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    trainer.train_stage(
        seg_train_loader, val_loader, "seg",
        config["seg_epochs"], config["patience"]
    )
    trainer.train_stage(
        seg_train_loader, val_loader, "det",
        config["det_epochs"], config["patience"]
    )

    full_val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size_full"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    trainer.train_stage(
        full_train_loader, full_val_loader, "full",
        config["full_epochs"], config["patience"]
    )

    annotated_val = [s for s in val_samples if s.get("has_lesion_annotation")]
    annotated_val_dataset = DRDataset(annotated_val, config["image_size"], val_transforms, mode="val")
    annotated_val_loader = DataLoader(
        annotated_val_dataset,
        batch_size=config["batch_size_lesion"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
    )

    seg_preds_all, seg_targets_all = [], []
    for batch in annotated_val_loader:
        imgs = batch["image"].to(device)
        with torch.no_grad():
            out = model(imgs)
        import torch.nn.functional as F
        prob = torch.sigmoid(out["seg_logits"]).cpu().numpy()
        for i in range(len(imgs)):
            if batch["has_lesion_annotation"][i]:
                seg_preds_all.append(prob[i])
                seg_targets_all.append(batch["masks"][i].numpy())

    optimal_thresholds = [0.5] * 4
    if seg_preds_all:
        optimal_thresholds = find_optimal_thresholds(seg_preds_all, seg_targets_all)

    results = run_inference(model, val_loader, device, seg_thresholds=optimal_thresholds)
    results["optimal_thresholds"] = optimal_thresholds

    results_path = fold_output_dir / "val_results.json"
    serializable = {
        k: v for k, v in results.items()
        if k != "raw" and not isinstance(v, np.ndarray)
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(
        f"Fold {fold_idx} Seed {seed} | "
        f"Acc: {results['grading']['accuracy']:.4f} | "
        f"QWK: {results['grading']['qwk']:.4f} | "
        f"Mean Dice: {results['segmentation']['mean']['dice'] if 'segmentation' in results else 'N/A':.4f}"
    )

    return results, model.state_dict(), optimal_thresholds


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DDR samples...")
    all_samples = load_ddr_samples(args.ddr_root, args.lesion_annotation_dir)
    logger.info(f"Total samples: {len(all_samples)}")

    seeds = list(range(args.n_seeds))
    all_results = []

    for seed in seeds:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting cross-validation with seed {seed}")
        logger.info(f"{'='*50}")

        folds = deduplicate_and_split(
            all_samples,
            n_splits=args.n_folds,
            seed=seed,
            hamming_threshold=config["hamming_threshold"],
        )

        seed_results = []
        for fold_idx, fold in enumerate(folds):
            logger.info(f"\nFold {fold_idx + 1}/{args.n_folds}")
            results, state_dict, thresholds = train_one_fold(
                fold_idx, seed,
                fold["train"], fold["val"],
                device, str(output_dir),
            )
            seed_results.append(results)

        all_results.append(seed_results)

    fold_accs = [r["grading"]["accuracy"] for sr in all_results for r in sr]
    fold_qwks = [r["grading"]["qwk"] for sr in all_results for r in sr]

    summary = {
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "qwk_mean": float(np.mean(fold_qwks)),
        "qwk_std": float(np.std(fold_qwks)),
        "n_folds": args.n_folds,
        "n_seeds": args.n_seeds,
    }

    with open(output_dir / "crossval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nCross-validation Summary:")
    logger.info(f"Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    logger.info(f"QWK:      {summary['qwk_mean']:.4f} ± {summary['qwk_std']:.4f}")


if __name__ == "__main__":
    main()
