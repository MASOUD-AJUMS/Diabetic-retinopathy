import sys
import json
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import config
from data.dataset import DRDataset, get_train_transforms, get_val_transforms, collate_fn
from data.data_utils import load_ddr_samples, deduplicate_and_split, compute_class_weights
from models.model import DRMultiTaskNet
from models.losses import MultiTaskLoss
from models.trainer import Trainer, make_balanced_sampler
from utils.metrics import run_inference, find_optimal_thresholds


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


ABLATION_CONFIGS = {
    "single_task_cls": {
        "use_seg": False,
        "use_det": False,
        "use_cross_attention": False,
        "use_uncertainty_weighting": False,
        "use_progressive_schedule": False,
    },
    "single_task_seg": {
        "use_seg": True,
        "use_det": False,
        "use_cross_attention": False,
        "use_uncertainty_weighting": False,
        "use_progressive_schedule": False,
        "seg_only": True,
    },
    "no_cross_attention": {
        "use_seg": True,
        "use_det": True,
        "use_cross_attention": False,
        "use_uncertainty_weighting": True,
        "use_progressive_schedule": True,
    },
    "two_task_seg_cls": {
        "use_seg": True,
        "use_det": False,
        "use_cross_attention": True,
        "use_uncertainty_weighting": True,
        "use_progressive_schedule": True,
    },
    "no_uncertainty_weighting": {
        "use_seg": True,
        "use_det": True,
        "use_cross_attention": True,
        "use_uncertainty_weighting": False,
        "use_progressive_schedule": True,
    },
    "no_progressive_schedule": {
        "use_seg": True,
        "use_det": True,
        "use_cross_attention": True,
        "use_uncertainty_weighting": True,
        "use_progressive_schedule": False,
    },
    "full_model": {
        "use_seg": True,
        "use_det": True,
        "use_cross_attention": True,
        "use_uncertainty_weighting": True,
        "use_progressive_schedule": True,
    },
}


class AblationModel(DRMultiTaskNet):
    def __init__(self, ablation_cfg, **kwargs):
        super().__init__(**kwargs)
        self.ablation_cfg = ablation_cfg

        if not ablation_cfg.get("use_cross_attention", True):
            import torch.nn as nn
            self.cls_head.attn_seg = nn.Identity()
            self.cls_head.attn_det = nn.Identity()

    def forward(self, x):
        import torch.nn.functional as F
        fpn_out = self.extract_features(x)
        p2, p3, p4, p5 = fpn_out["layer1"], fpn_out["layer2"], fpn_out["layer3"], fpn_out["layer4"]

        seg_logits = self.seg_head(p2)
        seg_logits_resized = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        det_cls, det_reg, det_centerness = self.det_head([p3, p4, p5])

        if self.ablation_cfg.get("use_cross_attention", True):
            seg_feat = F.interpolate(torch.sigmoid(seg_logits), size=p5.shape[2:], mode="bilinear", align_corners=False)
            seg_ctx = self.seg_proj(seg_feat)
            det_feat = F.interpolate(torch.sigmoid(det_cls[0]), size=p5.shape[2:], mode="bilinear", align_corners=False)
            det_ctx = self.det_proj(det_feat)
            grade_logits = self.cls_head(p5, seg_ctx, det_ctx)
        else:
            import torch.nn as nn
            grade_logits = self.cls_head.pool(p5).flatten(1)
            grade_logits = self.cls_head.classifier(grade_logits)

        return {
            "seg_logits": seg_logits_resized,
            "det_cls": det_cls,
            "det_reg": det_reg,
            "det_centerness": det_centerness,
            "grade_logits": grade_logits,
        }


def run_ablation_variant(name, abl_cfg, train_samples, val_samples, device, output_dir):
    logger.info(f"\nRunning ablation variant: {name}")

    class_weights = compute_class_weights(train_samples, num_classes=config["num_grades"])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_transforms = get_train_transforms(config["image_size"])
    val_transforms = get_val_transforms(config["image_size"])

    train_dataset = DRDataset(train_samples, config["image_size"], train_transforms, mode="train")
    val_dataset = DRDataset(val_samples, config["image_size"], val_transforms, mode="val")

    model = AblationModel(
        ablation_cfg=abl_cfg,
        num_grades=config["num_grades"],
        num_lesion_types=config["num_lesion_types"],
        fpn_channels=config["fpn_channels"],
        dropout=config["dropout"],
    ).to(device)

    loss_fn = MultiTaskLoss(class_weights=class_weights_tensor)
    variant_output_dir = Path(output_dir) / name
    trainer = Trainer(model, loss_fn, device, config, str(variant_output_dir))

    sampler = make_balanced_sampler(train_samples)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size_full"],
        sampler=sampler, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size_full"],
        shuffle=False, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True,
    )

    if abl_cfg.get("use_progressive_schedule", True) and abl_cfg.get("use_seg", True):
        annotated = [s for s in train_samples if s.get("has_lesion_annotation")]
        ann_dataset = DRDataset(annotated, config["image_size"], train_transforms, mode="train")
        ann_sampler = make_balanced_sampler(annotated)
        ann_loader = DataLoader(
            ann_dataset, batch_size=config["batch_size_lesion"],
            sampler=ann_sampler, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True,
        )
        trainer.train_stage(ann_loader, val_loader, "seg", config["seg_epochs"], config["patience"])
        if abl_cfg.get("use_det", True):
            trainer.train_stage(ann_loader, val_loader, "det", config["det_epochs"], config["patience"])
        trainer.train_stage(train_loader, val_loader, "full", config["full_epochs"], config["patience"])
    elif abl_cfg.get("seg_only", False):
        annotated = [s for s in train_samples if s.get("has_lesion_annotation")]
        ann_dataset = DRDataset(annotated, config["image_size"], train_transforms, mode="train")
        ann_sampler = make_balanced_sampler(annotated)
        ann_loader = DataLoader(
            ann_dataset, batch_size=config["batch_size_lesion"],
            sampler=ann_sampler, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True,
        )
        ann_val = [s for s in val_samples if s.get("has_lesion_annotation")]
        ann_val_dataset = DRDataset(ann_val, config["image_size"], val_transforms, mode="val")
        ann_val_loader = DataLoader(
            ann_val_dataset, batch_size=config["batch_size_lesion"],
            shuffle=False, collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True,
        )
        trainer.train_stage(ann_loader, ann_val_loader, "seg", config["full_epochs"], config["patience"])
    else:
        trainer.train_stage(train_loader, val_loader, "full", config["full_epochs"], config["patience"])

    results = run_inference(model, val_loader, device)
    return results


def mcnemar_test(preds_a, preds_b, targets):
    preds_a, preds_b, targets = np.array(preds_a), np.array(preds_b), np.array(targets)
    correct_a = preds_a == targets
    correct_b = preds_b == targets
    n01 = ((~correct_a) & correct_b).sum()
    n10 = (correct_a & (~correct_b)).sum()
    if n01 + n10 == 0:
        return 1.0
    from scipy.stats import chi2
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1 - chi2.cdf(statistic, df=1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddr_root", type=str, required=True)
    parser.add_argument("--lesion_annotation_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_ddr_samples(args.ddr_root, args.lesion_annotation_dir)
    folds = deduplicate_and_split(all_samples, n_splits=config["n_folds"], seed=args.seed)
    fold = folds[args.fold]
    train_samples, val_samples = fold["train"], fold["val"]

    variant_results = {}
    for name, abl_cfg in ABLATION_CONFIGS.items():
        results = run_ablation_variant(name, abl_cfg, train_samples, val_samples, device, str(output_dir))
        variant_results[name] = results

    full_preds = variant_results["full_model"]["raw"]["grade_preds"]
    full_targets = variant_results["full_model"]["raw"]["grade_targets"]

    p_values = {}
    for name in ABLATION_CONFIGS:
        if name == "full_model":
            continue
        other_preds = variant_results[name]["raw"]["grade_preds"]
        p = mcnemar_test(full_preds, other_preds, full_targets)
        p_values[name] = p

    variant_names = list(p_values.keys())
    raw_pvals = [p_values[n] for n in variant_names]
    _, adj_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")
    adjusted_p_values = {n: float(p) for n, p in zip(variant_names, adj_pvals)}

    summary = {}
    for name, results in variant_results.items():
        grading = results.get("grading", {})
        seg = results.get("segmentation", {}).get("mean", {})
        summary[name] = {
            "accuracy": grading.get("accuracy"),
            "qwk": grading.get("qwk"),
            "mean_dice": seg.get("dice"),
            "adj_p_vs_full": adjusted_p_values.get(name, None),
        }

    logger.info("\n--- Ablation Results ---")
    header = f"{'Variant':<35} {'Acc':>8} {'QWK':>8} {'Dice':>8} {'Adj-p':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, vals in summary.items():
        acc = f"{vals['accuracy']:.4f}" if vals["accuracy"] is not None else "---"
        qwk = f"{vals['qwk']:.4f}" if vals["qwk"] is not None else "---"
        dice = f"{vals['mean_dice']:.4f}" if vals["mean_dice"] is not None else "---"
        adj_p = f"{vals['adj_p_vs_full']:.4f}" if vals["adj_p_vs_full"] is not None else "ref"
        logger.info(f"{name:<35} {acc:>8} {qwk:>8} {dice:>8} {adj_p:>10}")

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nAblation results saved to {output_dir}")


if __name__ == "__main__":
    main()
