import sys
import json
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import config
from data.dataset import DRDataset, get_val_transforms, collate_fn
from data.data_utils import load_idrid_samples
from models.model import DRMultiTaskNet
from utils.metrics import run_inference, compute_segmentation_metrics, bootstrap_ci
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_reliability_diagram,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idrid_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default=None,
                        help="JSON file with optimal segmentation thresholds from cross-val")
    parser.add_argument("--output_dir", type=str, default="outputs/external_eval")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading IDRiD samples...")
    samples = load_idrid_samples(args.idrid_root)
    logger.info(f"IDRiD samples: {len(samples)}")

    seg_thresholds = [0.5] * 4
    if args.thresholds and Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            thr_data = json.load(f)
        seg_thresholds = thr_data.get("optimal_thresholds", [0.5] * 4)
    logger.info(f"Using segmentation thresholds: {seg_thresholds}")

    val_transforms = get_val_transforms(config["image_size"])
    dataset = DRDataset(samples, config["image_size"], val_transforms, mode="val")
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size_lesion"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    model = DRMultiTaskNet(
        num_grades=config["num_grades"],
        num_lesion_types=config["num_lesion_types"],
        fpn_channels=config["fpn_channels"],
        dropout=config["dropout"],
    )
    state = torch.load(args.checkpoint, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    logger.info("Running inference on IDRiD...")
    results = run_inference(model, loader, device, seg_thresholds=seg_thresholds)

    grade_preds = results["raw"]["grade_preds"]
    grade_probs = results["raw"]["grade_probs"]
    grade_targets = results["raw"]["grade_targets"]

    def acc_fn(data):
        preds = [d[0] for d in data]
        targets = [d[1] for d in data]
        return np.mean(np.array(preds) == np.array(targets))

    paired = list(zip(grade_preds, grade_targets))
    acc_lo, acc_hi = bootstrap_ci(acc_fn, paired, n_resamples=config["bootstrap_n_resamples"])

    grading = results["grading"]
    logger.info("\n--- IDRiD Grading Results ---")
    logger.info(f"Accuracy:  {grading['accuracy']:.4f}  95% CI: [{acc_lo:.4f}, {acc_hi:.4f}]")
    logger.info(f"QWK:       {grading['qwk']:.4f}")
    logger.info(f"Macro F1:  {grading['macro_f1']:.4f}")
    logger.info(f"Macro AUC: {grading['macro_auc']:.4f}")
    logger.info(f"Referable DR  - Sens: {grading['referable_dr']['sensitivity']:.4f}  Spec: {grading['referable_dr']['specificity']:.4f}")
    logger.info(f"VT DR         - Sens: {grading['vision_threatening_dr']['sensitivity']:.4f}  Spec: {grading['vision_threatening_dr']['specificity']:.4f}")
    logger.info(f"ECE: {results['calibration']['ece']:.4f}")

    if "segmentation" in results:
        seg = results["segmentation"]
        logger.info("\n--- IDRiD Segmentation Results ---")
        for lesion in ["MA", "HE", "EX", "SE"]:
            logger.info(f"{lesion}: Dice={seg[lesion]['dice']:.4f}  IoU={seg[lesion]['iou']:.4f}  AUPR={seg[lesion]['aupr']:.4f}")
        logger.info(f"Mean Dice={seg['mean']['dice']:.4f}  Mean IoU={seg['mean']['iou']:.4f}")

    cm = grading["confusion_matrix"]
    plot_confusion_matrix(cm, save_path=str(output_dir / "idrid_confusion_matrix.png"))
    plot_roc_curves(grade_probs, grade_targets,
                    num_classes=config["num_grades"],
                    save_path=str(output_dir / "idrid_roc_curves.png"))
    plot_reliability_diagram(grade_probs, grade_targets,
                              save_path=str(output_dir / "idrid_reliability_diagram.png"))

    serializable_results = {
        "grading": {
            k: v for k, v in grading.items()
            if not isinstance(v, np.ndarray)
        },
        "calibration": results["calibration"],
        "accuracy_ci": {"lower": acc_lo, "upper": acc_hi},
    }
    if "segmentation" in results:
        seg_out = {}
        for lesion in ["MA", "HE", "EX", "SE", "mean"]:
            seg_out[lesion] = {k: float(v) for k, v in results["segmentation"][lesion].items()}
        serializable_results["segmentation"] = seg_out

    with open(output_dir / "idrid_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
