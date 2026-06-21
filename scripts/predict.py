import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import config
from data.dataset import preprocess_fundus, IMAGENET_MEAN, IMAGENET_STD
from models.model import DRMultiTaskNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR", "Ungradable"]
LESION_TYPES = ["MA", "HE", "EX", "SE"]
LESION_COLORS = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]


def load_and_preprocess(image_path, image_size=512):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = preprocess_fundus(image, image_size)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    tensor = (image.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.tensor(tensor.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return tensor, image


@torch.no_grad()
def predict(model, image_tensor, device, seg_thresholds=None):
    if seg_thresholds is None:
        seg_thresholds = [0.5] * 4

    model.eval()
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)

    grade_logits = outputs["grade_logits"]
    grade_probs = torch.softmax(grade_logits, dim=1).squeeze(0).cpu().numpy()
    grade_pred = int(grade_probs.argmax())

    seg_logits = outputs["seg_logits"]
    seg_probs = torch.sigmoid(seg_logits).squeeze(0).cpu().numpy()
    seg_masks = np.stack([
        (seg_probs[i] >= seg_thresholds[i]).astype(np.uint8)
        for i in range(len(LESION_TYPES))
    ], axis=0)

    return {
        "grade_pred": grade_pred,
        "grade_name": GRADE_NAMES[grade_pred],
        "grade_probs": grade_probs.tolist(),
        "seg_probs": seg_probs,
        "seg_masks": seg_masks,
        "referable_dr": grade_pred >= 2,
        "vision_threatening_dr": grade_pred >= 3,
    }


def visualize_prediction(image_rgb, prediction, save_path=None):
    fig, axes = plt.subplots(1, len(LESION_TYPES) + 2, figsize=(4 * (len(LESION_TYPES) + 2), 4))

    axes[0].imshow(image_rgb)
    grade = prediction["grade_pred"]
    grade_name = prediction["grade_name"]
    ref = "REFERABLE" if prediction["referable_dr"] else "Non-referable"
    axes[0].set_title(f"Grade {grade}: {grade_name}\n{ref}", fontsize=10)
    axes[0].axis("off")

    probs = prediction["grade_probs"]
    bars = axes[1].barh(GRADE_NAMES, probs, color="steelblue")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Grade Probabilities")
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{p:.2f}", va="center", fontsize=8)

    for i, (lesion, color) in enumerate(zip(LESION_TYPES, LESION_COLORS)):
        mask = prediction["seg_masks"][i]
        overlay = image_rgb.copy()
        colored = np.zeros_like(overlay)
        colored[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)
        axes[i + 2].imshow(overlay)
        axes[i + 2].set_title(f"{lesion} Mask")
        axes[i + 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image or folder of images")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_thresholds = [0.5] * 4
    if args.thresholds and Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            thr_data = json.load(f)
        seg_thresholds = thr_data.get("optimal_thresholds", [0.5] * 4)

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

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.jpeg")) +
            list(input_path.glob("*.png")) +
            list(input_path.glob("*.JPG")) +
            list(input_path.glob("*.PNG"))
        )

    logger.info(f"Processing {len(image_paths)} image(s)...")
    all_results = []

    for img_path in image_paths:
        try:
            image_tensor, image_rgb = load_and_preprocess(img_path, config["image_size"])
            prediction = predict(model, image_tensor, device, seg_thresholds)

            result = {
                "image": str(img_path),
                "grade_pred": prediction["grade_pred"],
                "grade_name": prediction["grade_name"],
                "grade_probs": prediction["grade_probs"],
                "referable_dr": prediction["referable_dr"],
                "vision_threatening_dr": prediction["vision_threatening_dr"],
            }
            all_results.append(result)

            logger.info(
                f"{img_path.name} -> Grade {prediction['grade_pred']}: {prediction['grade_name']} "
                f"| Referable: {prediction['referable_dr']}"
            )

            if args.visualize:
                viz_path = output_dir / f"{img_path.stem}_prediction.png"
                visualize_prediction(image_rgb, prediction, save_path=str(viz_path))

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    results_path = output_dir / "predictions.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nPredictions saved to {results_path}")

    if len(all_results) > 1:
        grade_counts = {}
        for r in all_results:
            g = r["grade_name"]
            grade_counts[g] = grade_counts.get(g, 0) + 1
        referable_count = sum(1 for r in all_results if r["referable_dr"])
        logger.info("\nSummary:")
        for g, c in sorted(grade_counts.items()):
            logger.info(f"  {g}: {c}")
        logger.info(f"  Referable DR: {referable_count}/{len(all_results)}")


if __name__ == "__main__":
    main()
