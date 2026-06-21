import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import config
from data.dataset import DRDataset, get_val_transforms, collate_fn
from data.data_utils import load_ddr_samples, load_idrid_samples
from models.model import DRMultiTaskNet
from utils.visualization import (
    GradCAM,
    plot_gradcam_panel,
    plot_segmentation_results,
    plot_tsne,
    denormalize,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR", "Ungradable"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ddr", "idrid"], default="ddr")
    parser.add_argument("--lesion_annotation_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/interpretability")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_per_grade", type=int, default=3)
    return parser.parse_args()


def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, labels = [], []

    hooks = []
    captured = {}

    def hook_fn(module, input, output):
        captured["embedding"] = output.detach().cpu()

    h = model.cls_head.pool.register_forward_hook(hook_fn)
    hooks.append(h)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            _ = model(images)
            emb = captured["embedding"].squeeze(-1).squeeze(-1)
            embeddings.append(emb.numpy())
            labels.extend(batch["grade"].numpy().tolist())

    for h in hooks:
        h.remove()

    return np.concatenate(embeddings, axis=0), np.array(labels)


def select_representative_samples(samples, n_per_grade=3, num_grades=6):
    from collections import defaultdict
    import random
    grade_buckets = defaultdict(list)
    for s in samples:
        grade_buckets[s["grade"]].append(s)
    selected = []
    for g in range(num_grades):
        bucket = grade_buckets[g]
        chosen = random.sample(bucket, min(n_per_grade, len(bucket)))
        selected.extend(chosen)
    return selected


def run_gradcam_analysis(model, samples, device, output_dir, n_per_grade=3):
    selected = select_representative_samples(samples, n_per_grade=n_per_grade)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    val_transforms = get_val_transforms(config["image_size"])
    dataset = DRDataset(selected, config["image_size"], val_transforms, mode="val")

    images_list, cams_list, grade_preds_list, grade_targets_list = [], [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"].to(device)
        grade_target = sample["grade"].item()

        cam, grade_pred = gradcam.generate(image, target_class=None)

        images_list.append(sample["image"])
        cams_list.append(cam)
        grade_preds_list.append(grade_pred)
        grade_targets_list.append(grade_target)

    fig = plot_gradcam_panel(images_list, cams_list, grade_preds_list, grade_targets_list,
                              save_path=str(output_dir / "gradcam_panel.png"))
    logger.info(f"Grad-CAM panel saved.")
    return images_list, cams_list, grade_preds_list, grade_targets_list


def run_segmentation_visualization(model, samples, device, output_dir, n_samples=5):
    annotated = [s for s in samples if s.get("has_lesion_annotation")]
    if not annotated:
        logger.info("No annotated samples for segmentation visualization.")
        return

    import random
    chosen = random.sample(annotated, min(n_samples, len(annotated)))

    val_transforms = get_val_transforms(config["image_size"])
    dataset = DRDataset(chosen, config["image_size"], val_transforms, mode="val")

    model.eval()
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(image)
        seg_prob = torch.sigmoid(out["seg_logits"]).squeeze(0).cpu().numpy()
        gt_masks = sample["masks"].numpy()
        fig = plot_segmentation_results(
            sample["image"], seg_prob, gt_masks,
            save_path=str(output_dir / f"seg_viz_{i}.png")
        )
        logger.info(f"Segmentation visualization {i} saved.")


def run_tsne_analysis(model, loader, device, output_dir):
    logger.info("Extracting embeddings for t-SNE...")
    embeddings, labels = extract_embeddings(model, loader, device)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    from utils.visualization import plot_tsne
    fig = plot_tsne(embeddings, labels,
                    num_classes=config["num_grades"],
                    save_path=str(output_dir / "tsne_embeddings.png"))
    logger.info("t-SNE plot saved.")
    return embeddings, labels


def run_shap_analysis(model, loader, device, output_dir, n_background=50, n_explain=20):
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP analysis. Install with: pip install shap")
        return

    import matplotlib.pyplot as plt

    model.eval()
    background_images, explain_images, explain_grades = [], [], []

    for batch in loader:
        images = batch["image"]
        grades = batch["grade"]
        if len(background_images) < n_background:
            background_images.append(images)
        if len(explain_images) < n_explain:
            explain_images.append(images)
            explain_grades.extend(grades.numpy().tolist())
        if len(background_images) >= n_background and len(explain_images) >= n_explain:
            break

    background_tensor = torch.cat(background_images, dim=0)[:n_background].to(device)
    explain_tensor = torch.cat(explain_images, dim=0)[:n_explain].to(device)

    def model_predict(x):
        with torch.no_grad():
            out = model(torch.tensor(x, dtype=torch.float32).to(device))
            return torch.softmax(out["grade_logits"], dim=1).cpu().numpy()

    explainer = shap.GradientExplainer(
        (model, model.layer4),
        background_tensor,
    )
    shap_values = explainer.shap_values(explain_tensor)

    shap_output_dir = output_dir / "shap"
    shap_output_dir.mkdir(exist_ok=True)

    for grade_idx in range(config["num_grades"]):
        grade_mask = np.array(explain_grades[:n_explain]) == grade_idx
        if grade_mask.sum() == 0:
            continue

        grade_shap = np.array(shap_values[grade_idx])[grade_mask]
        mean_shap = np.abs(grade_shap).mean(axis=(0, 1))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(mean_shap)), mean_shap)
        ax.set_xlabel("Feature Channel")
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title(GRADE_NAMES[grade_idx])
        plt.tight_layout()
        plt.savefig(str(shap_output_dir / f"shap_grade_{grade_idx}.png"), dpi=150)
        plt.close()

    logger.info(f"SHAP analysis saved to {shap_output_dir}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ddr":
        samples = load_ddr_samples(args.data_root, args.lesion_annotation_dir)
    else:
        samples = load_idrid_samples(args.data_root)
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}")

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

    logger.info("Running Grad-CAM analysis...")
    run_gradcam_analysis(model, samples, device, output_dir, n_per_grade=args.n_per_grade)

    logger.info("Running segmentation visualization...")
    run_segmentation_visualization(model, samples, device, output_dir)

    logger.info("Running t-SNE analysis...")
    run_tsne_analysis(model, loader, device, output_dir)

    logger.info("Running SHAP analysis...")
    run_shap_analysis(model, loader, device, output_dir)

    logger.info(f"\nAll interpretability outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
