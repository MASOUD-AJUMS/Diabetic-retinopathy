import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import cv2

LESION_TYPES = ["MA", "HE", "EX", "SE"]
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR", "Ungradable"]
LESION_COLORS = ["red", "blue", "yellow", "green"]


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return img.transpose(1, 2, 0)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image_tensor, target_class=None):
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
        output = self.model(image_tensor)
        grade_logits = output["grade_logits"]

        if target_class is None:
            target_class = grade_logits.argmax(dim=1).item()

        self.model.zero_grad()
        grade_logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, target_class


def plot_segmentation_results(image, pred_masks, gt_masks=None, thresholds=None, save_path=None):
    if thresholds is None:
        thresholds = [0.5] * len(LESION_TYPES)

    n_cols = len(LESION_TYPES) + 1
    n_rows = 2 if gt_masks is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    img_disp = denormalize(image) if isinstance(image, torch.Tensor) else image

    axes[0, 0].imshow(img_disp)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    for i, (lesion, color) in enumerate(zip(LESION_TYPES, LESION_COLORS)):
        pred = (pred_masks[i] >= thresholds[i]).astype(np.uint8)
        overlay = img_disp.copy()
        mask_rgb = np.zeros_like(overlay)
        c = mcolors.to_rgb(color)
        mask_rgb[pred == 1] = [int(255 * x) for x in c]
        overlay = cv2.addWeighted(overlay, 0.7, mask_rgb.astype(np.float32) / 255, 0.3, 0)
        axes[0, i + 1].imshow(overlay)
        axes[0, i + 1].set_title(f"Pred {lesion}")
        axes[0, i + 1].axis("off")

    if gt_masks is not None:
        axes[1, 0].imshow(img_disp)
        axes[1, 0].set_title("Input Image")
        axes[1, 0].axis("off")
        for i, (lesion, color) in enumerate(zip(LESION_TYPES, LESION_COLORS)):
            gt = gt_masks[i].astype(np.uint8)
            overlay = img_disp.copy()
            mask_rgb = np.zeros_like(overlay)
            c = mcolors.to_rgb(color)
            mask_rgb[gt == 1] = [int(255 * x) for x in c]
            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb.astype(np.float32) / 255, 0.3, 0)
            axes[1, i + 1].imshow(overlay)
            axes[1, i + 1].set_title(f"GT {lesion}")
            axes[1, i + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_gradcam_panel(images, cams, grade_preds, grade_targets, save_path=None):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        img = denormalize(images[i]) if isinstance(images[i], torch.Tensor) else images[i]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"GT: {GRADE_NAMES[grade_targets[i]]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(img)
        axes[1, i].imshow(cams[i], cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"Pred: {GRADE_NAMES[grade_preds[i]]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    n = cm.shape[0]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    labels = GRADE_NAMES[:n]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(all_probs, all_targets, num_classes=6, title="ROC Curves", save_path=None):
    from sklearn.metrics import roc_curve, auc
    probs = np.array(all_probs)
    targets = np.array(all_targets)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        binary_targets = (targets == c).astype(int)
        if binary_targets.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_targets, probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[c], lw=1.5,
                label=f"{GRADE_NAMES[c]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tsne(embeddings, labels, num_classes=6, save_path=None):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    proj = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[c]], label=GRADE_NAMES[c],
                   s=10, alpha=0.7)

    ax.legend(markerscale=2, fontsize=9)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reliability_diagram(probs, labels, n_bins=10, save_path=None):
    probs = np.array(probs)
    labels = np.array(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_mids = (bin_lowers + bin_uppers) / 2

    avg_confs, avg_accs, counts = [], [], []
    for lo, hi in zip(bin_lowers, bin_uppers):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            avg_confs.append(confidences[mask].mean())
            avg_accs.append(correct[mask].mean())
            counts.append(mask.sum())
        else:
            avg_confs.append((lo + hi) / 2)
            avg_accs.append(0)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_mids, avg_accs, width=0.09, alpha=0.7, color="steelblue", label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
