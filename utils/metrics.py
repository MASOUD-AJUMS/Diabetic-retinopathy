import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
from scipy.stats import bootstrap as scipy_bootstrap
import warnings

LESION_TYPES = ["MA", "HE", "EX", "SE"]
NUM_GRADES = 6
EPS = 1e-7


def dice_coefficient(pred_mask, gt_mask, threshold=0.5):
    pred = (pred_mask >= threshold).astype(np.float32)
    gt = gt_mask.astype(np.float32)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union < EPS:
        return 1.0 if pred.sum() < EPS else 0.0
    return float(2.0 * intersection / (union + EPS))


def iou_score(pred_mask, gt_mask, threshold=0.5):
    pred = (pred_mask >= threshold).astype(np.float32)
    gt = gt_mask.astype(np.float32)
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union < EPS:
        return 1.0 if pred.sum() < EPS else 0.0
    return float(intersection / (union + EPS))


def compute_aupr(pred_prob, gt_mask):
    gt_flat = gt_mask.flatten().astype(int)
    pred_flat = pred_prob.flatten()
    if gt_flat.sum() == 0:
        return float("nan")
    return average_precision_score(gt_flat, pred_flat)


def compute_segmentation_metrics(all_preds, all_targets, thresholds=None):
    if thresholds is None:
        thresholds = [0.5] * len(LESION_TYPES)

    results = {}
    for i, lesion in enumerate(LESION_TYPES):
        preds = np.concatenate([p[i] for p in all_preds], axis=0)
        targets = np.concatenate([t[i] for t in all_targets], axis=0)
        thr = thresholds[i]

        dice_scores = [dice_coefficient(preds[j], targets[j], thr) for j in range(len(preds))]
        iou_scores = [iou_score(preds[j], targets[j], thr) for j in range(len(preds))]
        aupr_scores = [compute_aupr(preds[j], targets[j]) for j in range(len(preds))]

        aupr_valid = [v for v in aupr_scores if not np.isnan(v)]
        results[lesion] = {
            "dice": np.mean(dice_scores),
            "iou": np.mean(iou_scores),
            "aupr": np.mean(aupr_valid) if aupr_valid else float("nan"),
        }

    mean_dice = np.mean([results[l]["dice"] for l in LESION_TYPES])
    mean_iou = np.mean([results[l]["iou"] for l in LESION_TYPES])
    mean_aupr = np.nanmean([results[l]["aupr"] for l in LESION_TYPES])
    results["mean"] = {"dice": mean_dice, "iou": mean_iou, "aupr": mean_aupr}
    return results


def find_optimal_thresholds(all_preds, all_targets, threshold_range=None):
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.9, 0.05)

    optimal_thresholds = []
    for i, lesion in enumerate(LESION_TYPES):
        preds = np.concatenate([p[i] for p in all_preds], axis=0)
        targets = np.concatenate([t[i] for t in all_targets], axis=0)

        best_thr, best_dice = 0.5, -1
        for thr in threshold_range:
            dice_scores = [dice_coefficient(preds[j], targets[j], thr) for j in range(len(preds))]
            mean_d = np.mean(dice_scores)
            if mean_d > best_dice:
                best_dice = mean_d
                best_thr = thr
        optimal_thresholds.append(best_thr)
    return optimal_thresholds


def compute_grading_metrics(all_preds, all_targets, num_classes=NUM_GRADES):
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    valid = targets >= 0
    preds, targets = preds[valid], targets[valid]

    acc = accuracy_score(targets, preds)
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    try:
        targets_onehot = np.eye(num_classes)[targets]
        macro_auc = roc_auc_score(targets_onehot, np.eye(num_classes)[preds],
                                   multi_class="ovr", average="macro")
    except ValueError:
        macro_auc = float("nan")

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    per_class_sens, per_class_spec = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / max(tp + fn, EPS)
        spec = tn / max(tn + fp, EPS)
        per_class_sens.append(sens)
        per_class_spec.append(spec)

    referable_mask = targets >= 2
    vt_mask = targets >= 3
    pred_referable = preds >= 2
    pred_vt = preds >= 3

    ref_sens = (pred_referable & referable_mask).sum() / max(referable_mask.sum(), 1)
    ref_spec = (~pred_referable & ~referable_mask).sum() / max((~referable_mask).sum(), 1)
    vt_sens = (pred_vt & vt_mask).sum() / max(vt_mask.sum(), 1)
    vt_spec = (~pred_vt & ~vt_mask).sum() / max((~vt_mask).sum(), 1)

    return {
        "accuracy": acc,
        "qwk": qwk,
        "macro_f1": f1,
        "macro_auc": macro_auc,
        "confusion_matrix": cm,
        "per_class_sensitivity": per_class_sens,
        "per_class_specificity": per_class_spec,
        "referable_dr": {"sensitivity": ref_sens, "specificity": ref_spec},
        "vision_threatening_dr": {"sensitivity": vt_sens, "specificity": vt_spec},
    }


def bootstrap_ci(metric_fn, data, n_resamples=1000, confidence_level=0.95, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data))
    scores = []
    for _ in range(n_resamples):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        boot_data = [data[i] for i in boot_idx]
        try:
            scores.append(metric_fn(boot_data))
        except Exception:
            pass
    if not scores:
        return float("nan"), float("nan")
    alpha = (1 - confidence_level) / 2
    return float(np.percentile(scores, 100 * alpha)), float(np.percentile(scores, 100 * (1 - alpha)))


def compute_ece(probs, labels, n_bins=10):
    probs = np.array(probs)
    labels = np.array(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = correct[in_bin].mean()
            ece += in_bin.sum() / len(labels) * abs(avg_conf - avg_acc)
    return ece


@torch.no_grad()
def run_inference(model, loader, device, seg_thresholds=None):
    model.eval()
    seg_preds, seg_targets = [], []
    grade_preds, grade_probs, grade_targets = [], [], []

    if seg_thresholds is None:
        seg_thresholds = [0.5] * 4

    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)

        seg_logits = outputs["seg_logits"]
        seg_prob = torch.sigmoid(seg_logits).cpu().numpy()
        for i in range(len(images)):
            if batch["has_lesion_annotation"][i]:
                seg_preds.append(seg_prob[i])
                seg_targets.append(batch["masks"][i].numpy())

        grade_logit = outputs["grade_logits"]
        prob = torch.softmax(grade_logit, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        grade_probs.extend(prob.tolist())
        grade_preds.extend(pred.tolist())
        grade_targets.extend(batch["grade"].numpy().tolist())

    results = {}
    if seg_preds:
        results["segmentation"] = compute_segmentation_metrics(
            seg_preds, seg_targets, seg_thresholds
        )
    results["grading"] = compute_grading_metrics(grade_preds, grade_targets)
    results["calibration"] = {"ece": compute_ece(np.array(grade_probs), np.array(grade_targets))}
    results["raw"] = {
        "seg_preds": seg_preds,
        "seg_targets": seg_targets,
        "grade_preds": grade_preds,
        "grade_probs": grade_probs,
        "grade_targets": grade_targets,
    }
    return results
