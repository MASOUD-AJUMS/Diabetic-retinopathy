import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        B = pred.shape[0]
        pred_flat = pred.view(B, pred.shape[1], -1)
        target_flat = target.view(B, target.shape[1], -1)
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        target = target.float()
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha_t = torch.where(target == 1,
                              torch.ones_like(pred) * self.alpha,
                              torch.ones_like(pred) * (1 - self.alpha))
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return loss.mean()


class DetectionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reg_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.reg_weight = reg_weight

    def forward(self, det_cls, det_reg, det_centerness, det_targets, image_size=512):
        cls_losses = []
        for level_cls in det_cls:
            B, C, H, W = level_cls.shape
            dummy_target = torch.zeros_like(level_cls)
            cls_losses.append(self.focal(level_cls, dummy_target))
        cls_loss = sum(cls_losses) / len(cls_losses)
        return cls_loss


class ClassificationLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        weight = self.class_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weight, ignore_index=-1)


class HomoscedasticUncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    @property
    def task_weights(self):
        return torch.exp(-self.log_vars).detach()


class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.seg_loss_fn = SegmentationLoss()
        self.det_loss_fn = DetectionLoss()
        self.cls_loss_fn = ClassificationLoss(class_weights)
        self.uncertainty = HomoscedasticUncertaintyLoss(num_tasks=3)

    def forward(self, outputs, batch, stage="full"):
        grades = batch["grade"]
        masks = batch["masks"]
        has_annotation = batch["has_lesion_annotation"]
        det_targets = batch["det_targets"]

        cls_loss = self.cls_loss_fn(outputs["grade_logits"], grades)

        seg_loss = torch.tensor(0.0, device=grades.device)
        det_loss = torch.tensor(0.0, device=grades.device)

        annotated_idx = [i for i, h in enumerate(has_annotation) if h]
        if annotated_idx and stage in ["seg", "det", "full"]:
            idx = torch.tensor(annotated_idx, device=grades.device)
            ann_seg_pred = outputs["seg_logits"][idx]
            ann_seg_target = masks[idx]
            seg_loss = self.seg_loss_fn(ann_seg_pred, ann_seg_target)

            if stage in ["det", "full"]:
                ann_det_targets = [det_targets[i] for i in annotated_idx]
                det_loss = self.det_loss_fn(
                    outputs["det_cls"],
                    outputs["det_reg"],
                    outputs["det_centerness"],
                    ann_det_targets,
                )

        if stage == "full":
            total = self.uncertainty([seg_loss, det_loss, cls_loss])
        elif stage == "seg":
            total = seg_loss
        elif stage == "det":
            total = seg_loss + det_loss
        else:
            total = cls_loss

        return {
            "total": total,
            "seg": seg_loss.detach(),
            "det": det_loss.detach(),
            "cls": cls_loss.detach(),
        }
