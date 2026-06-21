import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json


LESION_TYPES = ["MA", "HE", "EX", "SE"]
NUM_GRADES = 6
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=512):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
        A.GaussNoise(var_limit=(5, 15), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']))


def get_val_transforms(image_size=512):
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']))


def preprocess_fundus(image, image_size=512):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        image = image[y:y+h, x:x+w]

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class DRDataset(Dataset):
    def __init__(self, samples, image_size=512, transform=None, mode="train"):
        self.samples = samples
        self.image_size = image_size
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        grade = sample.get("grade", -1)
        has_lesion_annotation = sample.get("has_lesion_annotation", False)

        image = cv2.imread(str(image_path))
        image = preprocess_fundus(image, self.image_size)

        masks = np.zeros((len(LESION_TYPES), self.image_size, self.image_size), dtype=np.float32)
        bboxes = []
        bbox_labels = []

        if has_lesion_annotation:
            for i, lesion in enumerate(LESION_TYPES):
                mask_path = sample.get(f"mask_{lesion}")
                if mask_path and Path(mask_path).exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)
                    masks[i] = (mask > 127).astype(np.float32)

            raw_bboxes = sample.get("bboxes", [])
            for bb in raw_bboxes:
                bboxes.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]])
                bbox_labels.append(bb["label"])

        if self.transform:
            masks_list = [masks[i] for i in range(len(LESION_TYPES))]
            valid_bboxes = []
            valid_labels = []
            if bboxes:
                for bb, lbl in zip(bboxes, bbox_labels):
                    x1, y1, x2, y2 = bb
                    if x2 > x1 and y2 > y1:
                        valid_bboxes.append([
                            max(0, x1), max(0, y1),
                            min(self.image_size, x2), min(self.image_size, y2)
                        ])
                        valid_labels.append(lbl)

            augmented = self.transform(
                image=image,
                masks=masks_list,
                bboxes=valid_bboxes if valid_bboxes else [[0, 0, 1, 1]],
                bbox_labels=valid_labels if valid_labels else [-1],
            )
            image = augmented["image"]
            aug_masks = augmented["masks"]
            masks = torch.stack([torch.tensor(m) for m in aug_masks], dim=0)
            aug_bboxes = list(augmented["bboxes"])
            aug_labels = list(augmented["bbox_labels"])
            if aug_labels == [-1]:
                aug_bboxes, aug_labels = [], []
        else:
            image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            masks = torch.tensor(masks, dtype=torch.float32)
            aug_bboxes, aug_labels = bboxes, bbox_labels

        det_target = {
            "boxes": torch.tensor(aug_bboxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(aug_labels, dtype=torch.long).reshape(-1),
        }

        return {
            "image": image,
            "grade": torch.tensor(grade, dtype=torch.long),
            "masks": masks,
            "det_target": det_target,
            "has_lesion_annotation": has_lesion_annotation,
            "image_path": str(image_path),
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    grades = torch.stack([b["grade"] for b in batch])
    masks = torch.stack([b["masks"] for b in batch])
    det_targets = [b["det_target"] for b in batch]
    has_annotations = [b["has_lesion_annotation"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    return {
        "image": images,
        "grade": grades,
        "masks": masks,
        "det_targets": det_targets,
        "has_lesion_annotation": has_annotations,
        "image_paths": image_paths,
    }
