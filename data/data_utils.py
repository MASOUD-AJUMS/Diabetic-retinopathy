import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import imagehash
from PIL import Image
from collections import defaultdict


LESION_TYPES = ["MA", "HE", "EX", "SE"]


def load_ddr_samples(ddr_root, lesion_annotation_dir=None):
    ddr_root = Path(ddr_root)
    samples = []

    for grade in range(6):
        grade_dir = ddr_root / "grading" / str(grade)
        if not grade_dir.exists():
            continue
        for img_path in sorted(grade_dir.glob("*.jpg")) + sorted(grade_dir.glob("*.png")):
            sample = {
                "image_path": img_path,
                "grade": grade,
                "has_lesion_annotation": False,
            }
            samples.append(sample)

    if lesion_annotation_dir:
        lesion_dir = Path(lesion_annotation_dir)
        annotated_ids = set()
        for lesion in LESION_TYPES:
            mask_dir = lesion_dir / lesion
            if mask_dir.exists():
                for mask_path in mask_dir.glob("*"):
                    annotated_ids.add(mask_path.stem)

        for s in samples:
            stem = Path(s["image_path"]).stem
            if stem in annotated_ids:
                s["has_lesion_annotation"] = True
                for lesion in LESION_TYPES:
                    for ext in [".png", ".jpg", ".tif"]:
                        mp = lesion_dir / lesion / (stem + ext)
                        if mp.exists():
                            s[f"mask_{lesion}"] = mp

        bbox_file = lesion_dir / "bboxes.json"
        if bbox_file.exists():
            with open(bbox_file) as f:
                bbox_data = json.load(f)
            stem_to_bboxes = defaultdict(list)
            for entry in bbox_data:
                stem_to_bboxes[entry["image_id"]].append(entry)
            for s in samples:
                stem = Path(s["image_path"]).stem
                if stem in stem_to_bboxes:
                    s["bboxes"] = stem_to_bboxes[stem]

    return samples


def load_idrid_samples(idrid_root):
    idrid_root = Path(idrid_root)
    samples = []

    seg_image_dir = idrid_root / "B. Disease Grading" / "1. Original Images" / "a. Training Set"
    grade_file = idrid_root / "B. Disease Grading" / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"

    id_to_grade = {}
    if grade_file.exists():
        import csv
        with open(grade_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["Image name"].strip()
                grade = int(row["Retinopathy grade"].strip())
                id_to_grade[img_id] = grade

    seg_mask_root = idrid_root / "A. Segmentation" / "2. All Segmentation Groundtruths" / "a. Training Set"

    for img_id, grade in id_to_grade.items():
        img_path = None
        for ext in [".jpg", ".JPG", ".png"]:
            candidate = seg_image_dir / (img_id + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        sample = {
            "image_path": img_path,
            "grade": grade,
            "has_lesion_annotation": False,
        }

        lesion_map = {
            "MA": "1. Microaneurysms",
            "HE": "2. Haemorrhages",
            "EX": "3. Hard Exudates",
            "SE": "4. Soft Exudates",
        }
        found_any = False
        for lesion, folder in lesion_map.items():
            mask_dir = seg_mask_root / folder
            if mask_dir.exists():
                for ext in [".tif", ".png", ".jpg"]:
                    mp = mask_dir / (img_id + "_" + lesion + ext)
                    if mp.exists():
                        sample[f"mask_{lesion}"] = mp
                        found_any = True
                        break
        if found_any:
            sample["has_lesion_annotation"] = True

        samples.append(sample)

    return samples


def compute_perceptual_hash(image_path, hash_size=8):
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None


def deduplicate_and_split(samples, n_splits=5, seed=42, hamming_threshold=5):
    grades = [s["grade"] for s in samples]

    hashes = []
    for s in samples:
        h = compute_perceptual_hash(s["image_path"])
        hashes.append(h)

    duplicate_groups = defaultdict(list)
    assigned = [False] * len(samples)
    for i in range(len(samples)):
        if assigned[i] or hashes[i] is None:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(samples)):
            if assigned[j] or hashes[j] is None:
                continue
            if hashes[i] - hashes[j] <= hamming_threshold:
                group.append(j)
                assigned[j] = True
        for idx in group:
            duplicate_groups[group[0]].append(idx)

    group_representatives = list(duplicate_groups.keys())
    rep_grades = [grades[r] for r in group_representatives]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_assignments = [-1] * len(samples)

    for fold_idx, (_, val_idx) in enumerate(skf.split(group_representatives, rep_grades)):
        for ri in val_idx:
            rep = group_representatives[ri]
            for member in duplicate_groups[rep]:
                fold_assignments[member] = fold_idx

    folds = []
    for fold in range(n_splits):
        val_samples = [samples[i] for i in range(len(samples)) if fold_assignments[i] == fold]
        train_samples = [samples[i] for i in range(len(samples)) if fold_assignments[i] != fold]
        folds.append({"train": train_samples, "val": val_samples})

    return folds


def compute_class_weights(samples, num_classes=6):
    counts = np.zeros(num_classes)
    for s in samples:
        g = s["grade"]
        if 0 <= g < num_classes:
            counts[g] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights
