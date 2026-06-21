config = {
    "seed": 42,
    "image_size": 512,
    "num_grades": 6,
    "num_lesion_types": 4,
    "fpn_channels": 256,
    "dropout": 0.5,

    "base_lr": 1e-4,
    "weight_decay": 0.01,
    "layer_decay": 0.9,
    "grad_clip": 1.0,

    "seg_epochs": 100,
    "det_epochs": 100,
    "full_epochs": 100,
    "patience": 10,

    "batch_size_lesion": 4,
    "batch_size_full": 8,
    "num_workers": 4,

    "n_folds": 5,
    "n_seeds": 3,
    "hamming_threshold": 5,

    "seg_threshold_range_start": 0.1,
    "seg_threshold_range_end": 0.9,
    "seg_threshold_range_step": 0.05,

    "bootstrap_n_resamples": 1000,
    "bootstrap_confidence_level": 0.95,

    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "dice_smooth": 1e-5,

    "detection_nms_threshold": 0.5,
    "detection_score_threshold": 0.05,
    "detection_iou_threshold": 0.5,
}
