import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import FeaturePyramidNetwork


NUM_LESION_TYPES = 4
NUM_GRADES = 6
FPN_CHANNELS = 256
ATTENTION_DIM = 256


class CrossTaskAttention(nn.Module):
    def __init__(self, embed_dim=ATTENTION_DIM):
        super().__init__()
        self.wq = nn.Conv2d(embed_dim, embed_dim, 1)
        self.wk = nn.Conv2d(embed_dim, embed_dim, 1)
        self.wv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.scale = embed_dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, query_feat, context_feat):
        B, C, H, W = query_feat.shape
        if context_feat.shape[2:] != (H, W):
            context_feat = F.interpolate(context_feat, size=(H, W), mode="bilinear", align_corners=False)

        Q = self.wq(query_feat).flatten(2).permute(0, 2, 1)
        K = self.wk(context_feat).flatten(2).permute(0, 2, 1)
        V = self.wv(context_feat).flatten(2).permute(0, 2, 1)

        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V).permute(0, 2, 1).reshape(B, C, H, W)

        return query_feat + self.gamma * out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=FPN_CHANNELS, num_classes=NUM_LESION_TYPES):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, fpn_feat):
        return self.decoder(fpn_feat)


class DetectionHead(nn.Module):
    def __init__(self, in_channels=FPN_CHANNELS, num_classes=NUM_LESION_TYPES):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness_head = nn.Conv2d(in_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        bias_init = float(-torch.log(torch.tensor(99.0)))
        nn.init.constant_(self.cls_head.bias, bias_init)

    def forward(self, fpn_feats):
        cls_preds, reg_preds, centerness_preds = [], [], []
        for feat in fpn_feats:
            x = self.shared(feat)
            cls_preds.append(self.cls_head(x))
            reg_preds.append(torch.exp(self.reg_head(x)))
            centerness_preds.append(self.centerness_head(x))
        return cls_preds, reg_preds, centerness_preds


class ClassificationHead(nn.Module):
    def __init__(self, in_channels=FPN_CHANNELS, num_classes=NUM_GRADES, dropout=0.5):
        super().__init__()
        self.attn_seg = CrossTaskAttention(in_channels)
        self.attn_det = CrossTaskAttention(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, grade_feat, seg_feat, det_feat):
        x = self.attn_seg(grade_feat, seg_feat)
        x = self.attn_det(x, det_feat)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class DRMultiTaskNet(nn.Module):
    def __init__(self, num_grades=NUM_GRADES, num_lesion_types=NUM_LESION_TYPES,
                 fpn_channels=FPN_CHANNELS, dropout=0.5):
        super().__init__()

        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        in_channels_dict = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(in_channels_dict.values()),
            out_channels=fpn_channels,
        )

        self.seg_head = SegmentationHead(fpn_channels, num_lesion_types)
        self.det_head = DetectionHead(fpn_channels, num_lesion_types)
        self.cls_head = ClassificationHead(fpn_channels, num_grades, dropout)

        self.seg_proj = nn.Conv2d(num_lesion_types, fpn_channels, 1)
        self.det_proj = nn.Conv2d(num_lesion_types, fpn_channels, 1)

    def extract_features(self, x):
        x = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        fpn_input = {"layer1": c1, "layer2": c2, "layer3": c3, "layer4": c4}
        fpn_out = self.fpn(fpn_input)
        return fpn_out

    def forward(self, x):
        fpn_out = self.extract_features(x)

        p2 = fpn_out["layer1"]
        p3 = fpn_out["layer2"]
        p4 = fpn_out["layer3"]
        p5 = fpn_out["layer4"]

        seg_logits = self.seg_head(p2)
        seg_logits_resized = F.interpolate(seg_logits, size=x.shape[2:], mode="bilinear", align_corners=False)

        det_cls, det_reg, det_centerness = self.det_head([p3, p4, p5])

        seg_feat = F.interpolate(torch.sigmoid(seg_logits), size=p5.shape[2:], mode="bilinear", align_corners=False)
        seg_ctx = self.seg_proj(seg_feat)

        det_feat = F.interpolate(torch.sigmoid(det_cls[0]), size=p5.shape[2:], mode="bilinear", align_corners=False)
        det_ctx = self.det_proj(det_feat)

        grade_logits = self.cls_head(p5, seg_ctx, det_ctx)

        return {
            "seg_logits": seg_logits_resized,
            "det_cls": det_cls,
            "det_reg": det_reg,
            "det_centerness": det_centerness,
            "grade_logits": grade_logits,
        }
