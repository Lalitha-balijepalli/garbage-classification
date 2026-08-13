"""
models.py
=========
Factory functions that build EfficientNet-B3, ResNet50, and MobileNetV3-Large
with ImageNet-pretrained backbones and a classifier head sized to the actual
number of dataset classes.

All three follow the same two-phase transfer-learning recipe:
  Phase 1 (head-only):  freeze the backbone, train only the new classifier
                         head. This adapts the head quickly without
                         destroying the pretrained features.
  Phase 2 (fine-tune):  unfreeze the last few backbone blocks and continue
                         training everything at a much lower learning rate,
                         so the model can specialise pretrained ImageNet
                         features toward garbage textures/shapes.
"""

import torch.nn as nn
from torchvision import models


def build_efficientnet_b3(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_resnet50(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_mobilenet_v3(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "efficientnet_b3": build_efficientnet_b3,
    "resnet50": build_resnet50,
    "mobilenet_v3": build_mobilenet_v3,
}


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(MODEL_BUILDERS)}.")
    return MODEL_BUILDERS[model_name](num_classes, freeze_backbone=freeze_backbone)


def unfreeze_for_finetuning(model: nn.Module, model_name: str, num_blocks: int = 2) -> nn.Module:
    """
    Unfreeze the last `num_blocks` feature blocks (plus the classifier head,
    which is already trainable) for phase-2 fine-tuning. Earlier layers stay
    frozen because they encode generic, low-level features (edges, textures)
    that transfer well regardless of the target domain.
    """
    if model_name in ("efficientnet_b3", "mobilenet_v3"):
        blocks = list(model.features.children())
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    elif model_name == "resnet50":
        for layer in [model.layer4, model.layer3][:num_blocks]:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")
    return model
