from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES: Sequence[str] = ("bcc", "benign", "melanoma", "scc")


class SkinLesionClassifier(nn.Module):
    """ResNet50 transfer-learning classifier used by this project.

    Torchvision's ResNet forward pass already performs global average pooling and
    flattening before the final fully connected layer. Replacing ``fc`` with
    ``Identity`` therefore returns a [batch, 2048] feature tensor; no additional
    AdaptiveAvgPool2d step should be applied after the backbone.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features)


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(
    checkpoint_path: str | Path,
    *,
    num_classes: int = len(CLASS_NAMES),
    device: str | torch.device = "cpu",
) -> SkinLesionClassifier:
    model = SkinLesionClassifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both a raw state_dict and a training checkpoint containing one.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
