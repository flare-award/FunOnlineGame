from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def load_model(model_path: str | Path, num_classes: int, in_channels: int = 3, device: str = "cpu") -> nn.Module:
    model_path = Path(model_path)
    if model_path.suffix == ".pt":
        try:
            model = torch.jit.load(str(model_path), map_location=device)
            model.eval()
            return model
        except RuntimeError:
            model = TinyCNN(num_classes=num_classes, in_channels=in_channels)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model
    raise ValueError(f"Unsupported model format: {model_path.suffix}")


def export_torchscript(model: nn.Module, path: str | Path, input_shape: Tuple[int, int, int] = (3, 96, 96)) -> None:
    model.eval()
    dummy = torch.randn(1, *input_shape)
    scripted = torch.jit.trace(model, dummy)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))


def export_onnx(model: nn.Module, path: str | Path, input_shape: Tuple[int, int, int] = (3, 96, 96)) -> None:
    model.eval()
    dummy = torch.randn(1, *input_shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )
