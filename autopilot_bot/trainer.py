from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from autopilot_bot.model import TinyCNN, export_onnx, export_torchscript


class FrameActionDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, action_list: Sequence[str], grayscale: bool = False) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.action_to_idx = {a: i for i, a in enumerate(action_list)}
        self.grayscale = grayscale
        self.samples: List[Tuple[Path, int]] = []

        labels_file = self.dataset_dir / "labels.csv"
        if not labels_file.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_file}")

        with labels_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                action = row["action"]
                if action not in self.action_to_idx:
                    continue
                self.samples.append((self.dataset_dir / "frames" / row["frame"], self.action_to_idx[action]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        frame_path, label = self.samples[index]
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image), torch.tensor(label, dtype=torch.long)


@dataclass
class TrainResult:
    best_val_acc: float
    best_model_path: str


def train_model(
    dataset_dir: str,
    action_list: Sequence[str],
    output_path: str,
    onnx_path: str,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    grayscale: bool = False,
) -> TrainResult:
    dataset = FrameActionDataset(dataset_dir, action_list, grayscale=grayscale)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    in_channels = 1 if grayscale else 3
    model = TinyCNN(num_classes=len(action_list), in_channels=in_channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = correct / max(1, total)
        if acc >= best_acc:
            best_acc = acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    export_torchscript(model.cpu(), output_path, input_shape=(in_channels, 96, 96))
    export_onnx(model.cpu(), onnx_path, input_shape=(in_channels, 96, 96))

    return TrainResult(best_val_acc=best_acc, best_model_path=output_path)


def create_tiny_demo_dataset(dataset_dir: str | Path, action_list: Sequence[str], n_per_action: int = 8) -> None:
    dataset_dir = Path(dataset_dir)
    frames_dir = dataset_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels_path = dataset_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "action"])
        writer.writeheader()

        idx = 0
        for action in action_list:
            for _ in range(n_per_action):
                image = np.zeros((96, 96, 3), dtype=np.uint8)
                cv2.putText(
                    image,
                    action[:8],
                    (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                noise = np.random.randint(0, 30, size=image.shape, dtype=np.uint8)
                image = cv2.add(image, noise)
                frame_name = f"frame_{idx:05d}.png"
                cv2.imwrite(str(frames_dir / frame_name), image)
                writer.writerow({"frame": frame_name, "action": action})
                idx += 1
