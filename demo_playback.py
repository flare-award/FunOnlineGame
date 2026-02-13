from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
import numpy as np

from autopilot_bot.config import load_config
from autopilot_bot.model import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Playback recorded session with prediction overlay")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--dataset", default="data/demo_dataset")
    p.add_argument("--max-frames", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    in_channels = 1 if cfg.grayscale else 3
    model = load_model(cfg.model_path, len(cfg.action_list), in_channels=in_channels)
    model.eval()

    rows = []
    labels = Path(args.dataset) / "labels.csv"
    with labels.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows[: args.max_frames]:
        frame_path = Path(args.dataset) / "frames" / row["frame"]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        prep = cv2.resize(frame, tuple(cfg.image_size)).astype(np.float32) / 255.0
        if cfg.grayscale:
            prep = cv2.cvtColor(prep, cv2.COLOR_BGR2GRAY)
            prep = np.expand_dims(prep, axis=-1)

        x = torch.tensor(np.transpose(prep, (2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        pred = cfg.action_list[int(idx)]
        gt = row["action"]
        text = f"GT:{gt} Pred:{pred} P:{float(conf):.2f}"
        cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("demo_playback", frame)

        if cv2.waitKey(80) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
