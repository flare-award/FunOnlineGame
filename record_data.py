from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import keyboard

from autopilot_bot.capture import CaptureRegion, ScreenCapture
from autopilot_bot.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record frames + labels during manual gameplay")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out", default="data/session_01")
    p.add_argument("--seconds", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "labels.csv"
    region = CaptureRegion(
        cfg.capture_region.left,
        cfg.capture_region.top,
        cfg.capture_region.width,
        cfg.capture_region.height,
    )

    interval = 1.0 / max(1, cfg.capture_fps)
    end_time = time.time() + args.seconds

    with labels_path.open("w", encoding="utf-8", newline="") as f, ScreenCapture(
        region, tuple(cfg.image_size), cfg.grayscale
    ) as cap:
        writer = csv.DictWriter(f, fieldnames=["frame", "action", "timestamp"])
        writer.writeheader()

        idx = 0
        while time.time() < end_time:
            t0 = time.perf_counter()
            frame = cap.grab_raw()
            frame = cv2.resize(frame, tuple(cfg.image_size))

            action = "idle"
            for a in cfg.action_list:
                key = cfg.action_keys.get(a)
                if key and keyboard.is_pressed(key):
                    action = a
                    break

            frame_name = f"frame_{idx:06d}.png"
            cv2.imwrite(str(frames_dir / frame_name), frame)
            writer.writerow({"frame": frame_name, "action": action, "timestamp": time.time()})
            idx += 1

            dt = time.perf_counter() - t0
            if dt < interval:
                time.sleep(interval - dt)


if __name__ == "__main__":
    main()
