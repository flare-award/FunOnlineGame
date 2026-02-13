from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import torch

from autopilot_bot.capture import CaptureRegion, ScreenCapture
from autopilot_bot.config import AppConfig, load_config, save_config
from autopilot_bot.controller import InputController
from autopilot_bot.gui import AppGUI
from autopilot_bot.model import load_model
from autopilot_bot.trainer import create_tiny_demo_dataset, train_model
from autopilot_bot.utils import is_window_active, setup_logger


class AutopilotRuntime:
    def __init__(self, config: AppConfig, gui: AppGUI | None = None) -> None:
        self.config = config
        self.gui = gui
        self.logger = setup_logger(config.log_file)
        self.enabled = False
        self.stop_event = threading.Event()
        self.controller = InputController(
            action_keys=config.action_keys,
            hold_ms=config.key_hold_ms,
            anti_spam_ms=config.anti_spam_ms,
        )
        self.model = None

    def set_enabled(self, state: bool) -> None:
        self.enabled = state
        self.logger.info("Autopilot enabled=%s", state)

    def update_keys(self, mapping: dict) -> None:
        self.config.action_keys.update(mapping)
        self.controller.action_keys = self.config.action_keys

    def ensure_model(self) -> None:
        model_path = Path(self.config.model_path)
        if model_path.exists():
            return

        self.logger.info("Model not found. Creating tiny demo dataset and training quick model...")
        create_tiny_demo_dataset(self.config.demo_dataset_dir, self.config.action_list)
        train_model(
            dataset_dir=self.config.demo_dataset_dir,
            action_list=self.config.action_list,
            output_path=self.config.model_path,
            onnx_path=self.config.onnx_path,
            epochs=1,
            batch_size=16,
            grayscale=self.config.grayscale,
        )

    def loop(self) -> None:
        self.ensure_model()
        in_channels = 1 if self.config.grayscale else 3
        self.model = load_model(self.config.model_path, len(self.config.action_list), in_channels=in_channels)

        region = CaptureRegion(
            left=self.config.capture_region.left,
            top=self.config.capture_region.top,
            width=self.config.capture_region.width,
            height=self.config.capture_region.height,
        )

        interval = 1.0 / max(1, self.config.inference_fps)

        with ScreenCapture(region, tuple(self.config.image_size), self.config.grayscale) as cap:
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                try:
                    frame = cap.grab_preprocessed()
                    x = np.transpose(frame, (2, 0, 1))
                    x = torch.tensor(x).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.model(x)
                        probs = torch.softmax(logits, dim=1)[0]
                        conf, idx = torch.max(probs, dim=0)
                    action = self.config.action_list[int(idx)]
                    confidence = float(conf)

                    allow_send = (
                        self.enabled
                        and not self.config.only_demo
                        and self.config.inference_mode
                        and is_window_active(self.config.game_window_title_contains)
                    )

                    if self.config.safe_mode and confidence < self.config.confidence_threshold:
                        allow_send = False

                    if allow_send:
                        self.controller.send_action(action)

                    msg = f"action={action}, conf={confidence:.2f}, send={allow_send}"
                    self.logger.info(msg)
                    if self.gui:
                        self.gui.push_log(msg)
                except Exception as exc:
                    self.logger.exception("Runtime loop error: %s", exc)
                    if self.gui:
                        self.gui.push_log(f"ERROR: {exc}")
                    time.sleep(0.5)

                dt = time.perf_counter() - t0
                if dt < interval:
                    time.sleep(interval - dt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local game autopilot")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--no-gui", action="store_true", help="Run without Tkinter UI")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (no key presses)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.demo:
        config.only_demo = True

    runtime = AutopilotRuntime(config)

    if args.no_gui:
        runtime.enabled = True
        runtime.loop()
        return

    gui = AppGUI(on_toggle=runtime.set_enabled, on_save_keys=runtime.update_keys)
    runtime.gui = gui
    worker = threading.Thread(target=runtime.loop, daemon=True)
    worker.start()
    gui.run()

    runtime.stop_event.set()
    save_config(config, args.config)


if __name__ == "__main__":
    main()
