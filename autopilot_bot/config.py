from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Region:
    left: int = 0
    top: int = 0
    width: int = 800
    height: int = 600


@dataclass
class AppConfig:
    capture_region: Region = field(default_factory=Region)
    capture_fps: int = 10
    inference_fps: int = 10
    image_size: List[int] = field(default_factory=lambda: [96, 96])
    grayscale: bool = False
    action_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "jump": "space",
            "interact": "e",
            "idle": "",
        }
    )
    action_list: List[str] = field(
        default_factory=lambda: [
            "idle",
            "forward",
            "backward",
            "left",
            "right",
            "jump",
            "interact",
        ]
    )
    key_hold_ms: int = 70
    anti_spam_ms: int = 120
    confidence_threshold: float = 0.65
    safe_mode: bool = True
    human_in_the_loop: bool = True
    only_demo: bool = True
    model_path: str = "models/tinycnn_scripted.pt"
    onnx_path: str = "models/tinycnn.onnx"
    log_file: str = "logs/autopilot.log"
    game_window_title_contains: Optional[str] = None
    training_mode: bool = False
    inference_mode: bool = True
    demo_dataset_dir: str = "data/demo_dataset"


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)

    region_raw = raw.get("capture_region", {})
    region = Region(
        left=region_raw.get("left", 0),
        top=region_raw.get("top", 0),
        width=region_raw.get("width", 800),
        height=region_raw.get("height", 600),
    )

    return AppConfig(
        capture_region=region,
        capture_fps=raw.get("capture_fps", 10),
        inference_fps=raw.get("inference_fps", 10),
        image_size=raw.get("image_size", [96, 96]),
        grayscale=raw.get("grayscale", False),
        action_keys=raw.get("action_keys", AppConfig().action_keys),
        action_list=raw.get("action_list", AppConfig().action_list),
        key_hold_ms=raw.get("key_hold_ms", 70),
        anti_spam_ms=raw.get("anti_spam_ms", 120),
        confidence_threshold=raw.get("confidence_threshold", 0.65),
        safe_mode=raw.get("safe_mode", True),
        human_in_the_loop=raw.get("human_in_the_loop", True),
        only_demo=raw.get("only_demo", True),
        model_path=raw.get("model_path", "models/tinycnn_scripted.pt"),
        onnx_path=raw.get("onnx_path", "models/tinycnn.onnx"),
        log_file=raw.get("log_file", "logs/autopilot.log"),
        game_window_title_contains=raw.get("game_window_title_contains"),
        training_mode=raw.get("training_mode", False),
        inference_mode=raw.get("inference_mode", True),
        demo_dataset_dir=raw.get("demo_dataset_dir", "data/demo_dataset"),
    )


def save_config(config: AppConfig, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "capture_region": vars(config.capture_region),
        "capture_fps": config.capture_fps,
        "inference_fps": config.inference_fps,
        "image_size": config.image_size,
        "grayscale": config.grayscale,
        "action_keys": config.action_keys,
        "action_list": config.action_list,
        "key_hold_ms": config.key_hold_ms,
        "anti_spam_ms": config.anti_spam_ms,
        "confidence_threshold": config.confidence_threshold,
        "safe_mode": config.safe_mode,
        "human_in_the_loop": config.human_in_the_loop,
        "only_demo": config.only_demo,
        "model_path": config.model_path,
        "onnx_path": config.onnx_path,
        "log_file": config.log_file,
        "game_window_title_contains": config.game_window_title_contains,
        "training_mode": config.training_mode,
        "inference_mode": config.inference_mode,
        "demo_dataset_dir": config.demo_dataset_dir,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        else:
            json.dump(payload, f, indent=2, ensure_ascii=False)
