from __future__ import annotations

import argparse

from autopilot_bot.config import load_config
from autopilot_bot.trainer import create_tiny_demo_dataset, train_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight action classifier")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--dataset", default=None, help="Path to dataset with frames/ + labels.csv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bootstrap-demo", action="store_true", help="Generate synthetic demo dataset first")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset = args.dataset or cfg.demo_dataset_dir
    if args.bootstrap_demo:
        create_tiny_demo_dataset(dataset, cfg.action_list)

    result = train_model(
        dataset_dir=dataset,
        action_list=cfg.action_list,
        output_path=cfg.model_path,
        onnx_path=cfg.onnx_path,
        epochs=args.epochs,
        grayscale=cfg.grayscale,
    )

    print(f"Training complete. Best val acc={result.best_val_acc:.3f}")
    print(f"Model saved: {result.best_model_path}")


if __name__ == "__main__":
    main()
