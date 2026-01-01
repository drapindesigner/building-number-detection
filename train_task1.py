# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Mohammad Saif Ul Haq
# Last Modified: 2025-10-03

"""Training entry point for Task 1: building number detection."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for Task 1 (building number localisation)")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Building Numbers.v2i.yolov8/data.yaml"),
        help="Path to the dataset YAML describing building-number bounding boxes",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device for training (e.g. '0', 'cpu')")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/task1_detect"),
        help="Directory where Ultralytics will write training artefacts",
    )
    parser.add_argument("--name", type=str, default="task1_detector", help="Experiment name inside the project directory")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epochs)")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the project/name directory")
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("data/detector/best.pt"),
        help="Where to copy the best model weights after training",
    )
    parser.add_argument("--skip-export", action="store_true", help="Skip copying weights to the assignment data folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
    )
    if args.resume:
        train_kwargs["resume"] = True
    model.train(**train_kwargs)

    if args.skip_export:
        return

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"Best weights not found at {best_weights}; skipping export.")
        return

    args.export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.export)
    print(f"Copied best checkpoint to {args.export}")


if __name__ == "__main__":
    main()
