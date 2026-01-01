

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

"""Run Task 1: Detect and localize building numbers in images."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from perception import load_assignment_config, load_config_overrides_from_file
from perception.detection import YoloDetector
from perception.io_utils import collect_images, ensure_dir, load_image, save_image
from perception.utils import clamp_box, crop, expand_box


# Cache detector instances to avoid reloading model weights
_DETECTOR_CACHE: Dict[Path, YoloDetector] = {}


def _extract_index(name: str) -> str:
    """
    Extract numeric index from filename (e.g., 'img3' -> '3').
    
    Args:
        name: Filename stem (without extension)
        
    Returns:
        Numeric index as string, or original name if no number found
    """
    match = re.search(r"(\d+)", name)
    return match.group(1) if match else name


def _get_detector(weights_path: Path, config) -> YoloDetector:
    """
    Get or create a cached YOLO detector instance.
    
    Args:
        weights_path: Path to YOLO model weights
        config: Detector configuration
        
    Returns:
        YoloDetector instance
    """
    key = weights_path.resolve()
    detector = _DETECTOR_CACHE.get(key)
    if detector is None:
        detector = YoloDetector(config)
        _DETECTOR_CACHE[key] = detector
    return detector


def run_task1(
    input_path: Union[str, Path],
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, Optional[Dict[str, Union[float, tuple]]]]:
    """
    Task 1: Detect and localize building numbers in images.
    
    This function processes all images in the input directory, detects building
    numbers using YOLO, and saves cropped regions containing the detected numbers.
    
    Args:
        input_path: Path to directory containing input images
        config: Optional configuration dictionary to override defaults
        
    Returns:
        Dictionary mapping image names to detection results (or None for negatives)
        
    Example:
        >>> results = run_task1("validation/task1")
        >>> print(results["img1.jpg"])
        {'confidence': 0.95, 'box': (100, 150, 300, 250)}
    """
    config_overrides = dict(config or {})
    assignment_cfg = load_assignment_config(config_overrides, base_path=Path.cwd())
    output_dir = ensure_dir(assignment_cfg.task_output_dir("task1"))

    detector = _get_detector(assignment_cfg.detection.weights_path, assignment_cfg.detection)

    image_paths = collect_images(Path(input_path))
    results: Dict[str, Optional[Dict[str, Union[float, tuple]]]] = {}

    for image_path in image_paths:
        image = load_image(image_path)
        detection = detector.best_detection(image)
        index = _extract_index(image_path.stem)
        output_path = output_dir / f"bn{index}.png"

        if detection is None:
            # Negative frame: remove any stale output
            output_path.unlink(missing_ok=True)
            results[image_path.name] = None
            continue

        height, width = image.shape[:2]
        box = clamp_box(detection.xyxy, width, height)
        if assignment_cfg.detection.crop_expand != 1.0:
            box = expand_box(box, (height, width), assignment_cfg.detection.crop_expand)
        # Only a single crop is allowed per image
        crop_img = crop(image, box)
        save_image(output_path, crop_img)

        results[image_path.name] = {
            "confidence": detection.confidence,
            "box": box.as_tuple(),
        }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 1 YOLO detector")
    parser.add_argument("input", type=str, help="Directory or image path")
    parser.add_argument("--config", type=str, default=None, help="Optional config overrides file")
    args = parser.parse_args()

    overrides = None
    if args.config:
        try:
            overrides = load_config_overrides_from_file(args.config)
        except FileNotFoundError:
            print(f"Config overrides not found: {args.config}")
        except Exception as exc:
            print(f"Failed to parse overrides {args.config}: {exc}")

    summary = run_task1(args.input, overrides)
    for name, info in summary.items():
        if info:
            print(f"{name}: confidence={info['confidence']:.3f}")
        else:
            print(f"{name}: negative")
