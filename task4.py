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

"""Run Task 4: Complete building number recognition pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

from perception import BuildingNumberPipeline, load_assignment_config, load_config_overrides_from_file
from perception.io_utils import collect_images, ensure_dir, load_image, save_text


def run_task4(
    input_path: Union[str, Path],
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, Optional[str]]:
    """
    Task 4: Complete building number recognition pipeline.
    
    This function processes input images through the complete pipeline:
    detection, segmentation, and recognition.
    
    Args:
        input_path: Path to directory containing input images
        config: Optional configuration dictionary to override defaults
        
    Returns:
        Dictionary mapping image names to recognized building numbers (or None for negatives)
        
    Example:
        >>> results = run_task4("validation/task1")
        >>> print(results["img1.jpg"])
        '314'
        >>> print(results["img5.jpg"])  # Negative image
        None
    """
    overrides = dict(config or {})
    assignment_cfg = load_assignment_config(overrides, base_path=Path.cwd())
    output_dir = ensure_dir(assignment_cfg.task_output_dir("task4"))

    pipeline = BuildingNumberPipeline(assignment_cfg)

    image_paths = collect_images(Path(input_path))
    results: Dict[str, Optional[str]] = {}

    for image_path in image_paths:
        image = load_image(image_path)
        pipeline_result = pipeline.run(image)
        output_path = output_dir / f"{image_path.stem}.txt"

        if pipeline_result.detection is None or not pipeline_result.text:
            # The assignment requires silence for negative frames, so delete stale files.
            output_path.unlink(missing_ok=True)
            results[image_path.name] = None
            continue

        text = pipeline_result.text
        # Persist the recognised sequence in a single-line text file (imgX.txt).
        save_text(output_path, text + "\n")
        results[image_path.name] = text

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 4 complete pipeline")
    parser.add_argument("input", type=str, help="Directory or image path for processing")
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

    summary = run_task4(args.input, overrides)
    for name, text in summary.items():
        if text is None:
            print(f"{name}: negative")
        else:
            print(f"{name}: \"{text}\"")
