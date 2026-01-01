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

"""Run Task 2: Character segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from perception import load_assignment_config, load_config_overrides_from_file
from perception.io_utils import collect_images, ensure_dir, load_image, save_image
from perception.segmentation import CharacterSegmenter


def _format_index(index: int) -> str:
    """Return the canonical filename for the given character index."""
    return f"c{index}.png"


def run_task2(
    input_path: Union[str, Path],
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, List[str]]:
    """
    Task 2: Character segmentation.
    
    This function processes input images to segment individual characters
    and saves them in appropriately named subdirectories.

    Args:
        input_path: Path to directory containing input images
        config: Optional configuration dictionary to override defaults

    Returns:
        Dictionary mapping image names to lists of saved character image filenames

    Example:
        >>> summary = run_task2("validation/task1")
        >>> print(summary["bn1.png"])
        ['c1.png', 'c2.png', 'c3.png']
        >>> print(summary["bn3.png"])
        ['c1.png', 'c2.png']
    """
    overrides = dict(config or {})
    assignment_cfg = load_assignment_config(overrides, base_path=Path.cwd())
    output_dir = ensure_dir(assignment_cfg.task_output_dir("task2"))
    segmenter = CharacterSegmenter(assignment_cfg.segmentation)

    image_paths = collect_images(Path(input_path))
    results: Dict[str, List[str]] = {}

    for image_path in image_paths:
        image = load_image(image_path)
        segments = segmenter.segment(image)

        # Skip if no segments detected (negative image)
        if not segments:
            results[image_path.name] = []
            # Remove any stale directory from previous runs
            subdir = output_dir / image_path.stem
            if subdir.exists():
                for obsolete in subdir.glob("c*.png"):
                    obsolete.unlink()
                # Try to remove the directory if it's empty
                try:
                    subdir.rmdir()
                except OSError:
                    pass  # Directory not empty, leave it
            continue

        subdir = ensure_dir(output_dir / image_path.stem)

        # Remove any stale character outputs from earlier runs to keep the folder clean.
        for obsolete in subdir.glob("c*.png"):
            obsolete.unlink()

        filenames: List[str] = []
        for idx, segment in enumerate(segments, start=1):
            filename = _format_index(idx)
            # Characters are saved left-to-right with contiguous numbering as required.
            save_image(subdir / filename, segment.crop)
            filenames.append(filename)
        results[image_path.name] = filenames

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 2 character segmentation")
    parser.add_argument("input", type=str, help="Directory containing bn*.png images")
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

    summary = run_task2(args.input, overrides)
    for name, files in summary.items():
        print(f"{name}: {len(files)} characters")
