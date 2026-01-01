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

"""Run Task 3: Recognize individual character images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from perception import (
    load_assignment_config,
    load_config_overrides_from_file,
    normalise_building_text,
)
from perception.io_utils import collect_images, ensure_dir, list_subdirectories, load_image, save_text
from perception.recognition import CharacterRecognizer


# Cache recognizer instances to avoid reloading model weights
_RECOGNIZER_CACHE: Dict[Path, CharacterRecognizer] = {}


def _get_recognizer(config) -> CharacterRecognizer:
    """
    Get or create a cached character recognizer instance.
    
    Args:
        config: Recognition configuration with model weights path
        
    Returns:
        CharacterRecognizer instance
    """
    key = config.weights_path.resolve()
    recognizer = _RECOGNIZER_CACHE.get(key)
    if recognizer is None:
        recognizer = CharacterRecognizer(config)
        _RECOGNIZER_CACHE[key] = recognizer
    return recognizer


def _collect_character_sets(root: Path) -> Dict[str, List[Path]]:
    """
    Collect character images organized by building number.
    
    Args:
        root: Root directory containing bnX subdirectories
        
    Returns:
        Dictionary mapping building number names to lists of character image paths
    """
    if root.is_file():
        raise ValueError("Expected a directory of character images, received a file")
    subdirs = list(list_subdirectories(root))
    if not subdirs:
        images = collect_images(root)
        return {root.stem: images}
    grouped: Dict[str, List[Path]] = {}
    for subdir in subdirs:
        images = collect_images(subdir)
        grouped[subdir.name] = images
    return grouped


def run_task3(
    input_path: Union[str, Path],
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """
    Task 3: Recognize individual character images.
    
    This function loads character images, applies a trained CNN classifier,
    and saves recognition results to individual text files.
    
    Args:
        input_path: Path to directory containing bnX subdirectories with character images
        config: Optional configuration dictionary to override defaults
        
    Returns:
        Dictionary mapping building number names to concatenated recognition strings
        
    Example:
        >>> results = run_task3("validation/task3")
        >>> print(results["bn1"])
        '314'
    """
    overrides = dict(config or {})
    assignment_cfg = load_assignment_config(overrides, base_path=Path.cwd())
    output_dir = ensure_dir(assignment_cfg.task_output_dir("task3"))
    recognizer = _get_recognizer(assignment_cfg.recognition)

    root = Path(input_path)
    character_sets = _collect_character_sets(root)

    results: Dict[str, str] = {}
    for name, image_paths in character_sets.items():
        # Skip if no images found (negative image)
        if not image_paths:
            results[name] = ""
            # Remove any stale directory from previous runs
            bn_output_dir = output_dir / name
            if bn_output_dir.exists():
                for obsolete in bn_output_dir.glob("c*.txt"):
                    obsolete.unlink()
                # Try to remove the directory if it's empty
                try:
                    bn_output_dir.rmdir()
                except OSError:
                    pass  # Directory not empty, leave it
            continue

        bn_output_dir = ensure_dir(output_dir / name)

        # Remove stale outputs so re-running does not leave outdated predictions.
        for obsolete in bn_output_dir.glob("c*.txt"):
            obsolete.unlink()

        images = [load_image(path) for path in image_paths]
        text, predictions = recognizer.predict_text(images)
        if len(predictions) != len(image_paths):
            raise RuntimeError("Mismatch between predictions and input characters")

        thresholded_chars: List[str] = list(text)
        if len(thresholded_chars) != len(image_paths):
            thresholded_chars = [pred.character for pred in predictions]

        characters: List[str] = []
        for idx, path in enumerate(image_paths):
            char = thresholded_chars[idx] if idx < len(thresholded_chars) else predictions[idx].character
            if char in {"", " "}:
                char = predictions[idx].character
            characters.append(char)
            char_output_path = bn_output_dir / f"{path.stem}.txt"
            # Each text file contains exactly one recognised character.
            save_text(char_output_path, char + "\n")

        raw_text = "".join(characters)
        normalized_text = normalise_building_text(raw_text)
        results[name] = normalized_text or raw_text
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Task 3 character recognition")
    parser.add_argument("input", type=str, help="Directory containing bn*/ character images")
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

    summary = run_task3(args.input, overrides)
    for name, text in summary.items():
        print(f"{name}: \"{text}\"")
