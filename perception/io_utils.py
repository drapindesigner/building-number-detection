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

"""Input/output helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import cv2 as cv
import numpy as np


_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def collect_images(path: Path) -> List[Path]:
    """Return sorted image paths under ``path`` (supports individual files)."""

    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files: List[Path] = []
    for pattern in _IMAGE_PATTERNS:
        files.extend(sorted(path.glob(pattern)))
    if not files:
        return []
    files.sort(key=lambda p: (_extract_index(p.stem), p.stem))
    return files


def _extract_index(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def load_image(path: Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    """Load an image via OpenCV and raise a descriptive error on failure."""

    image = cv.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Write ``image`` to ``path`` ensuring parent directories exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it for chaining."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_text(path: Path, content: str) -> None:
    """Persist UTF-8 text to ``path`` with directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def list_subdirectories(path: Path) -> Iterable[Path]:
    """Yield sorted immediate sub-directories under ``path`` (empty if missing)."""

    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir())
