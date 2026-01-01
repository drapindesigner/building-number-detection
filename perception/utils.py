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

"""General-purpose utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box expressed as pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.width * self.height


def clamp_box(xyxy: np.ndarray, width: int, height: int) -> BoundingBox:
    """Clamp floating point ``xyxy`` coordinates to image bounds."""

    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return BoundingBox(x1, y1, x2, y2)


def expand_box(box: BoundingBox, image_shape: Tuple[int, int], factor: float) -> BoundingBox:
    """Scale a bounding box around its centre while clamping to ``image_shape``."""

    height, width = image_shape
    cx = (box.x1 + box.x2) / 2.0
    cy = (box.y1 + box.y2) / 2.0
    w = box.width * factor
    h = box.height * factor
    x1 = max(0, int(round(cx - w / 2)))
    y1 = max(0, int(round(cy - h / 2)))
    x2 = min(width, int(round(cx + w / 2)))
    y2 = min(height, int(round(cy + h / 2)))
    return BoundingBox(x1, y1, x2, y2)


def crop(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Return the rectangular crop denoted by ``box``."""

    return image[box.y1:box.y2, box.x1:box.x2]


def set_global_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for deterministic runs."""

    if seed is None:
        return

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    os.environ.setdefault("PYTHONHASHSEED", str(value))

    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():  # depends on runtime hardware
            torch.cuda.manual_seed_all(value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
