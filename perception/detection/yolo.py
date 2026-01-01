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

"""YOLO-based building number detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from ..config import DetectionConfig


@dataclass
class DetectionResult:
    """Container for a single YOLO detection."""

    box: np.ndarray  # xyxy order
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> np.ndarray:
        return self.box

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


class YoloDetector:
    """Thin wrapper around the Ultralytics YOLO detector used in Task 1."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        if not config.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {config.weights_path}")
        # Loading the Ultralytics checkpoint lazily keeps the CLI lightweight.
        self.model = YOLO(config.weights_path)

    def _select(self, detections: List[DetectionResult]) -> Optional[DetectionResult]:
        """Return the highest-confidence detection meeting the configured threshold."""

        for detection in detections:
            if detection.confidence >= self.config.min_return_confidence:
                return detection
        return None

    def predict(
        self,
        image: np.ndarray,
        imgsz: Optional[int] = None,
        augment: Optional[bool] = None,
    ) -> List[DetectionResult]:
        """Run YOLO detection and return sorted results (highest confidence first)."""

        use_aug = self.config.use_augmentation if augment is None else augment
        result = self.model.predict(
            image,
            imgsz=imgsz or self.config.image_size,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device or None,
            max_det=self.config.max_det,
            verbose=False,
            augment=use_aug,
        )
        outputs: List[DetectionResult] = []
        if not result:
            return outputs
        first = result[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None:
            return outputs
        count = len(boxes)
        if count == 0:
            return outputs
        names = first.names or {}
        for idx in range(count):
            xyxy = boxes.xyxy[idx].cpu().numpy()
            conf = float(boxes.conf[idx].cpu().item())
            cls = int(boxes.cls[idx].cpu().item())
            outputs.append(
                DetectionResult(
                    box=xyxy,
                    confidence=conf,
                    class_id=cls,
                    class_name=names.get(cls, str(cls)),
                )
            )
        outputs.sort(key=lambda det: det.confidence, reverse=True)
        return outputs

    def _prediction_scenarios(self) -> Iterator[Tuple[Optional[int], bool]]:
        """Yield (imgsz, augment) combinations to try in order of preference."""

        yield (None, False)

        seen: set[int] = {self.config.image_size}
        for size in self.config.fallback_image_sizes:
            if size is None:
                continue
            if size in seen:
                continue
            seen.add(size)
            yield (size, False)
            if self.config.use_augmentation:
                yield (size, True)

        if self.config.use_augmentation:
            yield (None, True)

    def best_detection(self, image: np.ndarray) -> Optional[DetectionResult]:
        """Return the best available detection, exploring fallbacks and augmentation."""

        for size, augment in self._prediction_scenarios():
            detections = self.predict(image, imgsz=size, augment=augment)
            selection = self._select(detections)
            if selection is not None:
                return selection
        return None
