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

"""High-level pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import AssignmentConfig
from .domain import normalise_building_text
from .detection import DetectionResult, YoloDetector
from .recognition import CharacterRecognizer, Prediction
from .segmentation import CharacterSegment, CharacterSegmenter
from .utils import BoundingBox, clamp_box, crop, expand_box


@dataclass
class PipelineResult:
    detection: Optional[DetectionResult]
    crop: Optional[np.ndarray]
    box: Optional[BoundingBox]
    segments: List[CharacterSegment]
    text: Optional[str]
    predictions: List[Prediction]
    raw_text: Optional[str] = None


class BuildingNumberPipeline:
    def __init__(self, config: AssignmentConfig) -> None:
        self.config = config
        self.detector = YoloDetector(config.detection)
        self.segmenter = CharacterSegmenter(config.segmentation)
        self.recognizer = CharacterRecognizer(config.recognition)

    def run(self, image: np.ndarray) -> PipelineResult:
        height, width = image.shape[:2]
        detection = self.detector.best_detection(image)
        if detection is None:
            # Immediate exit: Task 4 expects no output for negative frames.
            return PipelineResult(detection=None, crop=None, box=None, segments=[], text=None, predictions=[], raw_text=None)
        box = clamp_box(detection.xyxy, width, height)
        if self.config.detection.crop_expand != 1.0:
            box = expand_box(box, image.shape[:2], self.config.detection.crop_expand)
        crop_img = crop(image, box)
        segments = self.segmenter.segment(crop_img)
        text: Optional[str] = None
        raw_text: Optional[str] = None
        predictions: List[Prediction] = []
        if segments:
            char_images = [segment.crop for segment in segments]
            raw_text, predictions = self.recognizer.predict_text(char_images)
            if predictions:
                min_required = float(self.config.recognition.confidence_threshold)
                min_conf = min(pred.confidence for pred in predictions)
                if min_conf < min_required:
                    # Low confidence: keep intermediate artefacts for debugging but defer text.
                    return PipelineResult(
                        detection=detection,
                        crop=crop_img,
                        box=box,
                        segments=segments,
                        text=None,
                        predictions=predictions,
                        raw_text=raw_text,
                    )
            text = normalise_building_text(raw_text or "")
        return PipelineResult(
            detection=detection,
            crop=crop_img,
            box=box,
            segments=segments,
            text=text,
            predictions=predictions,
            raw_text=raw_text,
        )
