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
"""Character segmentation implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np

from ..config import SegmentationConfig
from ..utils import BoundingBox, clamp_box


@dataclass
class _RotationInfo:
    matrix: np.ndarray
    inverse: np.ndarray
    original_shape: Tuple[int, int]

try:  # Lazy import to avoid mandatory dependency when using heuristics only
    from ultralytics import YOLO as _YOLO
except Exception:  # ultralytics may be unavailable in some environments
    _YOLO = None


def _box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = box_a.as_tuple()
    bx1, by1, bx2, by2 = box_b.as_tuple()
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    if inter_w == 0 or inter_h == 0:
        return 0.0
    inter_area = inter_w * inter_h
    union = box_a.area() + box_b.area() - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

@dataclass
class CharacterSegment:
    crop: np.ndarray
    box: BoundingBox
    mask: np.ndarray
    polarity: str
    score: float
    original_box: Optional[BoundingBox] = None

    @property
    def position(self) -> int:
        return self.box.x1


class CharacterSegmenter:
    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self.mode = str(config.mode or "heuristic").strip().lower()
        self._yolo_model = None
        self.last_processed_image: Optional[np.ndarray] = None
        self._last_rotation_info: Optional[_RotationInfo] = None
        if self.mode in {"learned", "hybrid"}:
            weights_path = config.model_weights
            if weights_path and weights_path.exists() and _YOLO is not None:
                self._yolo_model = _YOLO(str(weights_path))
            elif self.mode == "learned":
                if _YOLO is None:
                    raise ImportError("Ultralytics YOLO is required for learned segmentation but is not installed.")
                raise FileNotFoundError(f"Segmentation model weights not found: {weights_path}")
            else:
                if _YOLO is None or not (weights_path and weights_path.exists()):
                    print("[CharacterSegmenter] Learned segmentation disabled: YOLO unavailable or weights missing; falling back to heuristics.")

    def segment(self, image: np.ndarray) -> List[CharacterSegment]:
        self.last_processed_image = image
        self._last_rotation_info = None
        if self._yolo_model is not None:
            learned_segments = self._segment_with_yolo(image)
            if learned_segments or self.mode == "learned":
                return learned_segments
        return self._segment_with_heuristics(image)

    def _deskew_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[_RotationInfo]]:
        if not getattr(self.config, "deskew_enabled", False):
            return image, None

        if image.ndim == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        white_ratio = float(cv.countNonZero(binary)) / max(1, binary.size)
        if white_ratio > 0.5:
            binary = cv.bitwise_not(binary)

        moments = cv.moments(binary)
        denom = moments["mu20"] + moments["mu02"]
        if abs(denom) < 1e-6:
            return image, None

        angle_rad = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
        angle_deg = float(angle_rad * 180.0 / np.pi)
        min_angle = float(getattr(self.config, "deskew_min_angle", 0.0))
        max_angle = float(getattr(self.config, "deskew_max_angle", 45.0))
        if abs(angle_deg) < min_angle or abs(angle_deg) > max_angle:
            return image, None

        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        rotation = cv.getRotationMatrix2D(center, -angle_deg, 1.0)
        cos = abs(rotation[0, 0])
        sin = abs(rotation[0, 1])
        new_width = int(round(height * sin + width * cos))
        new_height = int(round(height * cos + width * sin))
        rotation[0, 2] += new_width / 2.0 - center[0]
        rotation[1, 2] += new_height / 2.0 - center[1]

        rotated = cv.warpAffine(
            image,
            rotation,
            (new_width, new_height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REPLICATE,
        )

        matrix = np.vstack([rotation, np.array([0.0, 0.0, 1.0])])
        inverse = np.linalg.inv(matrix)
        info = _RotationInfo(matrix=matrix, inverse=inverse, original_shape=(height, width))
        return rotated, info

    def _segment_with_heuristics(self, image: np.ndarray) -> List[CharacterSegment]:
        """Fallback segmentation pipeline relying on classic image processing."""

        working_image, rotation_info = self._deskew_if_needed(image)
        self.last_processed_image = working_image
        self._last_rotation_info = rotation_info

        gray = cv.cvtColor(working_image, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=self.config.clahe_clip, tileGridSize=(self.config.clahe_grid, self.config.clahe_grid))
        enhanced = clahe.apply(gray)
        blurred = cv.GaussianBlur(enhanced, (self._kernel_size(), self._kernel_size()), 0)

        target_guess = self._estimate_digit_count(working_image.shape[1], working_image.shape[0], [])

        variants = self._binarize_variants(blurred)
        best_processed: List[CharacterSegment] = []
        best_score = -1e9
        for idx, (mask, polarity, label) in enumerate(variants):
            # Evaluate several binarisation strategies and keep the best-scoring set.
            raw_segments = self._extract_segments(working_image, mask, polarity)
            processed_candidate, metrics = self._evaluate_candidate(raw_segments, working_image, target_guess)
            metrics.update({
                "source": "variant",
                "index": idx,
                "label": label,
                "polarity": polarity,
            })
            score_value = metrics.get("score")
            if isinstance(score_value, (int, float, np.floating)):
                score = float(score_value)
            else:
                score = -1e9
            if score > best_score:
                best_score = score
                best_processed = processed_candidate

        watershed_segments = self._watershed_segments(working_image, blurred)
        if watershed_segments:
            processed_candidate, metrics = self._evaluate_candidate(watershed_segments, working_image, target_guess)
            metrics.update({
                "source": "watershed",
                "label": "watershed",
            })
            score_value = metrics.get("score")
            if isinstance(score_value, (int, float, np.floating)):
                score = float(score_value)
            else:
                score = -1e9
            if score > best_score:
                best_score = score
                best_processed = processed_candidate

        processed = best_processed
        if not processed:
            fallback_segments = self._fallback_segments(working_image, blurred)
            if fallback_segments:
                processed, metrics = self._evaluate_candidate(fallback_segments, working_image, target_guess)
                metrics.update({
                    "source": "fallback",
                    "label": "fallback",
                })
        return self._finalize_segments(working_image, processed or [], blurred, rotation_info)

    def _evaluate_candidate(
        self,
        segments: List[CharacterSegment],
        image: np.ndarray,
        target_digits: int,
    ) -> Tuple[List[CharacterSegment], Dict[str, object]]:
        processed = self._postprocess_segments(list(segments), image)
        metrics = self._segment_metrics(processed, (int(image.shape[0]), int(image.shape[1])), target_digits)
        return processed, metrics

    def _segment_metrics(
        self,
        segments: List[CharacterSegment],
        image_shape: Tuple[int, int],
        target_digits: int,
    ) -> Dict[str, object]:
        metrics: Dict[str, object] = {}
        count = len(segments)
        metrics["count"] = count
        if count == 0:
            metrics["coverage"] = 0.0
            metrics["count_penalty"] = 0.0
            metrics["width_consistency"] = 0.0
            metrics["gap_consistency"] = 0.0
            metrics["score"] = -1.0
            return metrics

        height, width = image_shape[:2]
        total_area = max(1.0, float(width * height))
        coverage = float(sum(segment.box.area() for segment in segments)) / total_area
        coverage = float(np.clip(coverage, 0.0, 1.5))
        metrics["coverage"] = coverage

        if target_digits and target_digits > 0:
            deviation = abs(count - target_digits)
            count_penalty = float(np.exp(-0.75 * deviation))
        else:
            count_penalty = 1.0
        metrics["count_penalty"] = count_penalty

        widths = np.array([segment.box.width for segment in segments], dtype=np.float32)
        if widths.size > 1:
            width_cv = float(np.std(widths) / max(np.mean(widths), 1.0))
            width_consistency = 1.0 / (1.0 + width_cv)
        else:
            width_consistency = 0.9
        metrics["width_consistency"] = width_consistency

        if count > 1:
            sorted_segments = sorted(segments, key=lambda seg: seg.box.x1)
            gaps = []
            overlaps: List[float] = []
            for left, right in zip(sorted_segments[:-1], sorted_segments[1:]):
                gap = max(0.0, float(right.box.x1 - left.box.x2))
                gaps.append(gap)
                overlap_width = float(left.box.x2 - right.box.x1)
                if overlap_width > 0.0:
                    min_width = max(1.0, float(min(left.box.width, right.box.width)))
                    overlaps.append(overlap_width / min_width)
            if gaps:
                gaps_array = np.array(gaps, dtype=np.float32)
                mean_gap = float(np.mean(gaps_array) + 1e-5)
                gap_cv = float(np.std(gaps_array) / mean_gap)
                gap_consistency = 1.0 / (1.0 + gap_cv)
            else:
                gap_consistency = 1.0
            if overlaps:
                mean_overlap = float(np.mean(np.array(overlaps, dtype=np.float32)))
                overlap_penalty = float(np.exp(-3.0 * mean_overlap))
            else:
                overlap_penalty = 1.0
        else:
            gap_consistency = 0.8
            overlap_penalty = 1.0
        metrics["gap_consistency"] = gap_consistency
        metrics["overlap_penalty"] = overlap_penalty

        fill_ratios: List[float] = []
        for segment in segments:
            mask = segment.mask
            if mask.size == 0:
                continue
            fill_value = float(np.count_nonzero(mask)) / float(mask.size)
            fill_ratios.append(fill_value)
        if fill_ratios:
            avg_fill = float(np.mean(fill_ratios))
            min_fill = min(fill_ratios)
            fill_level = min(1.0, avg_fill / 0.55)
            low_penalty = min_fill / 0.1 if min_fill < 0.1 else 1.0
            fill_consistency = 0.5 * fill_level + 0.5 * min(low_penalty, 1.0)
        else:
            fill_consistency = 0.5
        metrics["fill_consistency"] = fill_consistency

        base_score = (
            0.3 * coverage
            + 0.25 * count_penalty
            + 0.12 * width_consistency
            + 0.1 * gap_consistency
            + 0.1 * fill_consistency
            + 0.13 * overlap_penalty
        )
        if target_digits and target_digits > 0:
            overlap_factor = overlap_penalty if isinstance(overlap_penalty, (float, int)) else 1.0
            if count == target_digits:
                match_bonus = 0.2 * float(overlap_factor)
            else:
                match_bonus = 0.2 * float(overlap_factor) * float(np.exp(-abs(count - target_digits)))
        else:
            match_bonus = 0.0
        metrics["score"] = float(base_score + match_bonus)
        return metrics

    def _segment_with_yolo(self, image: np.ndarray) -> List[CharacterSegment]:
        if self._yolo_model is None:
            return []

        self.last_processed_image = image
        self._last_rotation_info = None

        try:
            predictions = self._yolo_model.predict(
                image,
                imgsz=self.config.model_image_size,
                conf=self.config.model_confidence,
                iou=self.config.model_iou,
                max_det=self.config.model_max_det,
                device=self.config.model_device or None,
                verbose=False,
                augment=self.config.model_use_augmentation,
            )
        except Exception as exc:  # pragma: no cover - defensive against YOLO runtime errors
            print(f"[CharacterSegmenter] YOLO inference failed: {exc}")
            return []

        if not predictions:
            return []

        first = predictions[0]
        boxes = getattr(first, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image.shape[:2]
        segments: List[CharacterSegment] = []
        for idx in range(len(boxes)):
            conf = float(boxes.conf[idx].detach().cpu().item())
            if conf < float(self.config.model_min_return_confidence):
                continue
            xyxy = boxes.xyxy[idx].detach().cpu().numpy()
            box = clamp_box(xyxy, width, height)
            crop = image[box.y1:box.y2, box.x1:box.x2]
            mask = self._derive_local_mask(crop)
            filled = float(cv.countNonZero(mask)) / max(1.0, mask.size)
            score = conf * (1.0 + filled) * max(1, box.area())
            segments.append(
                CharacterSegment(
                    crop=crop,
                    box=box,
                    mask=mask,
                    polarity="bright",
                    score=score,
                )
            )

        processed = self._postprocess_segments(segments, image)
        return self._finalize_segments(image, processed, blurred=None, rotation_info=None)

    def _finalize_segments(
        self,
        image: np.ndarray,
        segments: List[CharacterSegment],
        blurred: Optional[np.ndarray],
        rotation_info: Optional[_RotationInfo],
    ) -> List[CharacterSegment]:
        """Apply final sanity checks, ordering, and rotation undoing to segments."""

        processed = list(segments) if segments else []

        if not processed and blurred is not None:
            fallback_segments = self._fallback_segments(image, blurred)
            if fallback_segments:
                processed = self._postprocess_segments(fallback_segments, image)

        processed.sort(key=lambda seg: seg.position)
        target_digits = self._estimate_digit_count(image.shape[1], image.shape[0], processed)

        if not processed or len(processed) < 2 or len(processed) > 4 or len(processed) != target_digits:
            mask_full = self._derive_local_mask(image)
            equal_segments = self._split_equal_segments(image, mask_full, target_digits)
            if equal_segments:
                processed = equal_segments

        processed.sort(key=lambda seg: seg.position)

        if rotation_info is not None and len(processed) == 2:
            ratio = image.shape[1] / max(1.0, float(image.shape[0]))
            if ratio >= 1.15:
                mask_full = self._derive_local_mask(image)
                three_segments = self._split_equal_segments(image, mask_full, 3)
                if len(three_segments) == 3:
                    processed = three_segments
                    processed.sort(key=lambda seg: seg.position)

        self._restore_original_boxes(processed, rotation_info)

        return processed

    def _restore_original_boxes(self, segments: List[CharacterSegment], rotation_info: Optional[_RotationInfo]) -> None:
        """Map segment boxes back to the original image if deskewing was applied."""

        if not segments:
            return

        if rotation_info is None:
            for segment in segments:
                segment.original_box = segment.box
            return

        inverse = rotation_info.inverse
        orig_height, orig_width = rotation_info.original_shape
        for segment in segments:
            x1, y1, x2, y2 = segment.box.as_tuple()
            corners = np.array(
                [
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x1, y2, 1.0],
                    [x2, y2, 1.0],
                ],
                dtype=np.float32,
            )
            transformed = (inverse @ corners.T).T
            xs = transformed[:, 0]
            ys = transformed[:, 1]
            min_x = float(xs.min())
            max_x = float(xs.max())
            min_y = float(ys.min())
            max_y = float(ys.max())
            original_box = clamp_box(
                np.array([min_x, min_y, max_x, max_y], dtype=np.float32),
                orig_width,
                orig_height,
            )
            segment.original_box = original_box

    def _kernel_size(self) -> int:
        size = max(3, int(self.config.gaussian_ksize))
        return size if size % 2 == 1 else size + 1

    def _binarize_variants(self, blurred: np.ndarray) -> List[Tuple[np.ndarray, str, str]]:
        """Generate a small ensemble of binarised masks with different polarities."""

        base_block = max(3, self._kernel_size() * 3)
        block_sizes = sorted({base_block, base_block + 10, base_block + 20})
        variants: List[Tuple[np.ndarray, str, str]] = []

        def _ensure_odd(value: int) -> int:
            return value if value % 2 == 1 else value + 1

        inverted = 255 - blurred

        for idx, bs in enumerate(block_sizes):
            bs = _ensure_odd(bs)
            adaptive_inv = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, bs, 5)
            variants.append((adaptive_inv, "bright", f"adaptive_inv_bs{bs}"))
            adaptive = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, bs, 5)
            variants.append((adaptive, "dark", f"adaptive_bs{bs}"))
            adaptive_inv_inverted = cv.adaptiveThreshold(inverted, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, bs, 5)
            variants.append((adaptive_inv_inverted, "dark", f"adaptive_inv_inverted_bs{bs}"))

        _, otsu_inv = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        variants.append((otsu_inv, "bright", "otsu_inv"))
        _, otsu = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        variants.append((otsu, "dark", "otsu"))
        _, otsu_inv_inverted = cv.threshold(inverted, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        variants.append((otsu_inv_inverted, "dark", "otsu_inv_inverted"))
        _, otsu_inverted = cv.threshold(inverted, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        variants.append((otsu_inverted, "bright", "otsu_inverted"))

        kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        cleaned: List[Tuple[np.ndarray, str, str]] = []
        for mask, polarity, label in variants:
            morph = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close, iterations=1)
            dilated = cv.dilate(morph, kernel_dilate, iterations=1)
            cleaned.append((dilated, polarity, f"{label}_close_dilate"))
        return cleaned

    def _extract_segments(self, image: np.ndarray, mask: np.ndarray, polarity: str) -> List[CharacterSegment]:
        """Cut out connected components that look like plausible characters."""

        white_ratio = float(cv.countNonZero(mask)) / max(1, mask.size)
        if white_ratio > 0.5:
            mask = cv.bitwise_not(mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        height, width = mask.shape
        segments: List[CharacterSegment] = []
        total_area = height * width
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            if area < self.config.min_area:
                continue
            if area > total_area * self.config.max_area_ratio:
                continue
            aspect = w / float(h)
            if not (self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio):
                continue
            height_ratio = h / float(height)
            if not (self.config.min_height_ratio <= height_ratio <= self.config.max_height_ratio):
                continue
            if x == 0 or y == 0 or x + w >= width or y + h >= height:
                continue
            padding = self.config.padding
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(width, x + w + padding)
            y1 = min(height, y + h + padding)
            box = clamp_box(np.array([x0, y0, x1, y1]), width, height)
            crop = image[box.y1:box.y2, box.x1:box.x2]
            mask_crop = mask[box.y1:box.y2, box.x1:box.x2]
            filled_ratio = float(cv.countNonZero(mask_crop)) / max(1.0, mask_crop.size)
            score = filled_ratio * area
            segments.append(CharacterSegment(crop=crop, box=box, mask=mask_crop, polarity=polarity, score=score))
        return segments

    def _watershed_segments(self, image: np.ndarray, blurred: np.ndarray) -> List[CharacterSegment]:
        gray = blurred
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if float(cv.countNonZero(binary)) / max(1, binary.size) > 0.5:
            binary = cv.bitwise_not(binary)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        _, sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv.subtract(sure_bg, sure_fg)

        num_markers, markers = cv.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        color_image = image.copy()
        cv.watershed(color_image, markers)

        height, width = markers.shape
        segments: List[CharacterSegment] = []
        total_area = height * width
        for marker in range(2, num_markers + 1):
            mask = np.uint8(markers == marker) * 255
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            if area < self.config.min_area:
                continue
            if area > total_area * self.config.max_area_ratio:
                continue
            aspect = w / float(h)
            if not (self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio):
                continue
            height_ratio = h / float(height)
            if not (self.config.min_height_ratio <= height_ratio <= self.config.max_height_ratio):
                continue
            padding = self.config.padding
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(width, x + w + padding)
            y1 = min(height, y + h + padding)
            box = clamp_box(np.array([x0, y0, x1, y1]), width, height)
            crop = image[box.y1:box.y2, box.x1:box.x2]
            mask_crop = mask[box.y1:box.y2, box.x1:box.x2]
            filled_ratio = float(cv.countNonZero(mask_crop)) / max(1.0, mask_crop.size)
            score = filled_ratio * area
            segments.append(CharacterSegment(crop=crop, box=box, mask=mask_crop, polarity="bright", score=score))

        return segments

    def _postprocess_segments(self, segments: List[CharacterSegment], image: np.ndarray) -> List[CharacterSegment]:
        if not segments:
            return []

        areas = np.array([segment.box.area() for segment in segments], dtype=np.float32)
        median_area = float(np.median(areas)) if areas.size else 0.0
        min_relative_area = float(self.config.min_relative_area)
        minimum_area = max(float(self.config.min_area), median_area * min_relative_area) if median_area > 0 else float(self.config.min_area)

        filtered: List[CharacterSegment] = []
        for segment in segments:
            if segment.box.area() < minimum_area:
                continue
            filtered.append(segment)
        if not filtered:
            filtered = segments

        deduped: List[CharacterSegment] = []
        for segment in sorted(filtered, key=lambda seg: seg.box.area(), reverse=True):
            if any(_box_iou(segment.box, keep.box) > self.config.duplicate_iou for keep in deduped):
                continue
            deduped.append(segment)

        if not deduped:
            deduped = filtered

        merged: List[CharacterSegment] = []
        segments_sorted = sorted(deduped, key=lambda seg: seg.box.x1)
        widths = np.array([segment.box.width for segment in segments_sorted], dtype=np.float32)
        median_width = float(np.median(widths)) if widths.size else 0.0
        min_width = median_width * float(self.config.min_width_ratio) if median_width > 0 else float(self.config.min_area)

        i = 0
        while i < len(segments_sorted):
            current = segments_sorted[i]
            if current.box.width >= min_width:
                merged.append(current)
                i += 1
                continue

            merged_segment = None
            if i + 1 < len(segments_sorted):
                neighbor = segments_sorted[i + 1]
                gap = neighbor.box.x1 - current.box.x2
                overlap = current.box.x2 - neighbor.box.x1
                if gap <= self.config.merge_gap_px or overlap >= -self.config.merge_overlap_px:
                    merged_segment = self._merge_segments(current, neighbor, image)
                    segments_sorted[i + 1] = merged_segment
                    i += 1
            if merged_segment is None and merged:
                neighbor = merged[-1]
                gap = current.box.x1 - neighbor.box.x2
                overlap = neighbor.box.x2 - current.box.x1
                if gap <= self.config.merge_gap_px or overlap >= -self.config.merge_overlap_px:
                    merged[-1] = self._merge_segments(neighbor, current, image)
                    i += 1
                    continue

            if merged_segment is None:
                if current.box.width >= min_width * 0.6:
                    merged.append(current)
                i += 1
            else:
                # merged into next, do not append current separately
                pass

        refined: List[CharacterSegment] = []
        for segment in merged:
            refined.extend(self._split_wide_segment(segment, image))

        if not refined:
            refined = merged
        else:
            refined = self._merge_small_segments(refined, max(1.0, min_width * 0.6), image)

        final_segments: List[CharacterSegment] = []
        for segment in refined:
            if any(_box_iou(segment.box, keep.box) > self.config.duplicate_iou for keep in final_segments):
                continue
            final_segments.append(segment)

        if not final_segments:
            final_segments = refined or deduped

        tightened = [self._tighten_segment(segment, image) for segment in final_segments]
        tightened = [segment for segment in tightened if segment is not None]
        return tightened

    def _merge_segments(self, seg_a: CharacterSegment, seg_b: CharacterSegment, image: np.ndarray) -> CharacterSegment:
        x1 = min(seg_a.box.x1, seg_b.box.x1)
        y1 = min(seg_a.box.y1, seg_b.box.y1)
        x2 = max(seg_a.box.x2, seg_b.box.x2)
        y2 = max(seg_a.box.y2, seg_b.box.y2)
        new_box = BoundingBox(x1, y1, x2, y2)

        crop = image[new_box.y1:new_box.y2, new_box.x1:new_box.x2]

        mask = np.zeros((new_box.height, new_box.width), dtype=np.uint8)

        def _paste(segment: CharacterSegment) -> None:
            sx1, sy1, sx2, sy2 = segment.box.as_tuple()
            px1 = sx1 - new_box.x1
            py1 = sy1 - new_box.y1
            px2 = px1 + segment.mask.shape[1]
            py2 = py1 + segment.mask.shape[0]
            region = mask[py1:py2, px1:px2]
            mask[py1:py2, px1:px2] = cv.bitwise_or(region, segment.mask)

        _paste(seg_a)
        _paste(seg_b)

        filled_ratio = float(cv.countNonZero(mask)) / max(1.0, mask.size)
        score = filled_ratio * new_box.area()
        return CharacterSegment(crop=crop, box=new_box, mask=mask, polarity=seg_a.polarity, score=score)

    def _split_wide_segment(self, segment: CharacterSegment, image: np.ndarray) -> List[CharacterSegment]:
        """Divide overly wide components that likely contain multiple digits."""

        width = float(segment.box.width)
        height = float(segment.box.height)
        if height <= 0 or width / max(1.0, height) <= float(self.config.split_aspect_ratio):
            return [segment]

        mask = self._derive_local_mask(segment.crop)
        if mask.size == 0:
            return [segment]

        projection = np.sum(mask > 0, axis=0).astype(np.float32)
        if projection.size == 0:
            return [segment]
        rel_projection = projection / max(1.0, mask.shape[0])
        gap_ratio = max(float(self.config.split_gap_fraction), 0.18)
        low_columns = rel_projection <= gap_ratio

        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for idx, value in enumerate(low_columns):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx))
                start = None
        if start is not None:
            runs.append((start, len(low_columns)))

        valid_splits: List[int] = []
        min_gap = max(1, int(mask.shape[1] * max(float(self.config.split_gap_fraction), 0.05)))
        for left, right in runs:
            if right - left >= min_gap:
                valid_splits.append((left + right) // 2)

        if not valid_splits:
            contrast = float(rel_projection.max() - rel_projection.min())
            if contrast < 0.25:
                return [segment]
            approx = int(round(width / max(1.0, height * 0.85)))
            approx = max(2, min(approx, 5)) if approx > 1 else 0
            if approx:
                step = mask.shape[1] / float(approx)
                valid_splits = [int(round(step * i)) for i in range(1, approx)]
        if not valid_splits:
            return [segment]

        valid_splits = sorted(set(split for split in valid_splits if 0 < split < mask.shape[1]))
        if not valid_splits:
            return [segment]

        intervals: List[Tuple[int, int]] = []
        prev = 0
        for split in valid_splits:
            intervals.append((prev, split))
            prev = split
        intervals.append((prev, mask.shape[1]))

        min_width = max(1, int(mask.shape[1] * float(self.config.split_min_width_fraction)))
        filtered_intervals = [interval for interval in intervals if interval[1] - interval[0] >= min_width]
        if len(filtered_intervals) < 2:
            return [segment]

        image_height, image_width = image.shape[:2]
        new_segments: List[CharacterSegment] = []
        for left, right in filtered_intervals:
            sub_mask = mask[:, left:right]
            coords = cv.findNonZero(sub_mask)
            if coords is None:
                continue
            x, y, w, h = cv.boundingRect(coords)
            if w <= 0 or h <= 0:
                continue

            global_x1 = segment.box.x1 + left + x
            global_y1 = segment.box.y1 + y
            global_x2 = global_x1 + w
            global_y2 = global_y1 + h

            raw_box = np.array([global_x1, global_y1, global_x2, global_y2])
            new_box = clamp_box(raw_box, image_width, image_height)
            if new_box.width <= 1 or new_box.height <= 1:
                continue

            crop = image[new_box.y1:new_box.y2, new_box.x1:new_box.x2]
            mask_x1 = new_box.x1 - segment.box.x1
            mask_y1 = new_box.y1 - segment.box.y1
            mask_x2 = mask_x1 + crop.shape[1]
            mask_y2 = mask_y1 + crop.shape[0]
            if mask_y2 > mask.shape[0] or mask_x2 > mask.shape[1]:
                continue
            sub_mask_refined = mask[mask_y1:mask_y2, mask_x1:mask_x2]
            filled_ratio = float(cv.countNonZero(sub_mask_refined)) / max(1.0, sub_mask_refined.size)
            score = filled_ratio * new_box.area()
            new_segments.append(
                CharacterSegment(
                    crop=crop,
                    box=new_box,
                    mask=sub_mask_refined,
                    polarity=segment.polarity,
                    score=score,
                )
            )

        return new_segments or [segment]

    def _fallback_segments(self, image: np.ndarray, blurred: np.ndarray) -> List[CharacterSegment]:
        """Last-resort segment candidates when other strategies fail."""

        masks: List[np.ndarray] = []
        _, otsu_inv = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        masks.append(otsu_inv)
        masks.append(cv.bitwise_not(otsu_inv))

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        candidates: List[Tuple[float, List[CharacterSegment]]] = []
        for mask in masks:
            cleaned = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
            cleaned = cv.dilate(cleaned, kernel, iterations=1)
            segments = self._extract_segments_loose(image, cleaned)
            if segments:
                score = sum(segment.score for segment in segments)
                candidates.append((score, segments))
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _extract_segments_loose(self, image: np.ndarray, mask: np.ndarray) -> List[CharacterSegment]:
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        height, width = mask.shape
        total_area = height * width
        min_area = max(25, int(total_area * 0.004))
        max_area_ratio = 0.9
        min_aspect = 0.1
        max_aspect = 4.0
        min_height_ratio = 0.18

        segments: List[CharacterSegment] = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            if area < min_area:
                continue
            if area > total_area * max_area_ratio:
                continue
            aspect = w / float(h)
            if not (min_aspect <= aspect <= max_aspect):
                continue
            height_ratio = h / float(height)
            if height_ratio < min_height_ratio:
                continue
            padding = int(self.config.padding)
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(width, x + w + padding)
            y1 = min(height, y + h + padding)
            box = clamp_box(np.array([x0, y0, x1, y1]), width, height)
            crop = image[box.y1:box.y2, box.x1:box.x2]
            mask_crop = mask[box.y1:box.y2, box.x1:box.x2]
            filled_ratio = float(cv.countNonZero(mask_crop)) / max(1.0, mask_crop.size)
            score = filled_ratio * box.area()
            segments.append(
                CharacterSegment(
                    crop=crop,
                    box=box,
                    mask=mask_crop,
                    polarity="bright",
                    score=score,
                )
            )

        return segments

    def _derive_local_mask(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(crop.shape[:2], dtype=np.uint8)
        if crop.ndim == 3:
            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        else:
            gray = crop
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        masks = []
        for thresh_type in (cv.THRESH_BINARY_INV, cv.THRESH_BINARY):
            _, mask = cv.threshold(blurred, 0, 255, thresh_type + cv.THRESH_OTSU)
            masks.append(mask)
        selected = masks[0]
        best_score = -1.0
        for mask in masks:
            ratio = float(cv.countNonZero(mask)) / max(1.0, mask.size)
            score = -abs(0.5 - ratio)
            if score > best_score:
                best_score = score
                selected = mask
        return selected

    def _merge_small_segments(self, segments: List[CharacterSegment], min_width: float, image: np.ndarray) -> List[CharacterSegment]:
        if not segments:
            return []
        segments_sorted = sorted(segments, key=lambda seg: seg.box.x1)
        merged: List[CharacterSegment] = []
        i = 0
        while i < len(segments_sorted):
            current = segments_sorted[i]
            if current.box.width >= min_width or min_width <= 0:
                merged.append(current)
                i += 1
                continue

            partner = None
            merge_with_next = False
            if i + 1 < len(segments_sorted):
                partner = segments_sorted[i + 1]
                merge_with_next = True
            elif merged:
                partner = merged[-1]
                merge_with_next = False

            if partner is None:
                merged.append(current)
                i += 1
                continue

            if merge_with_next:
                combined = self._merge_segments(current, partner, image)
                segments_sorted[i + 1] = combined
            else:
                combined = self._merge_segments(partner, current, image)
                merged[-1] = combined
            i += 1
        return merged

    def _estimate_digit_count(self, width: int, height: int, segments: List[CharacterSegment]) -> int:
        ratio = width / max(1, height)
        if ratio < 0.85:
            guess = 1
        elif ratio < 1.35:
            guess = 2
        elif ratio < 2.2:
            guess = 3
        else:
            guess = 4

        if segments:
            count = len(segments)
            if count == guess:
                return count
            if guess == 4 and count == 3:
                return count
            if guess == 3 and count == 2:
                return guess
            if guess == 2 and count in (1, 2):
                return guess
            if count > guess and count <= 4:
                return count

        return guess

    def _split_equal_segments(self, image: np.ndarray, mask: np.ndarray, target: int) -> List[CharacterSegment]:
        if target <= 0:
            return []
        if mask is None or mask.size == 0:
            return []
        height, width = mask.shape
        binary = (mask > 0).astype(np.uint8)
        col_sum = binary.sum(axis=0)
        nonzero_cols = np.flatnonzero(col_sum)
        if nonzero_cols.size == 0:
            return []
        left = int(nonzero_cols[0])
        right = int(nonzero_cols[-1]) + 1
        col_segment = col_sum[left:right]
        total = col_segment.sum()
        if total <= 0:
            return []

        cumulative = np.cumsum(col_segment)
        boundaries: List[int] = [left]
        for idx in range(1, target):
            threshold = total * idx / target
            column_index = int(np.searchsorted(cumulative, threshold, side="left"))
            boundaries.append(min(right - 1, left + column_index))
        boundaries.append(right)

        unique_boundaries = []
        for value in boundaries:
            if not unique_boundaries or value > unique_boundaries[-1]:
                unique_boundaries.append(value)
        boundaries = unique_boundaries
        if len(boundaries) <= 1:
            return []

        segments: List[CharacterSegment] = []
        padding = int(self.config.padding)
        row_sum_total = binary.sum(axis=1)
        nonzero_rows = np.flatnonzero(row_sum_total)
        if nonzero_rows.size == 0:
            return []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start <= 1:
                continue
            sub_mask = binary[:, start:end]
            col_rows = sub_mask.sum(axis=1)
            nz_rows = np.flatnonzero(col_rows)
            if nz_rows.size == 0:
                continue
            y1 = max(0, int(nz_rows[0]) - padding)
            y2 = min(height, int(nz_rows[-1]) + 1 + padding)
            x1 = max(0, start - padding)
            x2 = min(width, end + padding)

            raw_box = np.array([x1, y1, x2, y2])
            box = clamp_box(raw_box, width, height)
            crop = image[box.y1:box.y2, box.x1:box.x2]
            mask_crop = (binary[box.y1:box.y2, box.x1:box.x2] * 255).astype(np.uint8)
            score = float(np.count_nonzero(mask_crop))
            segments.append(
                CharacterSegment(
                    crop=crop,
                    box=box,
                    mask=mask_crop,
                    polarity="bright",
                    score=score,
                )
            )

        if len(segments) < target:
            return []
        segments.sort(key=lambda seg: seg.position)
        return segments

    def _tighten_segment(self, segment: CharacterSegment, image: np.ndarray) -> Optional[CharacterSegment]:
        mask = segment.mask
        if mask.size == 0:
            return segment
        coords = cv.findNonZero(mask)
        if coords is None:
            return None
        x, y, w, h = cv.boundingRect(coords)
        if w <= 0 or h <= 0:
            return None

        global_x1 = segment.box.x1 + x
        global_y1 = segment.box.y1 + y
        global_x2 = global_x1 + w
        global_y2 = global_y1 + h

        height, width = image.shape[:2]
        tightened_box = clamp_box(np.array([global_x1, global_y1, global_x2, global_y2]), width, height)
        crop = image[tightened_box.y1:tightened_box.y2, tightened_box.x1:tightened_box.x2]

        local_x1 = tightened_box.x1 - segment.box.x1
        local_y1 = tightened_box.y1 - segment.box.y1
        local_x2 = local_x1 + crop.shape[1]
        local_y2 = local_y1 + crop.shape[0]
        tightened_mask = mask[local_y1:local_y2, local_x1:local_x2]

        score = float(cv.countNonZero(tightened_mask))
        return CharacterSegment(
            crop=crop,
            box=tightened_box,
            mask=tightened_mask,
            polarity=segment.polarity,
            score=score,
        )
