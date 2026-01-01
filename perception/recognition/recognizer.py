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

"""Utilities for running the character CNN recogniser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch

from ..config import RecognitionConfig
from .model import CharacterCNN
from .preprocess import preprocess_character


@dataclass(slots=True)
class Prediction:
    character: str
    confidence: float
    probabilities: np.ndarray


@dataclass(slots=True)
class _ImageFeatures:
    """Pre-computed data for a character crop used during prediction."""

    processed: np.ndarray
    width_ratio: float
    hole_count: int
    fill_ratio: float
    vertical_balance: float


class CharacterRecognizer:
    def __init__(self, config: RecognitionConfig) -> None:
        if not config.weights_path.exists():
            raise FileNotFoundError(f"Recognizer weights not found: {config.weights_path}")
        self.config = config
        self.charset = list(config.charset)
        self.index_to_char = {idx: char for idx, char in enumerate(self.charset)}
        self.char_to_index = {char: idx for idx, char in enumerate(self.charset)}
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CharacterCNN(len(self.charset))
        state = torch.load(config.weights_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.secondary_model: Optional[CharacterCNN] = None
        if config.secondary_weights and config.secondary_weights.exists():
            self.secondary_model = CharacterCNN(len(self.charset))
            secondary_state = torch.load(config.secondary_weights, map_location="cpu")
            if isinstance(secondary_state, dict) and "model_state_dict" in secondary_state:
                secondary_state = secondary_state["model_state_dict"]
            self.secondary_model.load_state_dict(secondary_state)
            self.secondary_model.to(self.device)
            self.secondary_model.eval()
        self.templates = self._build_templates(self.config.image_size)

    def predict(self, images: Iterable[np.ndarray]) -> List[Prediction]:
        image_list = list(images)
        if not image_list:
            return []

        feature_list = [self._prepare_features(image) for image in image_list]
        tensor_inputs: List[torch.Tensor] = []
        tta_groups: List[List[int]] = []
        use_tta = bool(self.config.use_tta)

        def _tta_views(processed: np.ndarray) -> List[np.ndarray]:
            if not use_tta:
                return [processed]
            variants = [processed]
            variants.append(np.pad(processed, ((0, 0), (1, 0)), mode='edge')[:, :-1])  # shift left
            variants.append(np.pad(processed, ((0, 0), (0, 1)), mode='edge')[:, 1:])   # shift right
            variants.append(np.pad(processed, ((1, 0), (0, 0)), mode='edge')[:-1, :])  # shift up
            variants.append(np.pad(processed, ((0, 1), (0, 0)), mode='edge')[1:, :])   # shift down
            return variants

        for feature in feature_list:
            group: List[int] = []
            for view in _tta_views(feature.processed):
                tensor_inputs.append(torch.from_numpy(view.astype(np.float32)).unsqueeze(0))
                group.append(len(tensor_inputs) - 1)
            tta_groups.append(group)
        batch = torch.stack(tensor_inputs).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            if self.secondary_model is not None:
                logits = 0.6 * logits + 0.4 * self.secondary_model(batch)
            logits = logits / max(1e-5, self.config.temperature)
            probs = torch.softmax(logits, dim=1)
        if use_tta:
            aggregated_probs: List[torch.Tensor] = []
            for group in tta_groups:
                aggregated_probs.append(probs[group, :].mean(dim=0))
            probs = torch.stack(aggregated_probs, dim=0)
        predictions: List[Prediction] = []
        smoothing = float(self.config.smoothing)
        if smoothing > 0.0:
            probs = (1 - smoothing) * probs + smoothing / probs.size(1)
        threshold = float(self.config.confidence_threshold)
        for features, prob in zip(feature_list, probs):
            prob_array = prob.cpu().numpy()
            template_scores = self._template_scores(features.processed)
            # Blend learned logits with template matching for robustness on rare glyphs.
            combined = 0.65 * prob_array + 0.35 * template_scores
            idx = int(np.argmax(combined))
            confidence = float(combined[idx])
            width_ratio = features.width_ratio
            hole_count = features.hole_count
            fill_ratio = features.fill_ratio
            vertical_balance = features.vertical_balance
            heuristic_floor = 0.75
            if confidence < heuristic_floor:
                if hole_count >= 2 and fill_ratio < 0.35:
                    idx = self.char_to_index.get('6', idx)
                    confidence = max(confidence, 0.6)
                elif (
                    hole_count == 1
                    and width_ratio <= 0.6
                    and fill_ratio >= 0.5
                    and self.index_to_char[idx] not in {'0', '6'}
                ):
                    idx = self.char_to_index.get('6', idx)
                    confidence = max(confidence, 0.55)
                elif hole_count >= 1 and fill_ratio < 0.45 and self.index_to_char[idx] not in {'0', '6', '8', '9'}:
                    idx = self._best_index(combined, '0698', idx)
                elif hole_count >= 1:
                    six_idx = self.char_to_index.get('6')
                    if six_idx is not None and six_idx != idx:
                        six_template = float(template_scores[six_idx])
                        current_template = float(template_scores[idx])
                        if fill_ratio >= 0.35 and six_template >= current_template + 0.01 and combined[six_idx] >= 0.05:
                            idx = six_idx
                            confidence = max(confidence, combined[six_idx], six_template, 0.6)
                elif self.index_to_char[idx] == '5' and hole_count >= 1:
                    idx = self._best_index(combined, '69', idx)
                elif (
                    self.index_to_char[idx] == '1'
                    and width_ratio >= 0.48
                    and fill_ratio <= 0.28
                    and hole_count == 0
                ):
                    if vertical_balance >= 1.6:
                        two_idx = self.char_to_index.get('2')
                        if two_idx is not None:
                            idx = two_idx
                            confidence = max(confidence, float(combined[two_idx]), 0.5)
                elif (
                    width_ratio < 0.65
                    and fill_ratio < 0.25
                    and hole_count <= 1
                    and (width_ratio < 0.45 or vertical_balance >= 0.9)
                ):
                    one_idx = self.char_to_index.get('1')
                    if one_idx is not None:
                        current_score = float(combined[idx])
                        candidate_score = float(combined[one_idx])
                        template_boost = float(template_scores[one_idx]) - float(template_scores[idx])
                        if candidate_score >= current_score * 0.8 or template_boost >= 0.02:
                            idx = one_idx
                            confidence = max(confidence, candidate_score, 0.7)
                elif fill_ratio < 0.2 and hole_count == 0:
                    confidence = max(confidence, 0.5)
            character = self.index_to_char[idx]
            confidence = max(confidence, float(combined[idx]))
            if confidence < threshold:
                fallback_char, fallback_score = self._template_argmax(template_scores)
                if fallback_char is not None and fallback_score > confidence:
                    idx = self.char_to_index.get(fallback_char, idx)
                    character = self.index_to_char[idx]
                    confidence = fallback_score
            predictions.append(Prediction(character=character, confidence=confidence, probabilities=combined))
        return predictions

    def predict_text(self, images: Iterable[np.ndarray]) -> Tuple[str, List[Prediction]]:
        predictions = self.predict(images)
        threshold = float(self.config.confidence_threshold)
        characters = [pred.character if pred.confidence >= threshold else "?" for pred in predictions]
        return "".join(characters), predictions

    def _template_scores(self, processed: np.ndarray) -> np.ndarray:
        vector = processed.astype(np.float32).flatten()
        norm = float(np.linalg.norm(vector))
        scores = np.zeros(len(self.charset), dtype=np.float32)
        if norm < 1e-6:
            return scores
        for idx, (template_vector, template_norm) in self.templates.items():
            if template_norm < 1e-6:
                continue
            score = float(np.dot(vector, template_vector) / (norm * template_norm))
            if score > 0:
                scores[idx] = score
        total = scores.sum()
        if total > 0:
            scores /= total
        return scores

    def _template_argmax(self, scores: np.ndarray) -> Tuple[Optional[str], float]:
        if scores.size == 0:
            return None, 0.0
        idx = int(np.argmax(scores))
        if scores[idx] <= 0:
            return None, 0.0
        return self.index_to_char[idx], float(scores[idx])

    def _prepare_features(self, image: np.ndarray) -> _ImageFeatures:
        gray = self._to_gray(image)
        processed = preprocess_character(gray, self.config.image_size)
        width_ratio, hole_count, fill_ratio, vertical_balance = self._geometry_features(gray)
        return _ImageFeatures(
            processed=processed.astype(np.float32),
            width_ratio=width_ratio,
            hole_count=hole_count,
            fill_ratio=fill_ratio,
            vertical_balance=vertical_balance,
        )

    def _geometry_features(self, gray: np.ndarray) -> Tuple[float, int, float, float]:
        height, width = gray.shape[:2]
        width_ratio = width / float(max(1, height))
        blurred = cv.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        binary_for_vertical = binary.copy()
        fill_ratio = float(cv.countNonZero(binary)) / max(1, binary.size)
        if fill_ratio > 0.65:
            binary = cv.bitwise_not(binary)
            fill_ratio = 1.0 - fill_ratio
        _, hierarchy = cv.findContours(binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        holes = 0
        if hierarchy is not None:
            for entry in hierarchy[0]:
                if entry[3] != -1:
                    holes += 1
        vertical_balance = self._vertical_balance(binary_for_vertical)
        return width_ratio, holes, fill_ratio, vertical_balance

    def _vertical_balance(self, binary: np.ndarray) -> float:
        height = binary.shape[0]
        if height <= 1:
            return 1.0
        midpoint = height // 2
        top = float(np.count_nonzero(binary[:midpoint, :]))
        bottom = float(np.count_nonzero(binary[midpoint:, :]))
        if bottom <= 0:
            return 0.0
        return bottom / max(top, 1.0)

    def _best_index(self, scores: np.ndarray, chars: str, default_idx: int) -> int:
        best_idx = default_idx
        best_score = float(scores[default_idx])
        for char in chars:
            idx = self.char_to_index.get(char)
            if idx is None:
                continue
            score = float(scores[idx])
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def _build_templates(self, image_size: int) -> Dict[int, Tuple[np.ndarray, float]]:
        templates: Dict[int, Tuple[np.ndarray, float]] = {}
        for idx, char in enumerate(self.charset):
            canvas = np.zeros((image_size, image_size), dtype=np.uint8)
            cv.putText(
                canvas,
                char,
                (3, image_size - 4),
                cv.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255,),
                thickness=2,
                lineType=cv.LINE_AA,
            )
            processed = preprocess_character(canvas, image_size)
            vector = processed.astype(np.float32).flatten()
            templates[idx] = (vector, float(np.linalg.norm(vector)))
        return templates

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
