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

"""Shared preprocessing utilities for character images."""

from __future__ import annotations

import cv2 as cv
import numpy as np


def preprocess_character(gray: np.ndarray, image_size: int) -> np.ndarray:
    """Return a normalised square image in range [0, 1]."""
    if gray.ndim != 2:
        raise ValueError("Expected single-channel grayscale input")

    # Apply CLAHE for better contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Building number characters are typically light on dark background
    blurred = cv.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    coords = cv.findNonZero(binary)
    if coords is None:
        resized = cv.resize(enhanced, (image_size, image_size), interpolation=cv.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    x, y, w, h = cv.boundingRect(coords)
    digit_roi = enhanced[y : y + h, x : x + w]
    mask_roi = binary[y : y + h, x : x + w]

    side = max(w, h)
    pad_x_left = (side - w) // 2
    pad_x_right = side - w - pad_x_left
    pad_y_top = (side - h) // 2
    pad_y_bottom = side - h - pad_y_top

    digit_sq = cv.copyMakeBorder(digit_roi, pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, cv.BORDER_CONSTANT, value=(0,))
    mask_sq = cv.copyMakeBorder(mask_roi, pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, cv.BORDER_CONSTANT, value=(0,))

    resized_digit = cv.resize(digit_sq, (image_size, image_size), interpolation=cv.INTER_AREA)
    resized_mask = cv.resize(mask_sq, (image_size, image_size), interpolation=cv.INTER_AREA)

    normalized_digit = resized_digit.astype(np.float32) / 255.0
    mask = resized_mask.astype(np.float32) / 255.0

    digit_mask = mask > 0.5
    background_mask = mask <= 0.5
    digits_darker = False
    if np.any(digit_mask) and np.any(background_mask):
        digit_mean = float(normalized_digit[digit_mask].mean())
        background_mean = float(normalized_digit[background_mask].mean())
        digits_darker = digit_mean < background_mean

    normalized = normalized_digit * mask

    mean = float(normalized.mean())
    std = float(normalized.std())
    if std < 1e-6:
        std = 1.0
    normalized = (normalized - mean) / std
    normalized = np.clip(normalized * 0.5 + 0.5, 0.0, 1.0)
    if digits_darker:
        # Maintain a "light digits on dark background" polarity for the recogniser.
        normalized = 1.0 - normalized
    return normalized
