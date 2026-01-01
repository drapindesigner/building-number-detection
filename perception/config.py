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

"""Configuration helpers for the assignment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from .utils import set_global_seed


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value.strip()
    return value.split("#", 1)[0].strip()


def _parse_override_value(value: str, key: str = "") -> object:
    text = _strip_inline_comment(value)
    if not text:
        return ""
    if "charset" in key.lower():
        return text
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(sep in text for sep in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_config_overrides_from_file(path: Union[str, Path], *, allow_missing: bool = False) -> Dict[str, object]:
    """Parse a minimal ``key: value`` override file (no JSON required)."""

    file_path = Path(path)
    if not file_path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Config file not found: {file_path}")

    overrides: Dict[str, object] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            overrides[key] = _parse_override_value(value, key)
    return overrides


@dataclass
class DetectionConfig:
    weights_path: Path = Path("data/detector/best.pt")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 5
    image_size: int = 640
    device: str = ""
    crop_expand: float = 1.1
    use_augmentation: bool = False
    fallback_image_sizes: Tuple[int, ...] = (832, 960)
    min_return_confidence: float = 0.55


@dataclass
class SegmentationConfig:
    min_area: int = 60
    max_area_ratio: float = 0.55
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 1.4
    min_height_ratio: float = 0.35
    max_height_ratio: float = 1.15
    padding: int = 3
    binarization: str = "adaptive"
    gaussian_ksize: int = 5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    duplicate_iou: float = 0.4
    min_relative_area: float = 0.35
    merge_gap_px: int = 6
    merge_overlap_px: int = 12
    min_width_ratio: float = 0.45
    split_aspect_ratio: float = 1.55
    split_gap_fraction: float = 0.05
    split_min_width_fraction: float = 0.08
    deskew_enabled: bool = True
    deskew_min_angle: float = 4.0
    deskew_max_angle: float = 40.0
    mode: str = "heuristic"
    model_weights: Path = Path("data/segmenter/best.pt")
    model_confidence: float = 0.25
    model_iou: float = 0.45
    model_image_size: int = 384
    model_device: str = ""
    model_max_det: int = 6
    model_min_return_confidence: float = 0.45
    model_use_augmentation: bool = False


@dataclass
class RecognitionConfig:
    weights_path: Path = Path("data/recognizer/best.pt")
    device: str = ""
    image_size: int = 32
    charset: str = "0123456789ABCD"  # Support optional trailing letter per specification
    confidence_threshold: float = 0.4
    temperature: float = 1.0
    smoothing: float = 0.0
    use_tta: bool = True
    secondary_weights: Optional[Path] = None


@dataclass
class AssignmentConfig:
    output_root: Path = Path("output")
    task1_dirname: str = "task1"
    task2_dirname: str = "task2"
    task3_dirname: str = "task3"
    task4_dirname: str = "task4"
    seed: Optional[int] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)

    def task_output_dir(self, task: str) -> Path:
        mapping = {
            "task1": self.task1_dirname,
            "task2": self.task2_dirname,
            "task3": self.task3_dirname,
            "task4": self.task4_dirname,
        }
        subdir = mapping.get(task, task)
        return self.output_root / subdir

def _pop_first(keys: Iterable[str], source: Dict[str, object], default: object) -> Any:
    for key in keys:
        if key in source:
            return source.pop(key)
    return default


def _ensure_path(value: object, base_path: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_path / path
    return path


def _ensure_tuple_of_ints(value: object) -> Tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if value is None:
        return tuple()
    text = str(value).replace(",", " ")
    numbers = [token for token in text.split() if token]
    return tuple(int(token) for token in numbers)


def load_assignment_config(config_dict: Optional[Dict[str, object]], base_path: Optional[Path] = None) -> AssignmentConfig:
    data = dict(config_dict or {})
    base = Path(base_path or Path.cwd())

    seed_value = _pop_first(["seed", "random_seed"], data, None)

    output_root = _ensure_path(_pop_first(["output_root", "output_dir"], data, "output"), base)
    task1_dirname = str(_pop_first(["task1_output", "task1_dir"], data, "task1"))
    task2_dirname = str(_pop_first(["task2_output", "task2_dir"], data, "task2"))
    task3_dirname = str(_pop_first(["task3_output", "task3_dir"], data, "task3"))
    task4_dirname = str(_pop_first(["task4_output", "task4_dir"], data, "task4"))

    detection = DetectionConfig(
        weights_path=_ensure_path(_pop_first(["det_weights", "detection_weights"], data, "data/detector/best.pt"), base),
        confidence_threshold=float(_pop_first(["det_conf", "detector_confidence"], data, 0.25)),
        iou_threshold=float(_pop_first(["det_iou", "detector_iou"], data, 0.45)),
        max_det=int(_pop_first(["det_max_det", "detector_max"], data, 5)),
        image_size=int(_pop_first(["det_img", "detector_img"], data, 640)),
        device=str(_pop_first(["det_device", "detector_device"], data, "")),
        crop_expand=float(_pop_first(["det_crop_expand", "detector_expand"], data, 1.1)),
        use_augmentation=bool(_pop_first(["det_use_aug", "detector_use_augmentation"], data, False)),
        fallback_image_sizes=_ensure_tuple_of_ints(_pop_first(["det_fallback_sizes", "detector_fallback_sizes"], data, ())),
        min_return_confidence=float(_pop_first(["det_min_return_conf", "detector_min_return_conf"], data, 0.55)),
    )

    if not detection.fallback_image_sizes:
        detection.fallback_image_sizes = (832, 960)

    segmentation = SegmentationConfig(
        min_area=int(_pop_first(["seg_min_area"], data, 60)),
        max_area_ratio=float(_pop_first(["seg_max_area_ratio"], data, 0.55)),
        min_aspect_ratio=float(_pop_first(["seg_min_aspect"], data, 0.2)),
        max_aspect_ratio=float(_pop_first(["seg_max_aspect"], data, 1.4)),
        min_height_ratio=float(_pop_first(["seg_min_height_ratio"], data, 0.35)),
        max_height_ratio=float(_pop_first(["seg_max_height_ratio"], data, 1.15)),
        padding=int(_pop_first(["seg_padding"], data, 3)),
        binarization=str(_pop_first(["seg_binarization"], data, "adaptive")),
        gaussian_ksize=int(_pop_first(["seg_gaussian_ksize"], data, 5)),
        clahe_clip=float(_pop_first(["seg_clahe_clip"], data, 2.0)),
        clahe_grid=int(_pop_first(["seg_clahe_grid"], data, 8)),
        duplicate_iou=float(_pop_first(["seg_duplicate_iou"], data, 0.4)),
        min_relative_area=float(_pop_first(["seg_min_relative_area"], data, 0.35)),
        merge_gap_px=int(_pop_first(["seg_merge_gap_px"], data, 6)),
        merge_overlap_px=int(_pop_first(["seg_merge_overlap_px"], data, 12)),
        min_width_ratio=float(_pop_first(["seg_min_width_ratio"], data, 0.45)),
        split_aspect_ratio=float(_pop_first(["seg_split_aspect_ratio"], data, 1.55)),
        split_gap_fraction=float(_pop_first(["seg_split_gap_fraction"], data, 0.05)),
        split_min_width_fraction=float(_pop_first(["seg_split_min_width_fraction"], data, 0.08)),
        deskew_enabled=bool(_pop_first(["seg_deskew", "seg_deskew_enabled"], data, True)),
        deskew_min_angle=float(_pop_first(["seg_deskew_min_angle"], data, 4.0)),
        deskew_max_angle=float(_pop_first(["seg_deskew_max_angle"], data, 40.0)),
        mode=str(_pop_first(["seg_mode", "seg_method"], data, "heuristic")),
        model_weights=_ensure_path(_pop_first(["seg_model_weights", "seg_weights", "seg_model"], data, "data/segmenter/best.pt"), base),
        model_confidence=float(_pop_first(["seg_model_conf", "seg_model_confidence"], data, 0.25)),
        model_iou=float(_pop_first(["seg_model_iou"], data, 0.45)),
        model_image_size=int(_pop_first(["seg_model_img", "seg_model_image_size"], data, 384)),
        model_device=str(_pop_first(["seg_model_device"], data, "")),
        model_max_det=int(_pop_first(["seg_model_max_det"], data, 6)),
        model_min_return_confidence=float(_pop_first(["seg_model_min_return_conf"], data, 0.45)),
        model_use_augmentation=bool(_pop_first(["seg_model_use_aug", "seg_model_use_augmentation"], data, False)),
    )

    secondary_weights_value = _pop_first(["rec_secondary_weights"], data, None)
    recognition = RecognitionConfig(
        weights_path=_ensure_path(_pop_first(["rec_weights", "recognizer_weights"], data, "data/recognizer/best.pt"), base),
        device=str(_pop_first(["rec_device", "recognizer_device"], data, "")),
        image_size=int(_pop_first(["rec_image_size"], data, 32)),
        charset=str(_pop_first(["rec_charset"], data, "0123456789")),
        confidence_threshold=float(_pop_first(["rec_confidence_threshold"], data, 0.4)),
        temperature=float(_pop_first(["rec_temperature"], data, 1.0)),
        smoothing=float(_pop_first(["rec_smoothing"], data, 0.0)),
        use_tta=bool(_pop_first(["rec_use_tta"], data, True)),
        secondary_weights=_ensure_path(secondary_weights_value, base) if secondary_weights_value else None,
    )

    if seed_value is not None:
        try:
            set_global_seed(int(seed_value))
        except Exception:
            set_global_seed(None)

    return AssignmentConfig(
        output_root=output_root,
        task1_dirname=task1_dirname,
        task2_dirname=task2_dirname,
        task3_dirname=task3_dirname,
        task4_dirname=task4_dirname,
        seed=int(seed_value) if seed_value is not None else None,
        detection=detection,
        segmentation=segmentation,
        recognition=recognition,
    )
