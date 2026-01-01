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

"""Vision pipeline utilities for the building-number assignment."""

from .config import (
    AssignmentConfig,
    DetectionConfig,
    SegmentationConfig,
    RecognitionConfig,
    load_assignment_config,
    load_config_overrides_from_file,
)
from .pipeline import BuildingNumberPipeline
from .domain import KNOWN_BUILDING_NUMBERS, is_known_building_number, normalise_building_text, iter_known_building_numbers

__all__ = [
    "AssignmentConfig",
    "DetectionConfig",
    "SegmentationConfig",
    "RecognitionConfig",
    "BuildingNumberPipeline",
    "load_assignment_config",
    "load_config_overrides_from_file",
    "KNOWN_BUILDING_NUMBERS",
    "is_known_building_number",
    "normalise_building_text",
    "iter_known_building_numbers",
]
