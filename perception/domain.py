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

"""Domain knowledge helpers for Curtin building numbers."""

from __future__ import annotations

from typing import Iterable, Optional, Set


# Raw list provided (may contain duplicates/whitespace).
_BUILDING_NUMBER_VALUES = [
    "003",
    "004",
    "100",
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "109",
    "109",
    "110",
    "111",
    "112",
    "113",
    "115",
    "126",
    "129A",
    "129C",
    "200A",
    "200A",
    "200A",
    "200B",
    "201",
    "202",
    "203",
    "204",
    "205",
    "205",
    "206",
    "207",
    "208",
    "209",
    "210",
    "211",
    "212",
    "213",
    "215",
    "216",
    "217",
    "300",
    "301",
    "302",
    "303",
    "304",
    "305",
    "306",
    "307",
    "308",
    "309",
    "310",
    "311",
    "312",
    "314",
    "400",
    "401",
    "402",
    "403",
    "404",
    "405",
    "407",
    "408",
    "410",
    "418",
    "418",
    "420",
    "420",
    "431",
    "433",
    "500",
    "501",
    "502A",
    "502B",
    "502C",
    "510",
    "418",
    "418",
    "420",
    "420",
    "420",
    "420",
    "431",
    "431",
    "410",
    "418",
    "420",
    "431",
    "433",
    "100",
    "102",
    "106F",
    "109",
    "109",
    "200B",
    "200B",
    "108",
    "200A",
    "201",
    "204",
    "210",
    "213",
    "307",
    "401",
    "403",
    "405",
    "107",
    "111",
    "112",
    "113",
    "103",
    "106A",
    "106A",
    "106G",
    "104",
    "104",
    "105",
    "106A",
    "106C",
    "106C",
    "106G",
    "111",
    "200B",
    "204",
    "210",
    "210A",
    "200A",
    "208A",
    "418",
    "601",
    "602",
    "603",
    "604",
    "605",
    "609",
    "610",
    "610",
    "610",
    "611",
    "612",
    "613",
    "614",
    "616",
    "619",
    "620",
    "114",
    "116",
    "118",
    "119",
    "128",
    "135",
    "136",
    "431",
    "433",
]

# Map of ambiguous glyphs frequently mis-recognised as numeric characters.
_DIGIT_LOOKALIKE = {
    "O": ["0"],
    "Q": ["0"],
    "D": ["0"],
    "I": ["1"],
    "L": ["1"],
    "T": ["1"],
    "Z": ["2"],
    "S": ["5"],
    "G": ["6"],
    "B": ["8"],
}


def _canonicalise_source(value: str) -> Optional[str]:
    cleaned = "".join(ch for ch in value if ch.isalnum())
    if not cleaned:
        return None
    cleaned = cleaned.upper()
    if cleaned[-1].isalpha() and cleaned[:-1].isdigit():
        return cleaned[:-1].zfill(3) + cleaned[-1]
    if cleaned.isdigit():
        return cleaned.zfill(3)
    return None


KNOWN_BUILDING_NUMBERS: Set[str] = {
    canonical
    for canonical in (_canonicalise_source(value) for value in _BUILDING_NUMBER_VALUES)
    if canonical
}


def _numeric_variants(text: str) -> Set[str]:
    options: list[list[str]] = []
    for ch in text:
        if ch.isdigit():
            options.append([ch])
            continue
        replacements = _DIGIT_LOOKALIKE.get(ch, [])
        if not replacements:
            return set()
        options.append(replacements)

    variants: Set[str] = set()
    def _build(idx: int, buffer: list[str]) -> None:
        if idx == len(options):
            variants.add("".join(buffer))
            return
        for candidate in options[idx]:
            buffer.append(candidate)
            _build(idx + 1, buffer)
            buffer.pop()

    _build(0, [])
    return variants


def _candidate_variants(value: str) -> Set[str]:
    cleaned = "".join(ch for ch in value if ch.isalnum())
    if not cleaned:
        return set()
    cleaned = cleaned.upper()
    variants: Set[str] = set()

    if cleaned[-1].isalpha():
        suffix = cleaned[-1]
        numeric_part = cleaned[:-1]
        if not numeric_part:
            return set()
        for numeric_variant in _numeric_variants(numeric_part):
            canonical = numeric_variant.zfill(3) + suffix
            variants.add(canonical)
    else:
        for numeric_variant in _numeric_variants(cleaned):
            variants.add(numeric_variant.zfill(3))
    return variants


def normalise_building_text(text: str) -> Optional[str]:
    """Return canonical building number if the candidate matches known values.

    The function tolerates common OCR confusions (O↔0, I↔1, etc.) and pads
    numeric identifiers to three digits. Mixed alphanumeric identifiers (e.g.
    ``200B``) retain the trailing letter once the numeric portion is resolved.
    """

    if not text:
        return None
    for variant in _candidate_variants(text):
        if variant in KNOWN_BUILDING_NUMBERS:
            return variant
    return None


def is_known_building_number(text: str) -> bool:
    """Return ``True`` if ``text`` can be normalised into a known building number."""

    return normalise_building_text(text) is not None


def iter_known_building_numbers() -> Iterable[str]:
    """Yield the sorted list of canonical building identifiers."""

    return sorted(KNOWN_BUILDING_NUMBERS)

