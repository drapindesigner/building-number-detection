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

"""Utility script to remove generated assignment outputs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from assignment import read_config
except ImportError:
    read_config = None

from perception import AssignmentConfig, load_assignment_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove generated assignment outputs")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config.txt overrides (default: ./config.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    return parser.parse_args()


def _load_assignment_config(repo_root: Path, config_path: Path | None) -> AssignmentConfig:
    config_dict = {}
    if config_path is None:
        config_path = repo_root / "config.txt"
    if config_path.exists() and read_config is not None:
        config_dict = read_config(str(config_path))
    elif config_path.exists():
        raise RuntimeError("assignment.read_config is unavailable")
    return load_assignment_config(config_dict, base_path=repo_root)


def _gather_targets(output_root: Path) -> List[Path]:
    if not output_root.exists():
        return []
    targets: List[Path] = []
    for child in output_root.iterdir():
        targets.append(child)
    return targets


def _format_paths(paths: Iterable[Path]) -> str:
    return "\n".join(str(path) for path in paths)


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    assignment_cfg = _load_assignment_config(repo_root, args.config)
    output_root = assignment_cfg.output_root

    targets = _gather_targets(output_root)
    if not targets:
        print(f"Nothing to remove under {output_root}")
        return 0

    print(f"Preparing to remove {len(targets)} item(s) under {output_root}:")
    print(_format_paths(targets))

    if args.dry_run:
        print("Dry run requested; no files were removed.")
        return 0

    if not args.yes:
        response = input("Proceed? [y/N] ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted; no files were removed.")
            return 0

    for path in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as exc:
            print(f"Failed to remove {path}: {exc}", file=sys.stderr)
            return 1

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

