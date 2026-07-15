from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PROCESSED_EXTENSIONS = {".npy"}


@dataclass
class DatasetSummary:
    layout: str
    class_count: int
    files_per_class: Dict[str, int]
    skipped_files: List[str] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return sum(self.files_per_class.values())


def _class_dirs(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def detect_layout(data_dir: Path) -> str:
    classes = _class_dirs(data_dir)
    if not classes:
        raise ValueError(f"No class folders found in {data_dir}")

    has_npy = any(any(child.suffix.lower() in PROCESSED_EXTENSIONS for child in cls.iterdir()) for cls in classes)
    if has_npy:
        return "processed_npy"

    has_take_dir = any(any(child.is_dir() for child in cls.iterdir()) for cls in classes)
    if has_take_dir:
        return "class_takes_frames"
    return "class_media"


def validate_dataset(data_dir: Path) -> DatasetSummary:
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

    layout = detect_layout(data_dir)
    files_per_class: Dict[str, int] = {}
    skipped_files: List[str] = []

    for class_dir in _class_dirs(data_dir):
        valid_count = 0
        if layout == "processed_npy":
            for file in sorted(class_dir.iterdir()):
                if file.is_file() and file.suffix.lower() in PROCESSED_EXTENSIONS:
                    valid_count += 1
                elif file.is_file():
                    skipped_files.append(str(file))
        elif layout == "class_takes_frames":
            for take_dir in sorted(class_dir.iterdir()):
                if not take_dir.is_dir():
                    if take_dir.is_file():
                        skipped_files.append(str(take_dir))
                    continue
                has_any_frame = False
                for frame in sorted(take_dir.iterdir()):
                    if frame.is_file() and frame.suffix.lower() in IMAGE_EXTENSIONS:
                        has_any_frame = True
                    elif frame.is_file():
                        skipped_files.append(str(frame))
                if has_any_frame:
                    valid_count += 1
        else:
            for file in sorted(class_dir.iterdir()):
                if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS.union(IMAGE_EXTENSIONS):
                    valid_count += 1
                elif file.is_file():
                    skipped_files.append(str(file))

        if valid_count == 0:
            raise ValueError(f"Class '{class_dir.name}' has no valid files for layout '{layout}'")
        files_per_class[class_dir.name] = valid_count

    return DatasetSummary(layout=layout, class_count=len(files_per_class), files_per_class=files_per_class, skipped_files=skipped_files)
