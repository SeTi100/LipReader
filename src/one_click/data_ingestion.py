from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PROCESSED_EXTENSIONS = {".npy"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


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
    return sorted(
        [
            p
            for p in data_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in {"__MACOSX"}
        ]
    )


def _count_class_files_for_layout(class_dir: Path, layout: str) -> int:
    if layout == "processed_npy":
        return sum(1 for file in class_dir.iterdir() if file.is_file() and file.suffix.lower() in PROCESSED_EXTENSIONS)
    if layout == "class_takes_frames":
        valid_takes = 0
        for take_dir in class_dir.iterdir():
            if not take_dir.is_dir():
                continue
            has_frame = any(
                frame.is_file() and frame.suffix.lower() in IMAGE_EXTENSIONS for frame in take_dir.iterdir()
            )
            if has_frame:
                valid_takes += 1
        return valid_takes
    return sum(1 for file in class_dir.iterdir() if file.is_file() and file.suffix.lower() in MEDIA_EXTENSIONS)


def detect_layout(data_dir: Path) -> str:
    classes = _class_dirs(data_dir)
    if not classes:
        raise ValueError(f"No class folders found in {data_dir}")
    layout_order = ("processed_npy", "class_takes_frames", "class_media")
    layout_scores = {}
    for layout in layout_order:
        per_class_counts = [_count_class_files_for_layout(class_dir, layout) for class_dir in classes]
        valid_classes = sum(1 for count in per_class_counts if count > 0)
        total_files = sum(per_class_counts)
        layout_scores[layout] = (valid_classes, total_files)

    best_layout = max(layout_order, key=lambda layout: layout_scores[layout])
    if layout_scores[best_layout] == (0, 0):
        raise ValueError(
            "No supported dataset files found. Expected one of: "
            "processed_npy (.npy), class_takes_frames (.png/.jpg/.jpeg/.bmp), "
            "class_media (.mp4/.avi/.mov/.mkv/.png/.jpg/.jpeg/.bmp)."
        )
    return best_layout


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
                if file.is_file() and file.suffix.lower() in MEDIA_EXTENSIONS:
                    valid_count += 1
                elif file.is_file():
                    skipped_files.append(str(file))

        if valid_count == 0:
            expected = {
                "processed_npy": ".npy files in class folder",
                "class_takes_frames": "take subfolders with image frames (.png/.jpg/.jpeg/.bmp)",
                "class_media": "media files (.mp4/.avi/.mov/.mkv/.png/.jpg/.jpeg/.bmp)",
            }[layout]
            raise ValueError(
                f"Class '{class_dir.name}' has no valid files for layout '{layout}'. "
                f"Expected: {expected}."
            )
        files_per_class[class_dir.name] = valid_count

    return DatasetSummary(layout=layout, class_count=len(files_per_class), files_per_class=files_per_class, skipped_files=skipped_files)
