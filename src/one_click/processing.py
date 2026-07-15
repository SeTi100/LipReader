from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from .config import OneClickConfig
from .data_ingestion import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def _load_video_frames(path: Path, target_frames: int) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    frames: List[np.ndarray] = []
    while len(frames) < target_frames:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video file: {path}")
    while len(frames) < target_frames:
        frames.append(frames[-1].copy())
    return frames[:target_frames]


def _load_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image file: {path}")
    return image


def _process_frame(frame: np.ndarray, cfg: OneClickConfig) -> np.ndarray:
    import cv2

    output = frame
    if cfg.enable_grayscale:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if cfg.enable_resize:
        output = cv2.resize(output, (cfg.target_width, cfg.target_height))
    if cfg.enable_normalize:
        output = output.astype(np.float32) / 255.0
    else:
        output = output.astype(np.float32)
    return output


def prepare_processed_dataset(data_dir: Path, processed_dir: Path, layout: str, cfg: OneClickConfig, logger: logging.Logger) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processing dataset: layout=%s", layout)

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        target_class_dir = processed_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        if layout == "processed_npy":
            logger.info("Detected already-processed input for class '%s'; reusing files", class_dir.name)
            for npy_file in sorted(class_dir.glob("*.npy")):
                out_file = target_class_dir / npy_file.name
                if out_file.exists() and cfg.cache_processed:
                    continue
                array = np.load(npy_file)
                np.save(out_file, array)
            continue

        if layout == "class_takes_frames":
            for take_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                out_file = target_class_dir / f"{take_dir.name}.npy"
                if out_file.exists() and cfg.cache_processed:
                    logger.debug("Skipping cached take: %s", out_file)
                    continue
                frame_paths = sorted([f for f in take_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])
                if not frame_paths:
                    logger.warning("Skipping take without valid frames: %s", take_dir)
                    continue
                processed = [_process_frame(_load_image(p), cfg) for p in frame_paths[: cfg.target_frames]]
                while len(processed) < cfg.target_frames:
                    processed.append(processed[-1].copy())
                np.save(out_file, np.stack(processed, axis=0))
            continue

        for media_file in sorted(p for p in class_dir.iterdir() if p.is_file()):
            out_file = target_class_dir / f"{media_file.stem}.npy"
            if out_file.exists() and cfg.cache_processed:
                logger.debug("Skipping cached sample: %s", out_file)
                continue
            try:
                if media_file.suffix.lower() in VIDEO_EXTENSIONS:
                    frames = _load_video_frames(media_file, cfg.target_frames)
                elif media_file.suffix.lower() in IMAGE_EXTENSIONS:
                    image = _load_image(media_file)
                    frames = [image.copy() for _ in range(cfg.target_frames)]
                else:
                    logger.warning("Skipping unsupported file: %s", media_file)
                    continue
                processed_frames = [_process_frame(frame, cfg) for frame in frames]
                np.save(out_file, np.stack(processed_frames, axis=0))
            except Exception:
                logger.exception("Failed processing file: %s", media_file)

    return processed_dir
