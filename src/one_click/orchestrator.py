from __future__ import annotations

import random
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import OneClickConfig
from .data_ingestion import validate_dataset
from .logging_utils import setup_logging, write_json
from .processing import prepare_processed_dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def _run_training(processed_dir: Path, cfg: OneClickConfig, logger) -> Dict[str, Any]:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from train_mobilenet_LSTM import train_model

    model, history, word_labels = train_model(
        data_dir=str(processed_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
    )
    logger.info("Training finished successfully")
    return {
        "num_classes": len(word_labels),
        "epochs_ran": len(history.history.get("loss", [])),
        "labels": word_labels,
        "model_name": getattr(model, "name", "unknown"),
    }


def run_one_click_training(cfg: OneClickConfig) -> Dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    logger, log_path = setup_logging(output_dir, cfg.log_level)
    _set_seed(cfg.seed)

    run_started_at = datetime.now(timezone.utc).isoformat()
    logger.info("Starting one-click training")
    logger.info("Selected preset: %s", cfg.preset)

    effective_config = asdict(cfg)
    write_json(output_dir / "effective_config.json", effective_config)
    logger.debug("Effective config: %s", effective_config)

    try:
        data_dir = Path(cfg.data_dir)
        dataset_summary = validate_dataset(data_dir)
        logger.info(
            "Dataset summary: layout=%s classes=%d samples=%d skipped=%d",
            dataset_summary.layout,
            dataset_summary.class_count,
            dataset_summary.total_files,
            len(dataset_summary.skipped_files),
        )

        processed_dir = output_dir / "processed_data"
        training_data_dir = data_dir
        if cfg.enable_processing:
            if cfg.dry_run:
                logger.info("[DRY RUN] Would run processing into %s", processed_dir)
                training_data_dir = processed_dir
            else:
                training_data_dir = prepare_processed_dataset(data_dir, processed_dir, dataset_summary.layout, cfg, logger)
        else:
            logger.info("Processing disabled; expecting preprocessed input")

        result: Dict[str, Any] = {
            "status": "ok",
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "seed": cfg.seed,
            "preset": cfg.preset,
            "log_file": str(log_path),
            "dataset_summary": {
                "layout": dataset_summary.layout,
                "class_count": dataset_summary.class_count,
                "files_per_class": dataset_summary.files_per_class,
                "skipped_files": dataset_summary.skipped_files,
            },
        }

        if cfg.dry_run:
            logger.info("[DRY RUN] Would train model from %s", training_data_dir)
            result["planned_training_data_dir"] = str(training_data_dir)
            result["dry_run"] = True
            write_json(output_dir / "run_metadata.json", result)
            return result

        logger.info("Starting training stage")
        result["training"] = _run_training(Path(training_data_dir), cfg, logger)
        result["run_finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "run_metadata.json", result)
        return result
    except Exception as exc:
        logger.exception("One-click training failed")
        error_result = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "log_file": str(log_path),
        }
        write_json(output_dir / "run_metadata.json", error_result)
        raise RuntimeError(f"One-click training failed: {exc}. See log: {log_path}") from exc
