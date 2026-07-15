from __future__ import annotations

import inspect
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

# ---------------------------------------------------------------------------
# Domain Exceptions (OC-002)
# ---------------------------------------------------------------------------
class OneClickError(Exception):
    """Base exception for One-Click training pipeline."""
    pass

class ConfigError(OneClickError):
    """Raised when configuration is invalid or missing."""
    pass

class ValidationError(OneClickError):
    """Raised when dataset or input validation fails."""
    pass

class TrainingError(OneClickError):
    """Raised when the model training process fails."""
    pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _redact_paths(data: Any, root_path: str) -> Any:
    """
    Recursively sanitizes absolute file paths from metadata (OC-008).
    """
    if isinstance(data, str):
        return data.replace(str(root_path), "<WORKSPACE_ROOT>")
    elif isinstance(data, dict):
        return {k: _redact_paths(v, root_path) for k, v in data.items()}
    elif isinstance(data, list):
        return [_redact_paths(item, root_path) for item in data]
    return data


def _train_model_adapter(data_dir: str, epochs: int, batch_size: int) -> tuple:
    """
    Adapter boundary for external training module (OC-006).
    Ensures safe imports (OC-001) and validates module signatures.
    """
    try:
        # Module-safe import replacing dynamic sys.path mutation
        from src.train_mobilenet_LSTM import train_model
    except ImportError as e:
        raise ConfigError(f"Failed to import training module. Ensure you are running as a module (-m): {e}") from e

    # Signature validation guard
    sig = inspect.signature(train_model)
    required_params = {"data_dir", "epochs", "batch_size"}
    if not required_params.issubset(set(sig.parameters.keys())):
        raise ConfigError(f"train_model signature incompatible. Expected {required_params}, got {list(sig.parameters.keys())}")

    try:
        return train_model(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
    except Exception as e:
        raise TrainingError(f"Training failed in adapter layer: {e}") from e


def _run_training(processed_dir: Path, cfg: OneClickConfig, logger) -> Dict[str, Any]:
    model, history, word_labels = _train_model_adapter(
        data_dir=str(processed_dir),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size
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
    repo_root = Path(__file__).resolve().parents[2]
    
    logger, log_path = setup_logging(output_dir, cfg.log_level)
    _set_seed(cfg.seed)

    run_started_at = datetime.now(timezone.utc).isoformat()
    logger.info("Starting one-click training")
    logger.info("Selected preset: %s", cfg.preset)

    effective_config = asdict(cfg)
    write_json(output_dir / "effective_config.json", _redact_paths(effective_config, str(repo_root)))
    logger.debug("Effective config: %s", effective_config)

    try:
        data_dir = Path(cfg.data_dir)
        try:
            dataset_summary = validate_dataset(data_dir)
        except Exception as e:
            raise ValidationError(f"Dataset validation failed: {e}") from e

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
                # OC-005: Enforce dry-run contract
                logger.info("[DRY RUN] Bypassing processing phase. Artifacts will not be generated in %s", processed_dir)
                training_data_dir = processed_dir
            else:
                try:
                    training_data_dir = prepare_processed_dataset(data_dir, processed_dir, dataset_summary.layout, cfg, logger)
                except Exception as e:
                    raise ValidationError(f"Data processing failed: {e}") from e
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
            logger.info("[DRY RUN] Bypassing training stage. No model checkpoints will be written.")
            result["planned_training_data_dir"] = str(training_data_dir)
            result["dry_run"] = True
            
            # OC-008: Redact paths before export
            redacted_result = _redact_paths(result, str(repo_root))
            write_json(output_dir / "run_metadata.json", redacted_result)
            return redacted_result

        logger.info("Starting training stage")
        result["training"] = _run_training(Path(training_data_dir), cfg, logger)
        result["run_finished_at"] = datetime.now(timezone.utc).isoformat()
        
        redacted_result = _redact_paths(result, str(repo_root))
        write_json(output_dir / "run_metadata.json", redacted_result)
        return redacted_result

    except OneClickError as exc:
        # OC-002: Preserve class-specific domain handling
        error_type = type(exc).__name__
        logger.error(f"{error_type} occurred: {exc}")
        error_result = {
            "status": "error",
            "error_type": error_type,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "log_file": str(log_path),
        }
        write_json(output_dir / "run_metadata.json", _redact_paths(error_result, str(repo_root)))
        raise

    except Exception as exc:
        # Fallback for unexpected systemic errors
        logger.exception("Unexpected system failure during one-click training")
        error_result = {
            "status": "error",
            "error_type": "SystemError",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_started_at": run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "log_file": str(log_path),
        }
        write_json(output_dir / "run_metadata.json", _redact_paths(error_result, str(repo_root)))
        raise TrainingError(f"Unexpected failure: {exc}") from exc
