from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {"epochs": 5, "batch_size": 8, "log_level": "INFO"},
    "default": {"epochs": 20, "batch_size": 16, "log_level": "INFO"},
    "high_quality": {"epochs": 50, "batch_size": 16, "log_level": "DEBUG"},
}


@dataclass
class OneClickConfig:
    data_dir: str
    output_dir: str = "artifacts"
    preset: str = "default"
    config_file: Optional[str] = None
    dry_run: bool = False
    seed: int = 42
    epochs: int = 20
    batch_size: int = 16
    log_level: str = "INFO"
    cache_processed: bool = True
    enable_processing: bool = True
    enable_grayscale: bool = True
    enable_resize: bool = True
    enable_normalize: bool = True
    target_frames: int = 22
    target_height: int = 80
    target_width: int = 112


def _read_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in {".json"}:
        return json.loads(config_path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("YAML config requires PyYAML installed.") from exc
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raise ValueError("Unsupported config file type. Use JSON or YAML.")


def _clean_cli_overrides(cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in cli_overrides.items():
        if value is None:
            continue
        cleaned[key] = value
    return cleaned


def _validate_config(values: Dict[str, Any]) -> None:
    preset = values.get("preset", "default")
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(PRESETS.keys())}")
    if int(values["epochs"]) <= 0:
        raise ValueError("epochs must be > 0")
    if int(values["batch_size"]) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(values["target_frames"]) <= 0:
        raise ValueError("target_frames must be > 0")
    if int(values["target_height"]) <= 0 or int(values["target_width"]) <= 0:
        raise ValueError("target_height and target_width must be > 0")
    level = str(values.get("log_level", "INFO")).upper()
    if level not in {"INFO", "DEBUG"}:
        raise ValueError("log_level must be INFO or DEBUG")


def resolve_config(cli_overrides: Dict[str, Any]) -> OneClickConfig:
    """
    Resolve final config with precedence:
    defaults < preset < config file < CLI overrides.
    """
    base = asdict(OneClickConfig(data_dir=cli_overrides.get("data_dir", "")))
    cleaned_cli = _clean_cli_overrides(cli_overrides)
    preset_name = cleaned_cli.get("preset", base["preset"])
    file_values = _read_config_file(cleaned_cli.get("config_file"))

    merged = dict(base)
    merged.update(PRESETS.get(preset_name, {}))
    merged.update(file_values)
    merged.update(cleaned_cli)

    if not merged.get("data_dir"):
        raise ValueError("data_dir is required")

    merged["preset"] = str(merged["preset"])
    merged["log_level"] = str(merged["log_level"]).upper()
    _validate_config(merged)
    return OneClickConfig(**merged)
