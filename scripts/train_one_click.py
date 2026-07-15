#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from one_click.config import resolve_config
from one_click.orchestrator import run_one_click_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click LipReader training workflow")
    parser.add_argument("--data_dir", required=True, help="Input dataset directory")
    parser.add_argument("--output_dir", default=None, help="Output directory for logs/artifacts")
    parser.add_argument("--preset", default=None, choices=["quick", "default", "high_quality"], help="Training preset")
    parser.add_argument("--config_file", default=None, help="Optional config file (.json/.yml/.yaml)")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print plan only")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--log_level", default=None, choices=["INFO", "DEBUG"], help="Console/file log level")
    parser.add_argument("--disable_processing", action="store_true", help="Skip preprocessing pipeline")
    parser.add_argument("--no_cache_processed", action="store_true", help="Disable processed output cache")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cli_overrides = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "preset": args.preset,
        "config_file": args.config_file,
        "dry_run": args.dry_run,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "log_level": args.log_level,
        "enable_processing": not args.disable_processing if args.disable_processing else None,
        "cache_processed": False if args.no_cache_processed else None,
    }
    try:
        cfg = resolve_config(cli_overrides)
        print("Resolved configuration:")
        print(json.dumps(asdict(cfg), indent=2, sort_keys=True))
        result = run_one_click_training(cfg)
        print("\nRun completed:")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
