from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from one_click.config import resolve_config
from one_click.data_ingestion import validate_dataset
from one_click.orchestrator import run_one_click_training


class OneClickConfigTests(unittest.TestCase):
    def test_config_precedence_defaults_preset_file_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            class_dir = data_dir / "hello"
            class_dir.mkdir(parents=True)
            np.save(class_dir / "sample.npy", np.zeros((22, 80, 112), dtype=np.float32))

            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps({"epochs": 12, "batch_size": 7, "log_level": "DEBUG"}),
                encoding="utf-8",
            )

            cfg = resolve_config(
                {
                    "data_dir": str(data_dir),
                    "preset": "quick",
                    "config_file": str(config_path),
                    "batch_size": 4,
                }
            )
            self.assertEqual(cfg.epochs, 12)  # from config file
            self.assertEqual(cfg.batch_size, 4)  # CLI wins
            self.assertEqual(cfg.log_level, "DEBUG")  # from config file

    def test_config_file_preset_applies_when_cli_preset_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            class_dir = data_dir / "hello"
            class_dir.mkdir(parents=True)
            np.save(class_dir / "sample.npy", np.zeros((22, 80, 112), dtype=np.float32))

            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"preset": "quick"}), encoding="utf-8")

            cfg = resolve_config({"data_dir": str(data_dir), "config_file": str(config_path)})
            self.assertEqual(cfg.preset, "quick")
            self.assertEqual(cfg.epochs, 5)
            self.assertEqual(cfg.batch_size, 8)


class OneClickDataValidationTests(unittest.TestCase):
    def test_dataset_validation_reports_skipped_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            class_dir = data_dir / "word"
            class_dir.mkdir(parents=True)
            np.save(class_dir / "good.npy", np.zeros((22, 80, 112), dtype=np.float32))
            (class_dir / "bad.txt").write_text("invalid", encoding="utf-8")

            summary = validate_dataset(data_dir)
            self.assertEqual(summary.layout, "processed_npy")
            self.assertEqual(summary.class_count, 1)
            self.assertEqual(summary.files_per_class["word"], 1)
            self.assertTrue(any(path.endswith("bad.txt") for path in summary.skipped_files))

    def test_hidden_class_directories_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            hidden_dir = data_dir / ".ipynb_checkpoints"
            hidden_dir.mkdir(parents=True)
            class_dir = data_dir / "word"
            class_dir.mkdir(parents=True)
            np.save(class_dir / "good.npy", np.zeros((22, 80, 112), dtype=np.float32))

            summary = validate_dataset(data_dir)
            self.assertEqual(summary.class_count, 1)
            self.assertIn("word", summary.files_per_class)


class OneClickOrchestrationTests(unittest.TestCase):
    def test_dry_run_generates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            class_dir = data_dir / "word"
            class_dir.mkdir(parents=True)
            np.save(class_dir / "sample.npy", np.zeros((22, 80, 112), dtype=np.float32))

            output_dir = Path(tmp) / "out"
            cfg = resolve_config(
                {
                    "data_dir": str(data_dir),
                    "output_dir": str(output_dir),
                    "dry_run": True,
                    "enable_processing": False,
                }
            )

            result = run_one_click_training(cfg)
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["status"], "ok")
            self.assertTrue((output_dir / "effective_config.json").exists())
            self.assertTrue((output_dir / "run_metadata.json").exists())

    def test_cli_json_error_output(self) -> None:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_one_click.py"),
            "--data_dir",
            "/path/does/not/exist",
            "--json-errors",
        ]
        completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads(completed.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_class"], "ValidationError")


if __name__ == "__main__":
    unittest.main()
