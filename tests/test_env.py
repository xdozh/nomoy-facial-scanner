"""Narrowly scoped ENV unit tests.

Run with: python -m unittest tests.test_env
"""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_PATH = os.path.join(ROOT, "models.lock")
ENV_REPORT_PATH = os.path.join(ROOT, "env_report.json")


class TestEnv(unittest.TestCase):
    def test_models_lock_schema(self):
        self.assertTrue(os.path.exists(LOCK_PATH), "models.lock must exist")
        with open(LOCK_PATH) as f:
            lock = json.load(f)
        self.assertIn("models", lock, "models.lock must contain a 'models' list")
        required_keys = {"path", "download_url", "source", "model_version", "sha256"}
        versions = []
        for entry in lock["models"]:
            missing = required_keys - set(entry.keys())
            self.assertFalse(missing, f"models.lock entry missing keys: {missing}")
            self.assertEqual(
                len(entry["sha256"]),
                64,
                f"SHA-256 must be 64 hex chars: {entry['model_version']}",
            )
            versions.append(entry["model_version"])
        self.assertIn("yolo11n", versions)
        self.assertIn("yolo11s", versions)
        self.assertIn("buffalo_l", versions)

    def test_verify_models_passes(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "verify_models.py")],
            cwd=ROOT,
        )
        self.assertEqual(
            proc.returncode,
            0,
            "verify_models.py must pass for pinned artifacts present in this workspace",
        )

    def test_env_report_fields(self):
        self.assertTrue(os.path.exists(ENV_REPORT_PATH), "env_report.json must exist")
        with open(ENV_REPORT_PATH) as f:
            report = json.load(f)
        required = {"os", "python", "cpu", "gpu", "onnxruntime_providers"}
        missing = required - set(report.keys())
        self.assertFalse(missing, f"env_report.json missing fields: {missing}")
        self.assertIn("onnxruntime_version", report)


if __name__ == "__main__":
    unittest.main()
