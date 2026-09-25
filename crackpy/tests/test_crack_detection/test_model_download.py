import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import yaml

from crackpy.tests.test_crack_detection._model_download import (
    _download_artifact,
    _ensure_artifact,
    _main,
    _ModelArtifact,
    _prefetch_models,
)


class TestModelDownload(unittest.TestCase):

    def test_unit_and_integration_jobs_prefetch_and_cache_models(self):
        repository_root = Path(__file__).resolve().parents[3]
        with (repository_root / ".gitlab-ci.yml").open(encoding="utf-8") as file:
            pipeline = yaml.safe_load(file)

        cache_template = pipeline[".crack_detection_model_cache"]
        self.assertEqual(
            cache_template["cache"]["paths"],
            ["crackpy/crack_detection/models/"],
        )
        self.assertEqual(
            cache_template["cache"]["key"]["files"],
            ["crackpy/tests/test_crack_detection/_model_download.py"],
        )

        prefetch_command = "python -m crackpy.tests.test_crack_detection._model_download"
        for job_name in ("crack_detection", "test_crack_detection"):
            job = pipeline[job_name]
            self.assertEqual(job["extends"], ".crack_detection_model_cache")
            self.assertEqual(job["script"][0], prefetch_command)

    def test_cli_uses_the_requested_cache_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            with patch(
                "crackpy.tests.test_crack_detection._model_download._prefetch_models",
                return_value=(destination / "ParallelNets.pth", destination / "UNetPath.pth"),
            ) as prefetch_models:
                exit_code = _main(["--destination", str(destination)])

        self.assertEqual(exit_code, 0)
        prefetch_models.assert_called_once_with(destination)

    def test_prefetches_every_configured_model(self):
        payloads = {
            "https://example.test/first.pth": b"first model",
            "https://example.test/second.pth": b"second model",
        }
        artifacts = tuple(
            _ModelArtifact(
                filename=f"model-{index}.pth",
                url=url,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
            for index, (url, payload) in enumerate(payloads.items())
        )

        def opener(url, *, timeout):
            self.assertEqual(timeout, 120)
            return BytesIO(payloads[url])

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)

            downloaded = _prefetch_models(
                destination,
                artifacts=artifacts,
                attempts=3,
                retry_delay_seconds=0,
                opener=opener,
            )

            self.assertEqual(downloaded, tuple(destination / artifact.filename for artifact in artifacts))
            for artifact in artifacts:
                self.assertEqual((destination / artifact.filename).read_bytes(), payloads[artifact.url])

    def test_reuses_a_verified_cached_file_without_network_access(self):
        payload = b"verified model data"
        artifact = _ModelArtifact(
            filename="model.pth",
            url="https://example.test/model.pth",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        opener = Mock(side_effect=AssertionError("network access was not expected"))

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            target = destination / artifact.filename
            target.write_bytes(payload)

            result = _ensure_artifact(
                artifact,
                destination,
                attempts=3,
                retry_delay_seconds=0,
                opener=opener,
            )

            self.assertEqual(result, target)
            self.assertEqual(result.read_bytes(), payload)

        opener.assert_not_called()

    def test_retries_gateway_timeout_and_writes_verified_file(self):
        payload = b"verified model data"
        artifact = _ModelArtifact(
            filename="model.pth",
            url="https://example.test/model.pth",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        gateway_timeout = HTTPError(artifact.url, 504, "Gateway Time-out", {}, None)
        opener = Mock(side_effect=[gateway_timeout, gateway_timeout, BytesIO(payload)])
        sleep = Mock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)

            _download_artifact(
                artifact,
                destination,
                attempts=3,
                retry_delay_seconds=0,
                opener=opener,
                sleep=sleep,
            )

            self.assertEqual((destination / artifact.filename).read_bytes(), payload)

        self.assertEqual(opener.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
