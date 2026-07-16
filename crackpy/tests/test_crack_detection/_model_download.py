"""Download and validate crack-detection model artifacts for CI."""

import argparse
import hashlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class _ModelArtifact:
    """Describe one immutable model artifact used by crack-detection tests.

    Attributes:
        filename: Name used for the verified local model file.
        url: Address of the immutable remote model file.
        sha256: Expected SHA-256 digest of the complete model file.
    """

    filename: str
    url: str
    sha256: str


_ZENODO_RECORD = "https://zenodo.org/record/7245516/files"
_MODEL_ARTIFACTS = (
    _ModelArtifact(
        filename="ParallelNets.pth",
        url=f"{_ZENODO_RECORD}/ParallelNets.pth?download=1",
        sha256="7b548e7299dbd647d35a99fb80f00b7582f040b58c62f5ca8be41e4c19c30f36",
    ),
    _ModelArtifact(
        filename="UNetPath.pth",
        url=f"{_ZENODO_RECORD}/UNetPath.pth?download=1",
        sha256="c8431c23541e3560236f8ba570d2854ecd5a729512338273be2cd0d6c7385092",
    ),
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_artifact(
    artifact: _ModelArtifact,
    destination: Path,
    *,
    attempts: int,
    retry_delay_seconds: float,
    opener: Callable[..., BinaryIO] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download an artifact atomically and retry transient failures."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / artifact.filename
    partial = target.with_suffix(f"{target.suffix}.part")

    for attempt in range(1, attempts + 1):
        try:
            with opener(artifact.url, timeout=120) as response, partial.open("wb") as file:
                shutil.copyfileobj(response, file)

            if _sha256(partial) != artifact.sha256:
                raise OSError(f"Checksum mismatch for {artifact.filename}")

            partial.replace(target)
            return target
        except HTTPError as error:
            partial.unlink(missing_ok=True)
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == attempts:
                raise
        except (OSError, TimeoutError, URLError):
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise

        sleep(retry_delay_seconds * 2 ** (attempt - 1))

    raise RuntimeError(f"Failed to download {artifact.filename}")


def _ensure_artifact(
    artifact: _ModelArtifact,
    destination: Path,
    *,
    attempts: int,
    retry_delay_seconds: float,
    opener: Callable[..., BinaryIO] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Reuse a verified artifact or replace it with a verified download."""
    target = destination / artifact.filename
    if target.is_file() and _sha256(target) == artifact.sha256:
        return target

    target.unlink(missing_ok=True)
    return _download_artifact(
        artifact,
        destination,
        attempts=attempts,
        retry_delay_seconds=retry_delay_seconds,
        opener=opener,
        sleep=sleep,
    )


def _prefetch_models(
    destination: Path,
    *,
    artifacts: tuple[_ModelArtifact, ...] = _MODEL_ARTIFACTS,
    attempts: int = 5,
    retry_delay_seconds: float = 5,
    opener: Callable[..., BinaryIO] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Path, ...]:
    """Ensure every configured crack-detection model is verified locally."""
    return tuple(
        _ensure_artifact(
            artifact,
            destination,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
            opener=opener,
            sleep=sleep,
        )
        for artifact in artifacts
    )


def _main(arguments: list[str] | None = None) -> int:
    """Run the model prefetch command for a repository checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("crackpy/crack_detection/models"),
        help="directory used by CrackPy for downloaded model files",
    )
    options = parser.parse_args(arguments)

    for model_path in _prefetch_models(options.destination):
        print(f"Verified crack-detection model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
