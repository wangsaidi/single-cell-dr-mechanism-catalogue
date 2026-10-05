"""Download and verify the archived inputs used by the public workflow."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from analysis.paths import MANIFEST_PATH, REPO_ROOT


RELEASE_TAG = "analysis-reproducibility-v1.0.1"
ARCHIVE_NAME = "single-cell-dr-analysis-inputs-v1.0.1.zip"
ARCHIVE_URL = (
    "https://github.com/wangsaidi/single-cell-dr-mechanism-catalogue/"
    f"releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"
)
ARCHIVE_SHA256 = "f0b16690759d26235a8a2103ccc951b973e2d13e621464864626a7d5f6cad797"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local copy of the release archive.")
    parser.add_argument("--force", action="store_true", help="Replace an existing installed bundle.")
    args = parser.parse_args()

    if MANIFEST_PATH.exists() and not args.force:
        print(f"Inputs already installed: {MANIFEST_PATH}")
        print("Use --force only when a fresh extraction is required.")
        return

    with tempfile.TemporaryDirectory(prefix="single-cell-dr-download-") as temp_dir:
        archive = args.archive.resolve() if args.archive else Path(temp_dir) / ARCHIVE_NAME
        if not args.archive:
            print(f"Downloading {ARCHIVE_URL}")
            urllib.request.urlretrieve(ARCHIVE_URL, archive)
        observed = sha256(archive)
        if observed != ARCHIVE_SHA256:
            raise ValueError(f"Archive checksum mismatch: {observed}")
        safe_extract(archive, REPO_ROOT)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"The extracted bundle did not create {MANIFEST_PATH}")
    print(f"Installed and verified {ARCHIVE_NAME}")


if __name__ == "__main__":
    main()
