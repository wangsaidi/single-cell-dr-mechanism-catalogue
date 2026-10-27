"""Retrieve the recorded SAUCIE source revision used by legacy deep fits."""

from __future__ import annotations

import subprocess
from pathlib import Path

from analysis.legacy_runtime import SAUCIE_COMMIT, legacy_source


SAUCIE_REPOSITORY = "https://github.com/KrishnaswamyLab/SAUCIE.git"


def main() -> None:
    target = legacy_source()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(["git", "-c", "core.longpaths=true", "clone", SAUCIE_REPOSITORY, str(target)], check=True)
    if not (target / ".git").exists():
        raise RuntimeError("SAUCIE source directory is not a Git checkout; preserve it and select a new SCDR_SAUCIE_SOURCE.")
    git = ["git", "-c", "core.longpaths=true", "-C", str(target)]
    dirty = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    if dirty:
        raise RuntimeError("SAUCIE checkout has local changes; preserve them and select a clean SCDR_SAUCIE_SOURCE.")
    subprocess.run([*git, "fetch", "origin"], check=True)
    subprocess.run([*git, "checkout", "--detach", SAUCIE_COMMIT], check=True)
    observed = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if observed != SAUCIE_COMMIT:
        raise RuntimeError(f"Unexpected SAUCIE revision: {observed}")
    print(f"SAUCIE source ready at {target} ({observed})")


if __name__ == "__main__":
    main()
