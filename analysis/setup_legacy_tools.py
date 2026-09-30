"""Retrieve the recorded SAUCIE source revision used by legacy deep fits."""

from __future__ import annotations

import subprocess
from pathlib import Path

from analysis.paths import ANALYSIS_ROOT


SAUCIE_REPOSITORY = "https://github.com/KrishnaswamyLab/SAUCIE.git"
SAUCIE_COMMIT = "5ab7976fc8d19a3823b6005f51736dc70f2017fb"


def main() -> None:
    target = ANALYSIS_ROOT / "external" / "SAUCIE"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(["git", "clone", SAUCIE_REPOSITORY, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", SAUCIE_COMMIT], check=True)
    observed = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if observed != SAUCIE_COMMIT:
        raise RuntimeError(f"Unexpected SAUCIE revision: {observed}")
    print(f"SAUCIE source ready at {target} ({observed})")


if __name__ == "__main__":
    main()
