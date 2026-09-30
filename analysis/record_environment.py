"""Record a portable software report without machine-specific paths."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

from analysis.paths import LOGS_DIR, ensure_analysis_dirs


PACKAGES = [
    "anndata",
    "glmpca",
    "igraph",
    "leidenalg",
    "numpy",
    "pacmap",
    "pandas",
    "phate",
    "scanpy",
    "scikit-learn",
    "scipy",
    "scvi-tools",
    "torch",
    "umap-learn",
]


def main() -> None:
    ensure_analysis_dirs()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": versions,
        "command": sys.argv,
    }
    out = LOGS_DIR / "runtime_environment.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
