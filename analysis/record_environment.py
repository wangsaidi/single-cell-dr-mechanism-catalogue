"""Record a portable software report without machine-specific paths."""

from __future__ import annotations

import importlib.metadata
import argparse
import hashlib
import json
import platform
import os
import sys

from analysis.paths import ANALYSIS_ROOT, LOGS_DIR, ensure_analysis_dirs


PACKAGES = [
    "anndata",
    "glmpca",
    "igraph",
    "leidenalg",
    "matplotlib",
    "numpy",
    "pacmap",
    "pandas",
    "phate",
    "scanpy",
    "scikit-learn",
    "scipy",
    "scvi-tools",
    "seaborn",
    "torch",
    "umap-learn",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check pinned installed metadata, not importability or rebuild success.")
    args = parser.parse_args()
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
        "executable": sys.executable,
        "metadata_check_scope": "installed distribution versions only; no environment creation or model fit is implied",
        "execution_scale": os.environ.get("SCDR_EXECUTION_SCALE", "publication"),
    }
    sources = {}
    for path in ANALYSIS_ROOT.rglob("*"):
        relative = path.relative_to(ANALYSIS_ROOT)
        if not path.is_file() or any(part in {"external", "data", "results", "logs", "runs", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix in {".py", ".yml", ".yaml", ".json", ".txt", ".csv"}:
            sources[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["analysis_source_sha256"] = sources
    requirements = ANALYSIS_ROOT / "environment" / "requirements-analysis.txt"
    errors = []
    if args.check:
        from packaging.requirements import Requirement
        for line in requirements.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            requirement = Requirement(line)
            try:
                observed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"missing {requirement.name}")
                continue
            if observed not in requirement.specifier:
                errors.append(f"{requirement}: installed {observed}")
        if sys.version_info[:2] != (3, 14):
            errors.append("recorded modern environment requires Python 3.14")
    payload["constraint_errors"] = errors
    out = LOGS_DIR / "runtime_environment.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    if errors:
        raise SystemExit("Installed metadata does not satisfy recorded constraints: " + "; ".join(errors))


if __name__ == "__main__":
    main()
