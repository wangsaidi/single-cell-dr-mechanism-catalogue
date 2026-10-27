"""Run one module with read provenance and workflow-level input write guards.

Python's audit hook observes Python file opens, not every native-library read or
legacy child process. HDF5 paths are additionally covered by stage inventories.
This is provenance evidence, not a security sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InputAudit:
    def __init__(self, profile: str, data: Path, results: Path, protected: list[Path]):
        self.profile = profile
        self.data = data
        self.results = results
        self.protected = protected
        self.reads: dict[str, dict] = {}
        self.busy = False

    def __call__(self, event, args):
        if event != "open" or self.busy or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        protected = any(path.is_relative_to(root) for root in self.protected)
        if writing and protected:
            raise PermissionError(f"Workflow input is read-only: {path}")
        if not writing and protected and self.profile == "full-refit" and path.suffix.lower() in {".csv", ".gz", ".npy", ".npz", ".h5ad"}:
            raise PermissionError(f"Full refit cannot consume archived fitted inputs or publication tables: {path}")
        if writing or not path.is_file() or path.suffix.lower() not in {".csv", ".gz", ".npy", ".npz", ".h5ad", ".json"}:
            return
        if not (protected or path.is_relative_to(self.data) or path.is_relative_to(self.results)):
            return
        key = str(path)
        if key in self.reads:
            return
        self.busy = True
        try:
            parts = set(path.parts)
            if "raw_cache" in parts:
                role = "public_source_cache_including_processed_annotations"
            elif "analysis_objects" in parts or "expression_objects" in parts:
                role = "processed_expression_input"
            elif any(p in parts for p in ("anchor_embeddings", "embeddings", "robustness_embeddings", "legacy_method_cache", "simulation_cache")):
                role = "fitted_or_generated_input"
            else:
                role = "table_or_configuration"
            self.reads[key] = {"path": key, "role": role, "sha256": file_hash(path)}
        finally:
            self.busy = False


def main() -> None:
    from analysis.paths import ANALYSIS_ROOT, DATA_DIR, LOGS_DIR, RESULTS_DIR, SCVI_EMBEDDINGS_DIR, PUBLIC_SOURCE_DATA_DIR
    profile = os.environ["SCDR_WORKFLOW_PROFILE"]
    protected = [PUBLIC_SOURCE_DATA_DIR, ANALYSIS_ROOT / "data", ANALYSIS_ROOT / "results" / "embeddings"]
    protected.extend(Path(p) for p in json.loads(os.environ.get("SCDR_PROTECTED_INPUTS", "[]")))
    if profile in {"archived", "refit-scvi"}:
        protected.append(DATA_DIR)
    if profile == "archived":
        protected.append(SCVI_EMBEDDINGS_DIR)
    audit = InputAudit(profile, DATA_DIR, RESULTS_DIR, protected)
    sys.addaudithook(audit)
    stage, module, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        audit.busy = True
        payload = {"stage": stage, "profile": profile, "reads": list(audit.reads.values()),
                   "scope": "Python open audit; native HDF5 and legacy child reads may be absent; see inventories and fit metadata"}
        with (LOGS_DIR / f"{stage}.input_reads.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
