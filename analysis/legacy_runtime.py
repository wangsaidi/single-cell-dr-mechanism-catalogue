"""Validate the legacy interpreter using the same DLL paths as model fitting."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from analysis.paths import ANALYSIS_ROOT

SAUCIE_COMMIT = "5ab7976fc8d19a3823b6005f51736dc70f2017fb"


def legacy_environment(python_exe: Path, source: Path) -> dict[str, str]:
    env = os.environ.copy()
    prefixes = [python_exe.parent / part for part in
                ("", "Library/bin", "Library/mingw-w64/bin", "DLLs", "Scripts")]
    env["PATH"] = os.pathsep.join([*(str(p) for p in prefixes), env.get("PATH", "")])
    # The checkout itself is the SAUCIE package, not its import search root.
    env["PYTHONPATH"] = str(source.resolve().parent)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return env


def legacy_source() -> Path:
    return Path(os.environ.get("SCDR_SAUCIE_SOURCE", ANALYSIS_ROOT / "external" / "SAUCIE")).resolve()


def require_legacy_python() -> Path:
    configured = os.environ.get("SCDR_LEGACY_PYTHON", "").strip()
    if not configured or not Path(configured).is_file():
        raise FileNotFoundError(
            "SCDR_LEGACY_PYTHON must name an existing Python 3.7 executable file, "
            "not an empty value or directory."
        )
    python_exe = Path(configured).resolve()
    probe = (
        "import sys,json; "
        "assert sys.version_info[:2] == (3,7), 'Python 3.7 required'; "
        "import numpy,tensorflow,scscope,pkg_resources; "
        "assert numpy.__version__ == '1.18.5', 'NumPy 1.18.5 required'; "
        "assert tensorflow.__version__ == '1.15.0', 'TensorFlow 1.15.0 required'; "
        "assert pkg_resources.get_distribution('scscope').version == '0.1.5', 'scscope 0.1.5 required'; "
        "print(json.dumps({'python':sys.version,'numpy':numpy.__version__,"
        "'tensorflow':tensorflow.__version__,'scscope_file':scscope.__file__,"
        "'packages':{p:pkg_resources.get_distribution(p).version for p in ['scscope','h5py','scikit-learn','pandas']}}))"
    )
    try:
        result = subprocess.run([str(python_exe), "-c", probe], check=True,
                                capture_output=True, text=True, timeout=90,
                                env=legacy_environment(python_exe, legacy_source()))
    except (OSError, subprocess.SubprocessError) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"Legacy Python 3.7 / TensorFlow 1.15 / scscope preflight failed: {details}") from exc
    from analysis.paths import LOGS_DIR
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "legacy_runtime_probe.txt").write_text(result.stdout, encoding="utf-8")
    return python_exe


def verify_legacy_source() -> dict[str, str]:
    source = legacy_source()
    try:
        head = subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "-C", str(source), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.PIPE).strip()
        dirty = subprocess.check_output(
            ["git", "-c", "core.longpaths=true", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
            text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("SAUCIE source unavailable; run python -m analysis.setup_legacy_tools first.") from exc
    if head != SAUCIE_COMMIT or dirty:
        raise RuntimeError(f"SAUCIE must be clean at {SAUCIE_COMMIT}; observed {head}, dirty={bool(dirty)}")
    return {"path": str(source), "commit": head, "working_tree": "clean"}


def probe_saucie_import(python_exe: Path, source: Path) -> dict:
    """Import the checked-out package without constructing or training a model."""
    source = source.resolve()
    probe = ANALYSIS_ROOT / "legacy_import_probe.py"
    command = [str(python_exe), str(probe), "--saucie-parent", str(source)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                timeout=90, env=legacy_environment(python_exe, source))
    except (OSError, subprocess.SubprocessError) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            f"SAUCIE import preflight failed in the legacy fitting context for {source}: {details}"
        ) from exc
    try:
        return json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SAUCIE import preflight returned invalid JSON: {result.stdout!r}") from exc


def main() -> None:
    from analysis.paths import LOGS_DIR
    python_exe = require_legacy_python()
    source_record = verify_legacy_source()
    import_record = probe_saucie_import(python_exe, legacy_source())
    payload = {"executable": str(python_exe), "SAUCIE": source_record,
               "SAUCIE_import": import_record,
               "effective_batch_sizes": {"scScope": 64, "SAUCIE": 256},
               "scScope": {"layer_width": 50, "recurrence_depth": 2,
                           "official_predict_concatenates_recurrences": True,
                           "expected_latent_columns": 100, "projection_input_shape_recorded_per_fit": True},
               "runner_behavior": "historical runner unchanged; SAUCIE train omits batch_size"}
    (LOGS_DIR / "legacy_preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
