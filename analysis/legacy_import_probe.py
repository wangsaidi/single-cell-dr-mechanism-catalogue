"""Python 3.7-compatible import-only counterpart to the standalone fit runner."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saucie-parent", required=True)
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KMP_AFFINITY", "disabled")

    # Match legacy_deep_runner.run_saucie; both scripts have the same directory.
    sys.path.insert(0, args.saucie_parent)
    import tensorflow as tf
    import SAUCIE

    expected = (Path(args.saucie_parent) / "__init__.py").resolve()
    observed = Path(SAUCIE.__file__).resolve()
    if observed != expected:
        raise RuntimeError("SAUCIE import resolved to {} instead of {}".format(observed, expected))
    if not callable(getattr(SAUCIE, "SAUCIE", None)) or not callable(getattr(SAUCIE, "Loader", None)):
        raise RuntimeError("SAUCIE package must expose callable SAUCIE and Loader interfaces")
    payload = {
        "package_file": str(observed),
        "tensorflow": tf.__version__,
        "python": sys.version,
        "script_directory": str(Path(__file__).resolve().parent),
        "saucie_parent_argument": args.saucie_parent,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "working_directory": str(Path.cwd()),
        "model_instantiated": False,
        "training_executed": False,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
