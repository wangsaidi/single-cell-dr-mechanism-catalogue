"""Run isolated archived analysis, scVI-only refits or public-source full refits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analysis.paths import ANALYSIS_ROOT, REPO_ROOT
from analysis.workflow.stage_runner import file_hash


METRIC_STEPS = [
    ("audit_inputs", ["-m", "analysis.scripts.audit_inputs"]),
    ("audit_scvi", ["-m", "analysis.scripts.audit_scvi"]),
    ("diagnostics", ["-m", "analysis.scripts.compute_diagnostics"]),
    ("dimension_sensitivity", ["-m", "analysis.scripts.compute_dimension_sensitivity"]),
    ("clustering", ["-m", "analysis.scripts.compute_clustering"]),
    ("marker_concordance", ["-m", "analysis.scripts.compute_marker_concordance"]),
    ("trajectory", ["-m", "analysis.scripts.compute_trajectory"]),
    ("donor_predictability", ["-m", "analysis.scripts.compute_donor_predictability"]),
    ("summarise", ["-m", "analysis.scripts.summarise_results"]),
    ("audit_results", ["-m", "analysis.scripts.audit_results"]),
    ("export_figure4", ["-m", "analysis.export_figure4_source_data"]),
    ("threshold_calibration", ["-m", "analysis.scripts.calibrate_operational_thresholds"]),
]
ENVIRONMENT_STEP = ("record_environment", ["-m", "analysis.record_environment", "--check"])
ARCHIVED_STEPS = [
    ENVIRONMENT_STEP,
    ("verify_inputs", ["-m", "analysis.verify_reproducibility", "--inputs-only"]),
    *METRIC_STEPS,
    ("verify_results", ["-m", "analysis.verify_reproducibility"]),
    ("verify_calibration", ["-m", "analysis.verify_calibration"]),
]


def full_refit_steps(smoke: bool = False) -> list[tuple[str, list[str]]]:
    # All fit stages force outer and inner caches, including on resume.
    return [
        ENVIRONMENT_STEP,
        ("legacy_preflight", ["-m", "analysis.legacy_runtime"]),
        ("prepare_objects", ["-m", "analysis.scripts.prepare_analysis_objects"]),
        *[(f"anchor_embeddings_seed_{seed}", ["-m", "analysis.scripts.generate_anchor_embeddings",
            "--seed", str(seed), "--force"]) for seed in (range(2) if smoke else range(5))],
        ("known_truth_stress_tests", ["-m", "analysis.scripts.generate_known_truth_stress_tests", "--force"]),
        ("robustness_embeddings", ["-m", "analysis.scripts.generate_robustness_embeddings", "--force"]),
        ("fit_scvi", ["-m", "analysis.scripts.fit_scvi", "--force"]),
        *[(name, [*arguments, "--permutations", "19"] if smoke and name == "threshold_calibration" else arguments)
          for name, arguments in METRIC_STEPS],
        ("validate_refit", ["-m", "analysis.workflow.validate_refit"]),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_run(args) -> tuple[dict[str, str], Path, dict]:
    run_dir = args.run_dir.resolve()
    archive_data = args.archive_data.resolve()
    archive_scvi = args.archive_results.resolve() / "embeddings"
    protected = [ANALYSIS_ROOT / "data", ANALYSIS_ROOT / "results", REPO_ROOT / "data",
                 archive_data, args.archive_results.resolve()]
    for root in protected:
        if run_dir.is_relative_to(root) or root.is_relative_to(run_dir):
            raise ValueError(f"Run directory must not overlap archive/public inputs: {root}")
    metadata_path = run_dir / "run.json"
    nonempty = run_dir.exists() and any(run_dir.iterdir())
    if nonempty and not args.resume:
        raise ValueError("Run directory is not empty; choose a new directory or explicitly --resume.")
    if args.resume:
        if not metadata_path.is_file():
            raise ValueError("--resume requires an existing run.json; arbitrary caches are not accepted.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = (args.profile, str(archive_data), str(archive_scvi))
        if tuple(metadata.get(k) for k in ("profile", "archive_data", "archive_scvi")) != expected:
            raise ValueError("Resume profile/archive locations must match the original run.")
    else:
        metadata = {"profile": args.profile, "created_utc": utc_now(), "initially_empty": True,
                    "archive_data": str(archive_data), "archive_scvi": str(archive_scvi)}
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "SCDR_WORKFLOW_PROFILE": "full-refit" if args.profile == "smoke" else args.profile,
        "SCDR_EXECUTION_SCALE": "smoke" if args.profile == "smoke" else "publication",
        "SCDR_ANALYSIS_DATA": str(run_dir / "data" if args.profile in {"full-refit", "smoke"} else archive_data),
        "SCDR_ANALYSIS_RESULTS": str(run_dir / "results"),
        "SCDR_ANALYSIS_LOGS": str(run_dir / "logs"),
        "SCDR_SCVI_EMBEDDINGS": str(archive_scvi if args.profile == "archived" else run_dir / "results" / "embeddings"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SCDR_PROTECTED_INPUTS": json.dumps([str(root) for root in protected]),
    })
    metadata.update({"last_started_utc": utc_now(), "status": "running", "complete_workflow": False,
        "last_command": sys.argv,
        "paths": {k: v for k, v in env.items() if k.startswith("SCDR_ANALYSIS_") or k == "SCDR_SCVI_EMBEDDINGS"},
        "input_policy": "public counts plus source-supplied annotations/subsets; no archived fits/tables"
        if args.profile in {"full-refit", "smoke"} else "archived processed objects and fitted inputs; not raw-source refitting",
        "execution_scale": "smoke_not_publication" if args.profile == "smoke" else "publication"})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return env, run_dir, metadata


def inventory(roots: list[Path]) -> list[dict]:
    records = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    return records


def execute_steps(steps, env: dict[str, str], run_dir: Path) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if env["SCDR_WORKFLOW_PROFILE"] != "full-refit":
        inputs = inventory([Path(env["SCDR_ANALYSIS_DATA"]), Path(env["SCDR_SCVI_EMBEDDINGS"])])
        (logs / "archived_input_inventory.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    with (logs / "stages.jsonl").open("a", encoding="utf-8") as events:
        for name, arguments in steps:
            command = [sys.executable, "-m", "analysis.workflow.stage_runner", name, arguments[1], *arguments[2:]]
            started = time.monotonic()
            record = {"stage": name, "command": command, "started_utc": utc_now()}
            events.write(json.dumps({**record, "event": "start"}) + "\n")
            events.flush()
            print(f"START {name} ({logs / (name + '.log')})", flush=True)
            try:
                with (logs / f"{name}.log").open("a", encoding="utf-8") as output:
                    subprocess.run(command, check=True, env=env, cwd=REPO_ROOT, stdout=output, stderr=subprocess.STDOUT)
            except BaseException as exc:
                events.write(json.dumps({**record, "event": "failure", "finished_utc": utc_now(),
                    "elapsed_seconds": time.monotonic() - started, "error": str(exc),
                    "returncode": getattr(exc, "returncode", None)}) + "\n")
                events.flush()
                raise
            events.write(json.dumps({**record, "event": "finish", "finished_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started, "returncode": 0}) + "\n")
            events.flush()
            print(f"FINISH {name}", flush=True)
            if name == "prepare_objects":
                records = inventory([Path(env["SCDR_ANALYSIS_DATA"])])
                (logs / "prepared_input_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["archived", "refit-scvi", "full-refit", "smoke"], default="archived")
    parser.add_argument("--run-dir", type=Path, help="New empty isolated output directory; default: analysis/runs/<profile>_<UTC timestamp>.")
    parser.add_argument("--archive-data", type=Path, default=ANALYSIS_ROOT / "data")
    parser.add_argument("--archive-results", type=Path, default=ANALYSIS_ROOT / "results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refit profiles already force every fitting cache; disallowed for archived.")
    parser.add_argument("--from-step")
    parser.add_argument("--through-step")
    args = parser.parse_args()
    if args.run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.run_dir = ANALYSIS_ROOT / "runs" / f"{args.profile}_{timestamp}"
    if args.profile == "archived" and args.force:
        parser.error("--force cannot be used with the read-only archived profile")
    if args.profile == "archived":
        steps = ARCHIVED_STEPS
    elif args.profile == "refit-scvi":
        # Deliberately mixed: anchors/preprocessing remain archived. No exact-fit hash comparison.
        steps = [ENVIRONMENT_STEP,
                 ("verify_archived_data", ["-m", "analysis.verify_reproducibility", "--inputs-only", "--data-only"]),
                 ("fit_scvi", ["-m", "analysis.scripts.fit_scvi", "--force"]),
                 *METRIC_STEPS, ("validate_refit", ["-m", "analysis.workflow.validate_refit"])]
    else:
        steps = full_refit_steps(smoke=args.profile == "smoke")
    names = [name for name, _ in steps]
    for option in (args.from_step, args.through_step):
        if option and option not in names:
            parser.error(f"unknown step {option}; choose from {', '.join(names)}")
    start = names.index(args.from_step) if args.from_step else 0
    stop = names.index(args.through_step) if args.through_step else len(steps) - 1
    if start > stop:
        parser.error("--from-step must precede --through-step")
    if start and not args.resume:
        parser.error("--from-step requires --resume and previously generated inputs")
    try:
        env, run_dir, metadata = prepare_run(args)
    except ValueError as exc:
        parser.error(str(exc))
    selected = steps[start:stop + 1]
    metadata["selected_stages"] = [name for name, _ in selected]
    try:
        execute_steps(selected, env, run_dir)
        metadata["status"] = "completed" if start == 0 and stop == len(steps) - 1 else "partial"
        metadata["complete_workflow"] = metadata["status"] == "completed"
        metadata["publication_scale_full_refit_completed"] = metadata["complete_workflow"] and args.profile == "full-refit"
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        metadata["finished_utc"] = utc_now()
        metadata["outputs"] = inventory([run_dir / "data", run_dir / "results"])
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
