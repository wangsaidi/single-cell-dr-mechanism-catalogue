"""Register fixed inputs and launch isolated deep-model configuration controls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from analysis import legacy_runtime, paths


DATASETS = ("pbmc3k", "paul15", "heart_cell_atlas_subsampled")
CONFIGURATIONS = ("recurrent_latent", "reconstruction",
                  "task_architecture_reconstruction", "task_enabled")
RUNNER = paths.ANALYSIS_ROOT / "round2_deep_controls.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def build_jobs(output: Path, source: Path, count_hashes: dict[str, str]) -> list[dict]:
    jobs = []
    for dataset in DATASETS:
        for method, configurations in (
            ("scScope", ("recurrent_latent",)),
            ("SAUCIE", CONFIGURATIONS[1:]),
        ):
            for seed in (0, 1, 2):
                for configuration in configurations:
                    stem = f"{dataset}_{method}_{configuration}_seed{seed}"
                    jobs.append({
                        "dataset": dataset, "method": method,
                        "configuration": configuration, "seed": seed,
                        "input": str(output / f"input_{dataset}_{method}.npz"),
                        "out_dir": str(output / stem), "saucie_source": str(source),
                        "max_steps": 10000, "max_epochs": 600,
                        "original_counts_file_sha256": count_hashes[dataset],
                    })
    return jobs


def prepare(objects: Path, output: Path, source: Path, reuse: bool,
            fresh_attempt: bool = False) -> list[dict]:
    import anndata as ad
    import numpy as np
    from scipy import sparse

    for readonly in (objects, source):
        if output == readonly or output in readonly.parents or readonly in output.parents:
            raise ValueError("The control output must not overlap count objects or model source.")
    count_paths = {dataset: objects / f"{dataset}_counts_hvg.h5ad" for dataset in DATASETS}
    count_hashes = {dataset: sha256(path) for dataset, path in count_paths.items()}
    source_hashes = {"launcher": sha256(Path(__file__)), "child": sha256(RUNNER), "SAUCIE_model": sha256(source / "model.py"),
                     "SAUCIE_loader": sha256(source / "loader.py")}
    jobs = build_jobs(output, source, count_hashes)
    manifest = output / "analysis_manifest.json"
    if manifest.exists():
        if not reuse:
            raise FileExistsError("Registered output exists; use --resume or a fresh --output-dir.")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        saved_sources = saved["source_sha256"]
        compared_sources = {key: value for key, value in source_hashes.items()
                            if not (fresh_attempt and key == "launcher")}
        if (saved["jobs"] != jobs
                or any(saved_sources.get(key) != value for key, value in compared_sources.items())
                or saved["objects_dir"] != str(objects)):
            raise ValueError("Registered inputs, sources or settings changed; use a fresh output directory.")
        for record in saved["inputs"]:
            if sha256(output / record["file"]) != record["sha256"]:
                raise ValueError(f"Registered input changed: {record['file']}")
        return jobs
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is nonempty without a control registration; preserve it and choose a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, count_path in count_paths.items():
        counts = ad.read_h5ad(count_path)
        raw = counts.X.toarray() if sparse.issparse(counts.X) else np.asarray(counts.X)
        raw = raw.astype(np.float32)
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError(f"Nonfinite or negative count input: {dataset}")
        for method in ("scScope", "SAUCIE"):
            if method == "scScope":
                library = raw.sum(axis=1)
                if not (library > 0).any():
                    raise ValueError(f"No positive count libraries: {dataset}")
                target = float(np.median(library[library > 0]))
                value = np.zeros_like(raw)
                value[library > 0] = raw[library > 0] * (target / library[library > 0, None])
            else:
                value = np.log1p(raw)
            inp = output / f"input_{dataset}_{method}.npz"
            with inp.open("xb") as stream:
                np.savez_compressed(stream, X=value,
                                    cell_ids=np.asarray(counts.obs_names.astype(str), dtype=str),
                                    gene_ids=np.asarray(counts.var_names.astype(str), dtype=str))
            records.append({"file": inp.name, "sha256": sha256(inp),
                            "dataset": dataset, "method": method, "shape": list(value.shape)})
    write_new_json(manifest, {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "status": "registered before new comparative scores", "objects_dir": str(objects),
        "seeds_are_computational": True, "jobs": jobs, "inputs": records,
        "source_sha256": source_hashes,
        "source_scope": "three complete count-HVG analysis objects; no outcome-tuned settings",
        "saucie_penalty_rationale": "lambda_c=.1 and lambda_d=.2 from the fixed original-source example; architecture-matched reconstruction control included",
        "scscope_note": "50 units per recurrent state; official predict concatenates T=2 states to 100 columns",
        "stopping": "minimum 3000 SAUCIE steps or 300 scScope epochs and three consecutive relative diagnostic-loss changes below .001; capped at 10000 steps/600 epochs",
        "unmodified_task_enabled": "included; known numerical failures must be retained, not substituted",
    })
    return jobs


def select_jobs(jobs: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = [job for job in jobs
                if (not args.pilot or (job["dataset"] == "pbmc3k" and job["seed"] == 0))
                and job["configuration"] not in args.exclude_configuration
                and (not args.configuration or job["configuration"] in args.configuration)
                and (not args.seed or job["seed"] in args.seed)
                and Path(job["out_dir"]).name not in args.exclude_job]
    if not selected:
        raise ValueError("The explicit filters selected no jobs.")
    if args.stabilize_distances:
        if not args.attempt:
            raise ValueError("Numerical distance sensitivity requires a fresh labelled --attempt.")
        if any(job["method"] != "SAUCIE" for job in selected):
            raise ValueError("Select SAUCIE configurations explicitly for numerical sensitivity; use --configuration task_enabled for the nine-job control.")
    if args.attempt:
        selected = [{**job, "out_dir": job["out_dir"] + "_" + args.attempt,
                     "attempt": args.attempt} for job in selected]
    if args.max_epochs is not None or args.max_steps is not None:
        if not args.attempt:
            raise ValueError("Budget sensitivity requires a fresh labelled --attempt.")
        selected = [{**job,
                     "max_epochs": args.max_epochs if args.max_epochs is not None else job["max_epochs"],
                     "max_steps": args.max_steps if args.max_steps is not None else job["max_steps"],
                     "maximum_budget_extension": True} for job in selected]
    if args.stabilize_distances:
        selected = [{**job, "saucie_distance_stabilization": True} for job in selected]
    return selected


def run_job(job: dict, python: Path, source: Path) -> dict:
    out = Path(job["out_dir"])
    meta_path = out / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or any(meta.get(k) != v for k, v in job.items()):
            raise ValueError(f"Completed metadata does not match registration: {out}")
        return {"job": out.name, "status": "previously_completed_control",
                "checkpoint_stability_rule_met": meta.get("checkpoint_stability_rule_met")}
    if out.exists():
        raise FileExistsError(f"Preserve the failed/partial job and choose a fresh --attempt: {out}")
    out.mkdir()
    spec = out / "job.json"
    write_new_json(spec, job)
    env = legacy_runtime.legacy_environment(python, source)
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONHASHSEED": str(job["seed"]), "PYTHONIOENCODING": "utf-8",
                "CUDA_VISIBLE_DEVICES": ""})
    started = time.time()
    command = [str(python), "-B", str(RUNNER), str(spec)]
    with (out / "training.log").open("x", encoding="utf-8") as log:
        process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {"job": out.name, "status": "complete" if process.returncode == 0 else "failed",
              "returncode": process.returncode, "elapsed_seconds": time.time() - started,
              "command": command, "log": str(out / "training.log")}
    if process.returncode == 0:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or any(meta.get(k) != v for k, v in job.items()):
            raise ValueError(f"Child success did not match registered metadata: {out}")
        record["checkpoint_stability_rule_met"] = meta.get("checkpoint_stability_rule_met")
    else:
        write_new_json(out / "failure.json", record)
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path,
                        default=Path(os.environ.get("SCDR_CONTROL_OBJECTS_DIR", paths.ANALYSIS_OBJECTS_DIR)))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("SCDR_DEEP_CONTROLS_DIR", paths.RESULTS_DIR / "deep_configuration_controls")))
    parser.add_argument("--legacy-python", type=Path, default=os.environ.get("SCDR_LEGACY_PYTHON"))
    parser.add_argument("--saucie-source", type=Path, default=legacy_runtime.legacy_source())
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Validate registration and reuse completed jobs only; never restart a nonempty failed job.")
    parser.add_argument("--pilot", action="store_true", help="Select the four PBMC3k seed-zero jobs from the registered 36-job plan.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--configuration", action="append", choices=CONFIGURATIONS, default=[])
    parser.add_argument("--exclude-configuration", action="append", choices=CONFIGURATIONS, default=[])
    parser.add_argument("--exclude-job", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", choices=(0, 1, 2), default=[],
                        help="Select a registered computational seed; repeat to select several.")
    parser.add_argument("--max-epochs", type=int,
                        help="scScope epoch cap for a separately labelled budget attempt; default plan remains 600.")
    parser.add_argument("--max-steps", type=int,
                        help="SAUCIE update cap for a separately labelled budget attempt; default plan remains 10000.")
    parser.add_argument("--attempt", help="Fresh alphanumeric/underscore attempt label; never replaces the original plan.")
    parser.add_argument("--stabilize-distances", action="store_true",
                        help="Opt-in SAUCIE numerical implementation sensitivity; requires --attempt and explicit SAUCIE selection.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.attempt and not re.fullmatch(r"[A-Za-z0-9_]+", args.attempt):
        parser.error("--attempt must contain only letters, digits and underscores")
    if args.attempt == "original":
        parser.error("original is reserved for unlabelled jobs in score filtering; choose another attempt label")
    if args.stabilize_distances and not args.attempt:
        parser.error("--stabilize-distances requires --attempt")
    if args.max_epochs is not None or args.max_steps is not None:
        if not args.attempt:
            parser.error("--max-epochs/--max-steps require a fresh labelled --attempt")
        if any(value is not None and value <= 0 for value in (args.max_epochs, args.max_steps)):
            parser.error("Budget caps must be positive")
    return args


def main() -> None:
    args = parse_args()
    objects, output, source = (p.resolve() for p in (args.objects_dir, args.output_dir, args.saucie_source))
    jobs = prepare(objects, output, source, reuse=args.resume or bool(args.attempt),
                   fresh_attempt=bool(args.attempt))
    selected = select_jobs(jobs, args)
    if args.attempt:
        registration = output / f"attempt_{args.attempt}.json"
        if registration.exists():
            if not args.resume or json.loads(registration.read_text(encoding="utf-8")) != selected:
                raise FileExistsError("Attempt already registered; use --resume with identical filters or a new label.")
        else:
            write_new_json(registration, selected)
    print(f"Registered {len(jobs)} original-model fits; selected {len(selected)} jobs.", flush=True)
    if args.prepare_only:
        return
    if not args.legacy_python:
        raise ValueError("Set --legacy-python or SCDR_LEGACY_PYTHON to the legacy Python executable.")
    invocation = output / "executions" / str(time.time_ns())
    invocation.mkdir(parents=True)
    write_new_json(invocation / "plan.json", {"jobs": selected, "filters": {
        "pilot": args.pilot, "configuration": args.configuration,
        "exclude_configuration": args.exclude_configuration, "exclude_job": args.exclude_job,
        "seed": args.seed, "max_epochs": args.max_epochs, "max_steps": args.max_steps,
        "attempt": args.attempt, "stabilize_distances": args.stabilize_distances},
        "launcher_sha256": sha256(Path(__file__)), "child_sha256": sha256(RUNNER)})
    os.environ["SCDR_LEGACY_PYTHON"] = str(args.legacy_python.resolve())
    os.environ["SCDR_SAUCIE_SOURCE"] = str(source)
    # Keep helper preflight logs in this control invocation, not a primary run.
    previous_logs = paths.LOGS_DIR
    paths.LOGS_DIR = invocation / "runtime_logs"
    try:
        python = legacy_runtime.require_legacy_python()
        source_record = legacy_runtime.verify_legacy_source()
        import_record = legacy_runtime.probe_saucie_import(python, source)
    except Exception as exc:
        write_new_json(invocation / "preflight_failure.json", {"status": "failed_preflight",
                                                              "exception": repr(exc)})
        raise
    finally:
        paths.LOGS_DIR = previous_logs
    write_new_json(invocation / "preflight.json", {"python": str(python), "source": source_record,
                                                   "import": import_record})
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, python, source): job for job in selected}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = {"job": Path(futures[future]["out_dir"]).name,
                          "status": "failed", "exception": repr(exc)}
            completed.append(record)
            write_new_json(invocation / f"job_{len(completed):02d}.json", record)
            print(json.dumps(record), flush=True)
    write_new_json(invocation / "execution.json", completed)
    if any(record["status"] == "failed" for record in completed):
        raise SystemExit("A genuine failure was retained; no retry, replacement score or fallback was used.")


if __name__ == "__main__":
    main()
