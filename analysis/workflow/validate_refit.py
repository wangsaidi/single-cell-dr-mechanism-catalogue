"""Validate refit completeness, not equality to historical stochastic binary files."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from analysis.paths import LOGS_DIR, RESULTS_DIR, SOURCE_RESULTS_DIR
from analysis.execution import SMOKE, SEEDS


def validate(logs: Path, results: Path, source: Path, profile: str) -> list[str]:
    errors = []
    for filename in ("input_audit_summary.json", "scvi_output_audit_summary.json", "revision_result_audit.json"):
        path = logs / filename
        if not path.is_file():
            errors.append(f"missing audit: {path}")
            continue
        if not json.loads(path.read_text(encoding="utf-8")).get("audit_passed"):
            errors.append(f"failed audit: {path}")
    if profile == "full-refit":
        for seed in SEEDS:
            failures = logs / f"anchor_embedding_failures_seed{seed}.json"
            if not failures.is_file() or json.loads(failures.read_text(encoding="utf-8")):
                errors.append(f"missing/failed anchor fits: {failures}")
            path = source / f"fig3_family_local_label_metrics_seed{seed}.csv"
            if not path.is_file():
                errors.append(f"missing refitted seed metrics: {path}")
        for filename in ("fig6_mechanism_simulation_suite.csv", "fig6_output_dimension_response.csv",
                         "fig6_upstream_pca_response.csv", "fig6_dropout_noise_response.csv", "fig6_known_truth_simulation.csv"):
            path = source / filename
            if not path.is_file():
                errors.append(f"missing refit result: {path}")
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or any(row.get("support") == "not_run" or row.get("metric") == "run_status" for row in rows):
                errors.append(f"empty or incomplete fit coverage: {path}")
            if filename == "fig6_mechanism_simulation_suite.csv":
                methods = {"PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"}
                scenarios = {"linear_low_rank", "nonlinear_manifold", "branching_trajectory", "dropout_stress", "batch_shift", "rare_population"}
                replicates = [0] if SMOKE else range(5)
                expected = {(m, s, str(r)) for m in methods for s in scenarios for r in replicates}
                observed = {(row.get("method"), row.get("scenario"), row.get("replicate")) for row in rows}
                if observed != expected:
                    errors.append(f"simulation method/scenario/replicate coverage mismatch: {path}")
    if not (results / "calibration" / "metadata.json").is_file():
        errors.append("missing calibration/metadata.json")
    panels = list((results / "figure4_downstream").glob("*.csv"))
    if len(panels) != 6:
        errors.append(f"expected six Figure 4 panel tables, found {len(panels)}")
    return errors


def main() -> None:
    validation_mode = os.environ.get("SCDR_WORKFLOW_PROFILE", "full-refit")
    # Smoke uses full-refit coverage checks, but is not a publication-scale fit.
    profile = "smoke" if SMOKE else validation_mode
    errors = validate(LOGS_DIR, RESULTS_DIR, SOURCE_RESULTS_DIR, validation_mode)
    report = {"profile": profile, "validation_mode": validation_mode,
              "passed": not errors, "errors": errors,
              "execution_scale": "smoke_not_publication" if SMOKE else "publication",
              "validation": "schema/coverage/finite-output audits and current-run object hashes",
              "historical_binary_hash_comparison": False,
              "stochastic_agreement_or_convergence_established": False}
    (LOGS_DIR / "refit_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
