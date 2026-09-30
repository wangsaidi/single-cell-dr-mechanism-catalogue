"""Run archived-input verification or optional model refitting."""

from __future__ import annotations

import argparse
import subprocess
import sys


ARCHIVED_STEPS = [
    ("record_environment", ["-m", "analysis.record_environment"]),
    ("verify_inputs", ["-m", "analysis.verify_reproducibility", "--inputs-only"]),
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
    ("verify_results", ["-m", "analysis.verify_reproducibility"]),
    ("export_figure4", ["-m", "analysis.export_figure4_source_data"]),
]


def full_refit_steps() -> list[tuple[str, list[str]]]:
    steps = [
        ("prepare_objects", ["-m", "analysis.scripts.prepare_analysis_objects"]),
    ]
    for seed in range(5):
        steps.append(
            (
                f"anchor_embeddings_seed_{seed}",
                ["-m", "analysis.scripts.generate_anchor_embeddings", "--seed", str(seed), "--force"],
            )
        )
    steps.extend(
        [
            ("robustness_embeddings", ["-m", "analysis.scripts.generate_robustness_embeddings"]),
            ("known_truth_stress_tests", ["-m", "analysis.scripts.generate_known_truth_stress_tests"]),
            ("fit_scvi", ["-m", "analysis.scripts.fit_scvi", "--force"]),
        ]
    )
    return steps + ARCHIVED_STEPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["archived", "refit-scvi", "full-refit"],
        default="archived",
        help="Archived is the reviewer-facing exact-input workflow.",
    )
    parser.add_argument("--from-step")
    parser.add_argument("--through-step")
    args = parser.parse_args()

    if args.profile == "archived":
        steps = ARCHIVED_STEPS
    elif args.profile == "refit-scvi":
        steps = [("fit_scvi", ["-m", "analysis.scripts.fit_scvi", "--force"])] + ARCHIVED_STEPS
    else:
        steps = full_refit_steps()

    names = [name for name, _ in steps]
    start = names.index(args.from_step) if args.from_step else 0
    stop = names.index(args.through_step) if args.through_step else len(steps) - 1
    if start > stop:
        raise SystemExit("--from-step must precede --through-step")

    for name, arguments in steps[start : stop + 1]:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, *arguments], check=True)


if __name__ == "__main__":
    main()
