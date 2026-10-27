from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path(__file__).resolve().parent
MODULES = (
    "figures.plot_figure_2",
    "figures.plot_figures_3_4",
    "figures.plot_figure_4_downstream",
    "figures.plot_figures_5_6",
    "figures.plot_figure_7",
    "figures.plot_supplementary_figure_s1",
    "figures.plot_supplementary_figure_s2",
    "figures.plot_supplementary_figures_s3_s6",
    "figures.plot_supplementary_figures_s7_s8",
    "figures.plot_supplementary_figure_s9",
    "figures.plot_supplementary_figures_s10_s11",
    "figures.plot_supplementary_figure_s12",
    "figures.plot_supplementary_figure_s13",
    "figures.plot_supplementary_figures_s15_s16",
    "figures.plot_supplementary_figure_s17",
)


REVISION_PLOTS = (
    ("scripts.plot_deep_controls", "deep_controls", (
        "matched_endpoint_scores.csv",
        "matched_population_scores.csv",
        "checkpoint_representation_stability.csv",
    )),
    ("scripts.plot_reference_controls", "reference_attribution", (
        "reference_root_concordance.csv",
        "crossed_root_agreement.csv",
        "fixed_side_contrasts.csv",
        "matched_root_seed_summary.csv",
        "marker_consistency.csv",
        "legacy_geometry_continuum.csv",
        "Supplementary_Fig_S20_conceptual_source.json",
    )),
)


def revision_commands(source_root: Path, output_dir: Path) -> list[list[str]]:
    """Require public plotting inputs before launching any figure subprocess."""
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if (source_root == output_dir or source_root.is_relative_to(output_dir)
            or output_dir.is_relative_to(source_root)):
        raise ValueError("Revision output and source directories must not overlap.")
    missing = []
    commands = []
    for module, directory, filenames in REVISION_PLOTS:
        script = CODE_ROOT.joinpath(*module.split(".")).with_suffix(".py")
        if not script.is_file():
            missing.append(str(script))
        source = source_root / directory
        for filename in filenames:
            candidate = source / filename
            if not candidate.is_file() or candidate.stat().st_size == 0:
                missing.append(str(candidate))
        commands.append([
            sys.executable, "-m", module, "--source-dir", str(source),
            "--output-dir", str(output_dir),
        ])
    if missing:
        raise ValueError(
            "S18-S20 plotting requires the portable scripts and nonempty public "
            "source files below. Publish/export completed controls first; this "
            "command does not fit models or create missing evidence:\n  "
            + "\n  ".join(missing)
        )
    return commands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate supplied figure tables; no model training or simulations.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--include-revision-supplements", action="store_true",
                           help="Append S18-S20 from completed public control sources.")
    selection.add_argument("--revision-supplements-only", action="store_true",
                           help="Plot only S18-S20; do not refresh or rebuild established figures.")
    parser.add_argument("--revision-source-root", type=Path,
                        default=CODE_ROOT / "data/source_data",
                        help="Parent of deep_controls/ and reference_attribution/.")
    parser.add_argument("--revision-output-dir", type=Path,
                        default=CODE_ROOT / "outputs/supplementary_figures",
                        help="Output directory for S18-S20 only.")
    parser.add_argument("--check-only", action="store_true",
                        help="With a revision selection, run source validation without plotting or writes.")
    args = parser.parse_args(argv)
    revisions = args.include_revision_supplements or args.revision_supplements_only
    if args.check_only and not revisions:
        parser.error("--check-only requires a revision-supplements selection.")
    try:
        extra_commands = (
            revision_commands(args.revision_source_root, args.revision_output_dir)
            if revisions else []
        )
    except ValueError as error:
        parser.error(str(error))
    if args.check_only:
        for command in extra_commands:
            subprocess.run([*command, "--check-only"], check=True, cwd=CODE_ROOT)
        return
    if not args.revision_supplements_only:
        subprocess.run(
            [sys.executable, "-m", "analysis.scripts.refresh_biological_plot_inputs"],
            check=True, cwd=CODE_ROOT,
        )
        # Separate processes prevent one group's rcParams affecting later layouts.
        for module in MODULES:
            subprocess.run([sys.executable, "-m", module], check=True, cwd=CODE_ROOT)
    for command in extra_commands:
        subprocess.run(command, check=True, cwd=CODE_ROOT)


if __name__ == "__main__":
    main()
