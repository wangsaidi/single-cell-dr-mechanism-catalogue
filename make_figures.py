from __future__ import annotations

import subprocess
import sys


MODULES = (
    "figures.plot_figure_2",
    "figures.plot_figures_3_4",
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
)


def main() -> None:
    # Matplotlib configuration is process-global. Running each figure group in
    # a fresh interpreter prevents one script's rcParams from affecting later
    # layouts and makes the combined entry point match standalone execution.
    for module in MODULES:
        subprocess.run([sys.executable, "-m", module], check=True)


if __name__ == "__main__":
    main()
