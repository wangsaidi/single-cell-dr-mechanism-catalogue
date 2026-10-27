"""Explicit reduced execution budgets, never enabled by the publication profiles."""

import os

SMOKE = os.environ.get("SCDR_EXECUTION_SCALE") == "smoke"
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
OUTPUT_DIMENSIONS = [2, 10] if SMOKE else [2, 5, 10, 20]
RESOLUTIONS = [0.4, 1.0] if SMOKE else [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
