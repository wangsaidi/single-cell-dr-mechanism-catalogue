# Archived analysis inputs

The large evaluated inputs are distributed as a checksum-verified GitHub Release asset rather than stored in Git history. Run `python -m analysis.download_inputs` from the repository root. The archive installs processed AnnData objects, expression objects used for marker analyses, fixed anchor embeddings, dimensionality-sensitivity embeddings, scVI outputs and a read-only copy of the reference result tables. Individual files are checked against `analysis/ARCHIVE_SHA256SUMS.txt`.

Raw biological datasets remain available from the public sources listed in `metadata/public_data_sources.csv`. The optional `full-refit` profile reacquires them through Scanpy and scvi-tools before preprocessing.
