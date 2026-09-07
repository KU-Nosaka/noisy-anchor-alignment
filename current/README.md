# Current manuscript code

This package supplies the LFWA and CelebA experiment notebooks, the synthetic C-GDP equivalence check, and figure-generation code. Current result tables and generated figures remain separately in the author's Google Drive; they are not included in this checkout.

- [Fresh experiment setup](docs/EXPERIMENTS.md)
- [Active Colab notebooks](notebooks/README.md)
- [Figure generation with separately supplied results](docs/FIGURES.md)
- [Source provenance and validation](docs/PROVENANCE.md)

Run the notebooks with the required source datasets to create your own results. If you already have a completed result directory, pass its path with `--results-root` to `current/scripts/validate_results.py` or `current/scripts/reproduce_figures.py`. The scripts can then validate or plot those files without retraining models.
