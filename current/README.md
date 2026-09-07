# Current manuscript code and results

This package supplies the LFWA and CelebA experiment notebooks, the synthetic C-GDP equivalence check, figure-generation code, and the completed numerical [result bundle](results/).

- [Completed results and provenance](results/README.md)
- [Fresh experiment setup](docs/EXPERIMENTS.md)
- [Active Colab notebooks](notebooks/README.md)
- [Figure generation](docs/FIGURES.md)
- [Source provenance and validation](docs/PROVENANCE.md)

From the repository root, run `python current/scripts/validate_results.py` to check the bundled tables and `python current/scripts/reproduce_figures.py` to render the six quantitative figures without retraining. Install `current/requirements-figures.txt` first. Both scripts default to `current/results/`; use `--results-root` for an independent completed run.

Full experimental reruns and qualitative galleries require the source datasets and working deployment artifacts described in the setup guide. Those image inputs, model weights, and caches are separate from this compact numerical bundle.
