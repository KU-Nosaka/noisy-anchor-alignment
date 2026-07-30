# Anchor-Side Perturbation: Reproducibility Repository

This repository contains the minimal code and frozen artifacts needed to reproduce the experiments and figures for the paper *Anchor-Side Perturbation for Anchor-Aligned Independent Geometric Data Perturbation*.

It supports two distinct workflows:

1. **Reproduce the paper figures from frozen results.** This is CPU-only, does not download MNIST or CelebA, and does not retrain any model.
2. **Rerun the experiments.** The numbered notebooks are designed for Google Colab with persistent Google Drive storage and resumable, per-fit checkpoints.

Raw datasets, downloaded model weights, training checkpoints, caches, and duplicate result formats are deliberately excluded. The complete observations used in the paper are retained as compressed result tables. The available run, data-allocation, linkage, completion, and environment records are included alongside them; the exact coverage of those records is documented in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Quick figure reproduction

Python 3.12 or newer is recommended. The recorded publication figures were rendered with Python 3.14.2, NumPy 2.5.1, Matplotlib 3.11.1, and Pillow 12.3.0.

```bash
python -m venv .venv
python -m pip install -r requirements-figures.txt
python scripts/validate_repository.py
python scripts/reproduce_figures.py
```

On Windows, activate the environment with `.venv\Scripts\activate`; on macOS or Linux, use `source .venv/bin/activate`.

The command writes the manuscript-ready PDFs to:

```text
build/paper_figures/main/
build/paper_figures/supplement/
```

The source observations, spline coordinates, and renderer metadata are retained under `build/analysis/`.

## Full experiment reproduction

Upload the notebooks in `notebooks/` to Colab and follow the stage-specific controls in [notebooks/README.md](notebooks/README.md). The numerical order is:

1. `01_mnist_main.ipynb`
2. `02_mnist_supplemental_noise.ipynb`
3. `03_celeba_participant_sweep.ipynb`
4. `04_celeba_p010_supplemental_noise.ipynb`
5. `05_celeba_p030_p050_low_v.ipynb`
6. `06_celeba_p010_reconstruction_replay.ipynb`

The notebooks mount Google Drive and use atomic stage/checkpoint files. Set each notebook's result root to a new, empty Drive directory for an independent rerun. The CelebA stages are intended for a high-memory A100 runtime; interrupted stages can be resumed on a different runtime without discarding completed fits.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the exact stage composition, data preparation, randomization, and expected outputs.

## Repository contents

```text
notebooks/       Self-contained Colab experiment notebooks
results/         Compressed complete result records and provenance manifests
figure_inputs/   Compact lossless inputs for qualitative reconstruction grids
figures/         Deterministic publication-figure renderers
scripts/         One-command reproduction and validation
docs/            Experiment, figure, data, and provenance documentation
```

The correspondence between paper figures and scripts is listed in [docs/FIGURES.md](docs/FIGURES.md). `ARTIFACT_MANIFEST.json` records the size and SHA-256 digest of every other distributed file.

## Data and pretrained models

- MNIST is downloaded by `torchvision`.
- CelebA images are obtained by the notebooks from the Kaggle CelebA mirror used in the recorded runs; official identity metadata are downloaded separately and verified by the recorded manifests.
- The frozen face auditor uses `facenet-pytorch`'s InceptionResnetV1 weights pretrained on VGGFace2.

The source datasets, model weights, and caches are not redistributed. The three compact qualitative-input archives do contain selected dataset-derived source and reconstruction tiles so that the exact paper grids can be rebuilt without downloading or rerunning the datasets. Those tiles remain subject to the applicable MNIST and CelebA terms. Users are responsible for checking the dataset and pretrained-model licenses before redistributing the repository or its figures.

## Scope and provenance

The frozen tables are the authoritative source for the values and figures reported in the paper. They contain 609 MNIST records, 407 CelebA records for the \(p=10\) analysis, 207 core records each for \(p=30\) and \(p=50\), and 100 focused low-anchor-noise records for each of \(p=30\) and \(p=50\).

Source-snapshot limitations affecting one MNIST supplemental runner and the historical CelebA qualitative replay code are documented transparently in [docs/PROVENANCE.md](docs/PROVENANCE.md). They do not affect regeneration of the reported figures from the frozen records and frozen qualitative inputs.

No distribution license is selected in this archival copy. Choose and add an appropriate code/content license before publishing the repository.
