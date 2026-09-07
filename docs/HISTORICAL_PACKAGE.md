# Historical experiment package

This page preserves the reproduction instructions for the earlier MNIST and CelebA manuscript. For the current LFWA/CelebA manuscript, use [the current package](../current/).

# Anchor-Side Perturbation: Reproducibility Repository

This repository contains the minimal code and frozen artifacts needed to reproduce the experiments and figures for the paper *Anchor-Side Perturbation for Anchor-Aligned Independent Geometric Data Perturbation*.

It supports two distinct workflows:

1. **Reproduce the paper figures from frozen results.** This is CPU-only, does not download MNIST or CelebA, and does not retrain any model.
2. **Rerun the experiments.** The numbered notebooks are designed for Google Colab with persistent Google Drive storage and resumable, per-fit checkpoints.

Raw datasets, downloaded model weights, training checkpoints, caches, and duplicate result formats are deliberately excluded. The complete observations used in the paper are retained as compressed result tables. The available run, data-allocation, linkage, completion, and environment records are included alongside them; the exact coverage of those records is documented in [docs/PROVENANCE.md](PROVENANCE.md).

The manuscript names the anchor-noise method *NAA-GDP* (noisy-anchor-aligned GDP) and the noiseless anchor-aligned variants *AA-GDP*; the code, notebooks, and frozen result tables keep the identifiers of the recorded runs (`pa_i_gdp`, `aa_i_gdp`, `aa_i_gdp_an`, `c_gdp_an`, and the attack names `PA-OP`, `PA-MP`, `PA-AM`), and the figure renderers map them to the manuscript's labels.

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

Upload the notebooks in `notebooks/` to Colab and follow the stage-specific controls in [notebooks/README.md](../notebooks/README.md). The primary experiment order is:

1. `01_mnist_main.ipynb`
2. `02_mnist_supplemental_noise.ipynb`
3. `03_celeba_p010_main.ipynb`
4. `04_celeba_p010_supplemental_noise.ipynb`
5. `05_celeba_p010_reconstruction_replay.ipynb`

The revised Section 5.3.3 participant-count replay is split into three independent notebooks that may run in parallel:

6. `06_celeba_participant_replay_p050.ipynb`
7. `07_celeba_participant_replay_p002_p020.ipynb`
8. `08_celeba_participant_replay_p005_p010.ipynb`
9. `09_celeba_participant_replay_supplemental_v0_0p05.ipynb`

Together, Notebooks 06--08 evaluate \(p\in\{2,5,10,20,50\}\), with 100 paired draws per participant count and \(v\sim\mathrm{Uniform}(0,0.1)\). Only the corrected `p005_p010` shard is distributed; the failed pre-correction attempt is not part of this repository. Notebook 09 adds, for every participant count, 100 paired supplemental draws in the low-noise regime \(0<v<0.05\): draw \(j\) reuses the quantile, noise tensors, and seeds of published draw \(j\) at exactly half its amplitude, after rebuilding the reviewed modules from the pinned repository commit and verifying the regenerated deployment against the published hashes. The participant-count figure uses all 200 draws per condition.

The notebooks mount Google Drive and use atomic stage/checkpoint files. Set each notebook's result root to a new, empty Drive directory for an independent rerun. The CelebA stages are intended for a high-memory A100 runtime; interrupted stages can be resumed on a different runtime without discarding completed fits.

See [docs/EXPERIMENTS.md](EXPERIMENTS.md) for the exact stage composition, data preparation, randomization, and expected outputs.

## Repository contents

```text
notebooks/       Self-contained Colab experiment notebooks
results/         Compressed complete result records and provenance manifests
figure_inputs/   Compact lossless inputs for qualitative reconstruction grids
figures/         Deterministic publication-figure renderers; latex/ holds the layered participant-count figure body
scripts/         One-command reproduction and validation
docs/            Experiment, figure, data, and provenance documentation
```

The correspondence between paper figures and scripts is listed in [docs/FIGURES.md](FIGURES.md). `ARTIFACT_MANIFEST.json` records the size and SHA-256 digest of every other distributed file.

## Data and pretrained models

- MNIST is downloaded by `torchvision`.
- CelebA images are obtained by the notebooks from the Kaggle CelebA mirror used in the recorded runs; official identity metadata are downloaded separately and verified by the recorded manifests.
- The frozen face auditor uses `facenet-pytorch`'s InceptionResnetV1 weights pretrained on VGGFace2.

The source datasets, model weights, and caches are not redistributed. The three compact qualitative-input archives do contain selected dataset-derived source and reconstruction tiles so that the exact paper grids can be rebuilt without downloading or rerunning the datasets. Those tiles remain subject to the applicable MNIST and CelebA terms. Users are responsible for checking the dataset and pretrained-model licenses before redistributing the repository or its figures.

## Scope and provenance

The frozen tables are the authoritative source for the values and figures reported in the paper. They contain 609 MNIST records, 407 CelebA records for the primary \(p=10\) analysis, and 103 records for each of \(p\in\{2,5,10,20,50\}\) in the revised participant-count replay. Each replay condition consists of 100 anchor-noise observations and three deterministic controls. The paired low-noise stratum of Notebook 09 adds 100 anchor-noise observations per participant count; its compact per-draw tables are distributed under `results/celeba/participant_replay/supplemental/`, and the complete per-fit records will be added in the schema of the published tables.

Source-snapshot limitations affecting one MNIST supplemental runner and the historical CelebA qualitative replay code are documented transparently in [docs/PROVENANCE.md](PROVENANCE.md). They do not affect regeneration of the reported figures from the frozen records and frozen qualitative inputs.

## Citation

Citation metadata for this reproducibility package are provided in [`CITATION.cff`](../CITATION.cff). GitHub's **Cite this repository** control can export the citation in common formats.

## Licensing

This is a mixed-license repository:

- original source code and notebooks are licensed under the [MIT License](../LICENSE);
- documentation and generated numerical results are licensed under [CC BY 4.0](../LICENSE-DATA.md); and
- dataset-derived image tiles, pretrained model assets, and other third-party materials are excluded from those grants and remain subject to their original terms.

See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the scope and reuse restrictions.

