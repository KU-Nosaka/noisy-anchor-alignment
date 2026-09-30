# Noisy Anchor Alignment

This repository contains experiment code, notebooks, and frozen artifacts for *Noisy Anchor Alignment*. The current manuscript evaluates **CelebA and VGGFace2 with a frozen CASIA-WebFace auditor**. The original MNIST/CelebA package is retained as a historical workflow.

## Current manuscript: four notebooks from downloaded datasets

Start with the downloaded images and metadata inside `data/raw/` in a clone of this repository. Each notebook performs preprocessing, identity allocation, **public SVD feature extraction**, anchor/design construction, and evaluation. No prior Drive study, prepared features, or historical training checkpoints are needed. Model weights are downloaded and checksum-verified automatically.

| Notebook | Open in Colab | Purpose |
|---|---|---|
| [10 — CelebA privacy–utility](notebooks/current/10_celeba_casia_privacy_utility.ipynb) | [Colab](https://colab.research.google.com/github/KU-Nosaka/noisy-anchor-alignment/blob/main/notebooks/current/10_celeba_casia_privacy_utility.ipynb) | All `p = 2, 5, 10, 20, 50`; both noise placements |
| [11 — VGGFace2 privacy–utility](notebooks/current/11_vggface2_casia_privacy_utility.ipynb) | [Colab](https://colab.research.google.com/github/KU-Nosaka/noisy-anchor-alignment/blob/main/notebooks/current/11_vggface2_casia_privacy_utility.ipynb) | All five participant counts; both noise placements |
| [12 — CelebA MP/AM/OP comparison](notebooks/current/12_celeba_casia_attack_comparison.ipynb) | [Colab](https://colab.research.google.com/github/KU-Nosaka/noisy-anchor-alignment/blob/main/notebooks/current/12_celeba_casia_attack_comparison.ipynb) | 500 uniform anchor-noise draws at `p=10`; no utility training |
| [13 — VGGFace2 MP/AM/OP comparison](notebooks/current/13_vggface2_casia_attack_comparison.ipynb) | [Colab](https://colab.research.google.com/github/KU-Nosaka/noisy-anchor-alignment/blob/main/notebooks/current/13_vggface2_casia_attack_comparison.ipynb) | 500 uniform anchor-noise draws at `p=10`; no utility training |

Use a T4 runtime for each dataset. The main studies use 400-dimensional representations, 100 identities per participant, the original CNN, and spectral alignment. They target 20 accepted realizations per noise placement in each linkage bin `[0,15)`, `[15,25)`, `[25,35)`, `[35,45)`, `[45,100]` percent, with a 1,500-proposal cap per placement and participant count. This is **linkage-calibrated sampling**. The supplementary notebooks instead use 500 unconditional uniform scales in `[0,0.08)` and compare all three attacks on the same uploads.

See [the current input layouts, settings, execution steps, and resume rules](docs/CURRENT_CASIA_EXPERIMENTS.md). VGGFace2 requires its training archive, official metadata, and MAAD-Face Smiling labels. The notebooks explain each stage with readable cells; supporting code is in `reproduction/casia/`. Generated data, features, model weights, and checkpoints stay outside Git.

## Historical package

It supports two distinct workflows:

1. **Reproduce the paper figures from frozen results.** This is CPU-only, does not download MNIST or CelebA, and does not retrain any model.
2. **Rerun the experiments.** The numbered notebooks are designed for Google Colab with persistent Google Drive storage and resumable, per-fit checkpoints.

Raw datasets, downloaded model weights, training checkpoints, caches, and duplicate result formats are deliberately excluded. The complete observations used in the paper are retained as compressed result tables. The available run, data-allocation, linkage, completion, and environment records are included alongside them; the exact coverage of those records is documented in [docs/PROVENANCE.md](docs/PROVENANCE.md).

The manuscript names the anchor-noise method *NAA-GDP* (noisy-anchor-aligned GDP) and the noiseless anchor-aligned variants *AA-GDP*; the code, notebooks, and frozen result tables keep the identifiers of the recorded runs (`pa_i_gdp`, `aa_i_gdp`, `aa_i_gdp_an`, `c_gdp_an`, and the attack names `PA-OP`, `PA-MP`, `PA-AM`), and the figure renderers map them to the manuscript's labels.

## Historical figure reproduction

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

## Historical experiment reproduction

Upload the notebooks in `notebooks/` to Colab and follow the stage-specific controls in [notebooks/README.md](notebooks/README.md). The primary experiment order is:

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

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the exact stage composition, data preparation, randomization, and expected outputs.

## Repository contents

```text
notebooks/       Colab notebooks; current/ contains the four current CASIA workflows
reproduction/    Readable raw-input preparation, features, attacks, training, and figure code
results/         Compressed complete result records and provenance manifests
figure_inputs/   Compact lossless inputs for qualitative reconstruction grids
figures/         Deterministic publication-figure renderers; latex/ holds the layered participant-count figure body
scripts/         One-command reproduction and validation
docs/            Experiment, figure, data, and provenance documentation
```

The correspondence between paper figures and scripts is listed in [docs/FIGURES.md](docs/FIGURES.md). `ARTIFACT_MANIFEST.json` records the size and SHA-256 digest of every other distributed file.

## Historical data and pretrained models

- MNIST is downloaded by `torchvision`.
- CelebA images are obtained by the notebooks from the Kaggle CelebA mirror used in the recorded runs; official identity metadata are downloaded separately and verified by the recorded manifests.
- The frozen face auditor uses `facenet-pytorch`'s InceptionResnetV1 weights pretrained on VGGFace2.

The source datasets, model weights, and caches are not redistributed. The three compact qualitative-input archives do contain selected dataset-derived source and reconstruction tiles so that the exact paper grids can be rebuilt without downloading or rerunning the datasets. Those tiles remain subject to the applicable MNIST and CelebA terms. Users are responsible for checking the dataset and pretrained-model licenses before redistributing the repository or its figures.

## Scope and provenance

The frozen historical tables are the authoritative source for the earlier MNIST/CelebA figures. They contain 609 MNIST records, 407 CelebA records for the historical \(p=10\) analysis, and 103 records for each of \(p\in\{2,5,10,20,50\}\) in that participant-count replay. Each replay condition consists of 100 anchor-noise observations and three deterministic controls. The paired low-noise stratum of Notebook 09 adds 100 anchor-noise observations per participant count; its compact per-draw tables are distributed under `results/celeba/participant_replay/supplemental/`. The current CASIA notebooks generate new result records in their own output directories; they do not read these historical tables as experimental inputs.

Source-snapshot limitations affecting one MNIST supplemental runner and the historical CelebA qualitative replay code are documented transparently in [docs/PROVENANCE.md](docs/PROVENANCE.md). They do not affect regeneration of the reported figures from the frozen records and frozen qualitative inputs.

## Citation

Citation metadata for this reproducibility package are provided in [`CITATION.cff`](CITATION.cff). GitHub's **Cite this repository** control can export the citation in common formats.

## Licensing

This is a mixed-license repository:

- original source code and notebooks are licensed under the [MIT License](LICENSE);
- documentation and generated numerical results are licensed under [CC BY 4.0](LICENSE-DATA.md); and
- dataset-derived image tiles, pretrained model assets, and other third-party materials are excluded from those grants and remain subject to their original terms.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the scope and reuse restrictions.
