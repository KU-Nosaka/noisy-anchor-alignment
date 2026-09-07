# Noisy Anchor Alignment

Code for *Geometric Data Perturbation with Noisy Anchor Alignment for Privacy-Preserving Collaborative Learning*.

The current manuscript uses **LFWA and CelebA**. Its experiment notebooks and figure code are in [`current/`](current/). The current package distributes code; the saved experimental results remain separately in the author's Google Drive and are not included here.

## Current experiments

| Notebook | Purpose |
|---|---|
| [LFWA_Privacy_Utility.ipynb](current/notebooks/LFWA_Privacy_Utility.ipynb) | Full LFWA privacy–utility sweep at p = 5 |
| [CelebA_P50.ipynb](current/notebooks/CelebA_P50.ipynb) | Shared preparation, p = 50, and local models |
| [CelebA_P2_P20.ipynb](current/notebooks/CelebA_P2_P20.ipynb) | CelebA p = 2 and p = 20 |
| [CelebA_P5_P10.ipynb](current/notebooks/CelebA_P5_P10.ipynb) | CelebA p = 5 and p = 10, private noise, attack replay, and merge |
| [C_GDP_AA_GDP_Equivalence.ipynb](current/notebooks/C_GDP_AA_GDP_Equivalence.ipynb) | Synthetic numerical check of C-GDP equivalence |
| [Experiment_Figures.ipynb](current/notebooks/Experiment_Figures.ipynb) | Render manuscript figures from separately supplied saved results |

The image-experiment design uses five deployment seeds, ten positive anchor-noise levels from 0.006 to 0.060, and four realizations per level. Private noise uses ten standard deviations from 0.3 to 3.0 (CelebA at p = 10 only). The code averages the four realizations within each seed before computing the mean and sample standard deviation across seed means. Unmeasured I-GDP and Local-only linkage remains missing data.

## Run in Google Colab

Follow [`current/docs/EXPERIMENTS.md`](current/docs/EXPERIMENTS.md) to obtain the source datasets and initialize a fresh experiment. The notebook guide is [`current/notebooks/README.md`](current/notebooks/README.md). Experiment code writes resumable records to your own Google Drive. The default project folder is `MyDrive/Noisy Anchor Alignment`; use separate output paths for an independent rerun.

Run LFWA and the equivalence check independently. For CelebA, complete shared preparation in `CelebA_P50.ipynb`, then run the three disjoint workers on separate runtimes. Once all workers finish, run the final replay-and-merge cell in `CelebA_P5_P10.ipynb`.

## Use saved results

If you have the saved result directory from a completed run, the quantitative figures can be generated on CPU. Supply its local path explicitly:

```bash
python -m pip install -r current/requirements-figures.txt
python current/scripts/validate_results.py --results-root '/path/to/Noisy Anchor Alignment/results'
python current/scripts/reproduce_figures.py --results-root '/path/to/Noisy Anchor Alignment/results'
```

A fresh checkout does not contain those inputs. See [`current/docs/FIGURES.md`](current/docs/FIGURES.md) for the required file layout and Colab workflow. Figure 3 is generated as separate LFWA and CelebA subfigures. [`current/docs/PROVENANCE.md`](current/docs/PROVENANCE.md) describes source snapshots and validation conventions.

Dataset images, reconstruction tiles, current numerical result tables, generated figures, model weights, and training checkpoints are not distributed in `current/`. The supplied notebooks create these artifacts in your own working directory when run with the required data.

## Historical package

The original top-level `notebooks/`, `results/`, `figure_inputs/`, `figures/`, `scripts/`, and `docs/` are preserved for the earlier MNIST and CelebA experiments. Their figure numbers, spline analyses, and result counts refer to that earlier manuscript. Use [`current/`](current/) for the current LFWA/CelebA workflow. Historical reproduction instructions remain in [`docs/HISTORICAL_PACKAGE.md`](docs/HISTORICAL_PACKAGE.md).

## Citation and licensing

Use [`CITATION.cff`](CITATION.cff) to cite the code. Original source code and notebooks use the [MIT License](LICENSE); documentation and the previously distributed numerical results use [CC BY 4.0](LICENSE-DATA.md). Dataset-derived materials and third-party assets remain subject to their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
