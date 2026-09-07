# Current manuscript figure map

The current manuscript uses the figures below. The overview schematic (Figure 1) is manuscript artwork; it is not an experimental plot.

| Location | Output | Source |
|---|---|---|
| Main Figure 2 | `lfwa_privacy_utility.pdf` | LFWA full sweep at p = 5 |
| Main Figure 3(a) | `face_reconstruction_gallery_lfwa.pdf` | LFWA saved seed 0, realization 0 |
| Main Figure 3(b) | `face_reconstruction_gallery_celeba.pdf` | CelebA saved seed 0, realization 0, p = 10 |
| Main Figure 4 | `celeba_privacy_utility.pdf` | Exact p = 10 subset of merged CelebA results |
| Main Figure 5 | `celeba_participant_sensitivity.pdf` | Merged CelebA p = 2, 5, 10, 20, 50 |
| Supplement | `celeba_gpm_diagnostics.pdf` | The 200 positive-noise CelebA p = 10 alignments |
| Supplement | `lfwa_anchor_noise_identity_leakage.pdf` | LFWA MP, AM, OP linkage over all ten levels |
| Supplement | `celeba_noise_identity_leakage.pdf` | CelebA OP and disclosed-common-transform inversion |

## Quantitative figures from the bundled results

The completed numerical result bundle is included in [`../results/`](../results/). It contains the inputs for all six quantitative plots and preserves the reviewed scalar CSV values. Generated plots are written locally. For an independent run, keep the notebooks' directory structure when copying results out of Drive.

The scripts default to `current/results/`. An alternative directory supplied with `--results-root` must contain these required files:

| Subdirectory | Required inputs |
|---|---|
| `lfwa_full_sweep/` | `raw_results.csv`, `summary.csv`, `attack_results.csv`, `attack_summary.csv`, `protocol.json`, `progress.json` |
| `celeba_parallel/merged/` | `raw_results.csv`, `summary.csv`, `attack_results.csv`, `privacy_utility_by_attack.csv`, `attack_summary.csv`, `progress.json` |
| `equivalence/` | `equivalence_raw.csv` |
| `figures/manuscript_inputs/` | `celeba_gpm_diagnostics.csv` exported by the figure notebook |

The bundled compact export places the diagnostics CSV in `celeba_parallel/merged/`. The scripts prefer the original figure-input location when both exist. Missing inputs produce an error identifying the required files.

From the repository root:

```bash
python -m pip install -r current/requirements-figures.txt
python current/scripts/validate_results.py
python current/scripts/reproduce_figures.py
```

The script writes the six quantitative PDFs and PNG previews to `build/current_figures/`, with the seed-level plotting statistics and a source-hash manifest. It does not fit models or load images. The plotting code derives from the current Colab figure notebook; convergence labels are calculated from the supplied diagnostics.

The positive-noise conditions first average four realizations within seed; error bars and utility bands report sample SD across five seed means. LFWA and CelebA privacy–utility curves connect noise levels in their evaluated order and include each series' zero-noise control. I-GDP and Local-only appear only as horizontal utility references. The participant-count figure pairs the same target, queries, and anchor-noise draws across the nested participant counts.

## Full Colab workflow and qualitative galleries

Use `current/notebooks/Experiment_Figures.ipynb` after the experiment results are complete. Mount Drive, load its helper definitions, export the saved gallery inputs, render the numerical figures, and run the cell labeled **Figure 3: separate LFWA and CelebA subfigures**. The earlier combined gallery output is retained for compatibility; the manuscript includes the two separate PDFs.

Gallery export requires the source images and saved deployment data from the experiment runs. The source images and full saved deployments are separate working inputs; the compact public result tables alone do not contain the reconstruction tiles. The notebook reconstructs the first ten distinct saved queries in deployment order, without selecting images by quality. Each column preserves its source face across methods; positive-noise scales increase across columns. Displayed linkage percentages cover all 1,000 saved ten-way lineups for the selected condition, rather than the individual tile or a five-seed average.

For an LFWA-only update, mount Drive and run the final **Refresh LFWA figures only** cell. It validates the full sweep and saved gallery provenance before writing the LFWA trade-off, LFWA attack comparison, and LFWA Figure 3 panel. This cell also writes a ZIP with its statistics and provenance manifest.
