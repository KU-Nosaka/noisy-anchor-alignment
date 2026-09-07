# Current experiment setup

The six notebooks in `current/notebooks/` are the manuscript's current execution entry points. They embed their own implementations and run in Google Colab. Keep the Drive working folder named `Noisy Anchor Alignment`; its name is part of the saved paths used by the recorded runs.

## Execution order

| Stage | Notebook | Runtime and action | Result directory |
|---|---|---|---|
| Algebraic implementation check | `C_GDP_AA_GDP_Equivalence.ipynb` | CPU; run cells in order; optional last cell saves to Drive | `results/equivalence/` |
| CelebA shared preparation | `CelebA_P50.ipynb` | GPU; run through **Prepare shared deployments** and wait for completion | `results/celeba_parallel/shared/` |
| CelebA workers | `CelebA_P50.ipynb`, `CelebA_P2_P20.ipynb`, `CelebA_P5_P10.ipynb` | Separate GPU runtimes; then run each worker | `results/celeba_parallel/jobs/` |
| CelebA attack replay and reporting | `CelebA_P5_P10.ipynb` | Run **Replay and merge** after every worker completes | `results/celeba_parallel/merged/` |
| LFWA full sweep | `LFWA_Privacy_Utility.ipynb` | A100 GPU; Run all | `results/lfwa_full_sweep/` |
| Manuscript figures | `Experiment_Figures.ipynb` | Mount Drive; run against completed saved results | `results/figures/manuscript/` |

Three tabs connected to the same runtime do not constitute three GPU workers. Never launch two runtimes for the same worker directory. Rerunning the notebooks resumes verified checkpoints. Source, configuration, environment, deployment and auditor checks prevent mixing incompatible results.

The notebooks define the full experimental schedules and write their own progress and completion records. Local aggregate rows summarize participant models and are not additional fits. Current saved tables and completion reports are not included in the GitHub checkout.

## Existing Drive experiment

When resuming an existing Drive experiment migrated from the serial runner, the active workers read any reused fit records under `results/celeba/`. They also require that run's protocol, source, environment and auditor manifests. Shared deployment artifacts and worker-specific checkpoints belong under `results/celeba_parallel/`. These files are private working inputs, separate from the public code checkout.

Deleting the superseded serial notebook does not remove any of these dependencies. Preserve the existing `results/celeba/`, `results/celeba_parallel/`, `results/lfwa_full_sweep/`, `results/equivalence/` and figure-input directories. The older `results/lfwa/` contains the first three-level run and a public source-archive cache used by the full-sweep notebook.

## Fresh public installation

A GitHub checkout contains the current code and source snapshots. It does not contain the current result tables, image archives, trained models, generated figures, or fitted deployment caches. Obtain the dataset inputs separately, then place the notebooks in your own Drive working directory.

The CelebA configuration reads the historical raw cache at:

`/content/drive/MyDrive/NAA-GDP/CelebA_PublicSVD_identity100_cache_v1`

This directory must contain:

- `raw_archive/celeba_kaggle_mirror.zip`, the raw source archive containing the images and exactly one `list_attr_celeba.csv`;
- `raw_archive/celeba_kaggle_mirror_manifest.json`, including that archive's `bytes` and `sha256`;
- official CelebA identity metadata under one of the recognized names: `identity_CelebA.txt`, `identity_CelebA.csv`, `list_identity_celeba.txt`, or `list_identity_celeba.csv`.

The data pipeline joins attributes and identities by filename and checks all 202,599 images and 10,177 identities. It verifies archive/cache hashes and records preprocessing and image-order provenance. A different archive representation or input manifest creates a different source identity; it must not be mixed with checkpoints from the manuscript run.

Verified `rgb64_shards/` and frozen public `torch_home/` weights may be reused as caches. They are optional: missing RGB pixels are reconstructed from the archive and the frozen auditor loads its public weights. These operations can take time and storage, but do not fit utility models.

The public **CelebA_P50** notebook adds a small initializer to the **parallel prepare** definitions. When `results/celeba/protocol.json` is absent and the destination contains no unidentified results or metadata, it:

1. Uses the original source-preparation and frozen-auditor implementations.
2. Records the current runtime environment and exact source/auditor identities.
3. Writes a protocol manifest with the unchanged scientific configuration and code identity.
4. Records the truthful status `initialized`, with **zero completed fits and zero result rows**.
5. Runs the original shared-deployment preparation to freeze all five projections and transformation banks.

The following worker cells perform the actual fitting. To initialize and prepare without utility training, run **CelebA_P50** through **Prepare shared deployments**, and stop before **Run this worker**. In the supplied notebook these are code-cell indices 1, 2, 4, 6, 8, 10, 12, 14 and 16 when counted from zero; prefer the section names in the Colab interface. Cell 18 starts the p50 worker.

For LFWA, the current notebook contains its own full initialization pipeline and source-download/audit code. Its required A100 check is deliberate. It can use `results/lfwa/cache/source/` for verified public archives and `results/celeba/cache/torch_home/` for the frozen auditor cache; it does not import scores or fitted models from the older LFWA experiment.

## Source provenance and the fresh-start addition

`current/source_snapshots/CelebA_P50.ipynb` preserves the exact cell source from the Drive notebook downloaded after the recorded experiment. `current/archive/CelebA_Privacy_Utility_and_Scaling.ipynb` preserves the superseded serial source. Outputs, execution counts and execution metadata are removed from these public copies. The archived notebook is for provenance; it is not an additional active execution entry point. `FRESH_START_SOURCE_MANIFEST.json` records original file hashes and original/published code-cell hashes.

The public P50 notebook embeds `current/scripts/prepare_celeba.py` verbatim. Its only executable changes relative to the recorded P50 source are this initializer, a call before the original protocol read, and acceptance of `initialized` alongside the original nonrunning statuses. The scientific configuration, model/attack/data definitions, schedule, seed logic, code-identity constants and numerical criteria remain unchanged.

For an existing protocol manifest, the initializer returns immediately without reading the GPU or writing files. The original preparation and worker checks still validate that run. The addition initializes a new run without claiming historical completion. Fresh reproductions record their own environment, source and generated deployment hashes.

The initializer was checked locally with synthetic metadata and stubbed source/auditor dependencies: fresh zero-fit counts and signatures; exact no-op behavior for existing initialized, interrupted and complete states; refusal to adopt orphaned metadata; cleanup and retry after missing input; unchanged scientific executable cells; and syntax validation. These checks perform no GPU computation or model training. A new full 1,525-fit experiment was not run as part of repository cleanup.

## Statistical and figure conventions

Both image experiments use five deployment seeds, ten positive anchor-noise levels, ten positive private-noise levels and four realizations per positive condition. Within each seed and condition, average the four realizations before computing the reported mean and sample SD across the five seed means. Noiseless controls occur once per seed. Preserve capped GPM runs and their convergence flags.

Main privacy–utility plots use OP linkage for NAA and the known-common-transform comparator for private noise. I-GDP and Local-only linkage is unmeasured. Reconstruction-gallery annotations instead refer to saved seed-0, realization-0 releases. The separate Figure 3 panels and LFWA refresh cell preserve this distinction.
