# Current experiment notebooks

These six notebooks supply the current manuscript workflow. Completed numerical tables are bundled in [`../results/`](../results/); full deployment caches and generated qualitative figures remain separate working artifacts. Open the GitHub notebook in Google Colab or upload it to Colab. The CPU equivalence check is independent; the image experiments use GPU runtimes and persistent Google Drive storage.

| Notebook | Execution role |
|---|---|
| `LFWA_Privacy_Utility.ipynb` | Full ten-level LFWA sweep at p = 5; resumable execution. |
| `CelebA_P50.ipynb` | Initialize and verify shared deployments, then run p = 50 and the local models. |
| `CelebA_P2_P20.ipynb` | Run the p = 2 and p = 20 worker after shared preparation. |
| `CelebA_P5_P10.ipynb` | Run p = 5 and p = 10, private-noise controls, attack replay, and the final merge. |
| `C_GDP_AA_GDP_Equivalence.ipynb` | Run 800 coupled synthetic comparisons across 160 deployments. |
| `Experiment_Figures.ipynb` | Read saved results, export galleries, and render manuscript figures. |

Use the [setup instructions](../docs/EXPERIMENTS.md) before starting the CelebA workers. Their shared preparation and bootstrap dependencies matter on a fresh Drive. Only run one writer for a given worker directory; the three different workers can run concurrently on separate runtimes. Wait for preparation to complete before starting the two follower workers. Once all three jobs finish, run the final replay-and-merge cell in `CelebA_P5_P10.ipynb` again if it previously reported incomplete peers.

The original serial CelebA runner is retained under [`../archive/`](../archive/) as a source and migration record. Do not run it concurrently with the parallel workflow. Files under `../source_snapshots/` preserve exact Drive source snapshots where the public notebook adds setup support.

The figure notebook has an LFWA-only final cell for refreshing saved, completed results after mounting Drive. This refresh checks the 420-row schedule, seed aggregation, and qualitative provenance. See the [figure guide](../docs/FIGURES.md) for the full rendering workflow and separate Figure 3 panels.

The default Drive project is `MyDrive/Noisy Anchor Alignment`. Preserve its directory structure, or update every corresponding configuration path for a separate rerun. Completed historical checkpoints retain their recorded source and configuration hashes; changing scientific parameters requires a new output directory.
