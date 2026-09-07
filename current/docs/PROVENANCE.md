# Source provenance and validation

The current package publishes the LFWA/CelebA experiment code, figure workflow, and completed numerical [result bundle](../results/). The bundle includes scalar tables, completion records, attack and solver evidence, and compact provenance metadata. Image inputs, generated figures, and full deployment caches remain separate working artifacts. An independent run writes new artifacts to the user's own output directory.

All 39 reviewed bundle files are published unchanged. [SOURCE_MANIFEST.json](../results/SOURCE_MANIFEST.json) records original and packaged hashes and the documented metadata portability edits made when the bundle was assembled. Scalar CSV values are preserved; [VALIDATION.json](../results/VALIDATION.json) records the numerical checks.

## Code snapshots

The active notebooks derive from the Drive workflow, with outputs, execution counts, and execution metadata cleared for publication. The public CelebA P50 notebook adds fresh-start initialization support. Its original source and the superseded serial source are preserved in `source_snapshots/` and `archive/`; see [the setup guide](EXPERIMENTS.md). The source manifest records the original file and code-cell hashes so the setup adaptation can be inspected independently of experimental outputs.

The archived serial notebook is a provenance record and must not be run concurrently with the parallel workers. The public setup helper records new source, runtime, and deployment identities for a fresh run; it does not fabricate historical completion records.

## Bundled results and independent runs

The notebooks record configuration and source identities, deployment information, run identifiers, and progress alongside their output tables. Preserve those records with the tables when working in Drive or copying your results locally. Do not mix checkpoints after changing scientific configuration or source data. Different GPU hardware and package versions may affect training outputs, so source hashes alone do not establish byte-for-byte reproducibility across environments.

The local validator checks the bundled `current/results/` directory by default and accepts a separately supplied result directory using `--results-root`. It checks the expected design, unique run IDs, noise grids, five outer seeds, summary aggregation, applicable attack tables, solver records, and progress consistency. It is configured for the manuscript's experiment schedule and stopping criteria; a newly configured study needs corresponding validator changes. Converged and capped totals are calculated from the supplied flags rather than fixed to a reported outcome. Nothing is downloaded from private Drive by these local scripts.

## Aggregation and figures

Each positive-noise condition has four realizations within each deployment seed. Average those four values first, then compute the mean and sample SD across the five seed means. Each noiseless control occurs once per seed and participant count. Nested CelebA participant-count evaluations reuse the shared deployment; the primary CelebA analysis filters the merged table at p = 10.

Primary NAA-GDP linkage uses OP, while the private-noise comparison uses the disclosed common transform. Separate attack tables retain the applicable reconstructions. I-GDP and Local-only linkage stays missing because the workflow does not measure privacy for those utility references.

Preserve GPM stopping flags and capped observations when calculating utility and linkage summaries. The absolute step-size stopping criterion and the separately recorded relative fixed-point residual are different diagnostics. Figure code reads the diagnostics from the supplied result files.

Qualitative gallery generation requires the source images and a completed saved deployment. Its annotations use the selected seed and realization, rather than a five-seed average. The source and reconstruction tiles are not part of the current public code package.
