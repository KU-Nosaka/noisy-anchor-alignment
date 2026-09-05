# CelebA participant-count replay

These five gzip-compressed CSV files are the complete records for the revised Section 5.3.3 experiment. The design fixes one deterministic 50-participant master allocation, GDP-parameter bank, centered unit-isometric anchor, public-SVD reducer, model procedure, and frozen face auditor.

For each \(p\in\{2,5,10,20,50\}\), the corresponding file contains 103 records:

- 100 AA-I-GDP anchor-noise observations, with draw indices 0 through 99 and \(v\sim\mathrm{Uniform}(0,0.1)\);
- one noiseless C-GDP control;
- one noiseless I-GDP control; and
- one zero-anchor-noise AA-I-GDP control.

The 100 scalar noise draws are paired across participant counts. The validator also checks the shared anchor and deployment-bank hashes, record schedules, unique run IDs, and GPM stopping counts. The CSV serialization is retained because each row contains the complete attack, utility, optimization, runtime, and provenance fields produced by its Colab shard.

## Supplemental low-noise stratum (`supplemental/`)

The five files `supplemental/p0xx_supplemental_compact.csv` add, for each \(p\in\{2,5,10,20,50\}\), 100 further anchor-noise observations of the same method in the low-noise regime \(0<v<0.05\) (`run_role = supplemental_anchor_v_0_0p05_v1`, produced by `notebooks/09_celeba_participant_replay_supplemental_v0_0p05.ipynb`). Supplemental draw \(j\) reuses the uniform quantile, the participant-indexed noise tensors, the model seed, the minibatch order, and the GPM restart seed of published draw \(j\) at exactly half the amplitude, \(v_j^{\mathrm{supp}} = v_j/2\); the notebook rebuilds the reviewed modules byte for byte from the pinned repository commit and verifies the regenerated anchor matrix, GDP-parameter bank, and source draws against the hashes recorded in the published tables before any fit runs. The supplemental draws are therefore paired across participant counts in the same way as the published draws, and the participant-count figure uses all 200 draws per condition.

The compact tables carry the per-draw fields consumed by the figure renderer and the validator: `p`, `draw`, `anchor_v`, `test_balanced_accuracy`, `op_cross_image_identity_top1`, `gpm_converged`, `gpm_iterations`, `gpm_elapsed_seconds`, and `runtime_seconds` (a rerun of Notebook 09 also writes `run_role` and `run_id`). The complete per-fit records, in the schema of the published tables, are retained by the authors and will be added as `supplemental/p0xx_supplemental_results.csv.gz`; the renderer prefers them whenever they are present. The validator checks the row counts, the draw indices, the exact half-amplitude pairing with the published draws, and the GPM stopping counts (48, 58, 71, 81, and 90 converged fits of 100 for \(p=2,5,10,20,50\)).
