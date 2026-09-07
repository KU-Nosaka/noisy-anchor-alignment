# Completed experiment results

This directory contains the compact numerical evidence from the completed Drive experiments used by the current Colab notebooks. It was assembled on 2026-09-07 (UTC). The original scalar CSV values are preserved. `SOURCE_MANIFEST.json` records original content hashes, source modification times, packaged hashes, and the few documented portability edits to JSON metadata.

| Experiment | Canonical directory | Result rows | Fitted models | Completion evidence |
|---|---|---:|---:|---|
| LFWA, full ten-level sweep | `lfwa_full_sweep/` | 420 | 440 | `progress.json`, source completed 2026-09-07 11:01 UTC |
| CelebA, merged parallel runs | `celeba_parallel/merged/` | 1,300 | 1,525 | `progress.json`, source completed 2026-09-07 07:24 UTC |
| C-GDP / AA-GDP implementation equivalence | `equivalence/` | 800 paired comparisons | Not a model-fitting experiment | `equivalence_manifest.json`, completed 2026-09-06 05:39 UTC |

LFWA supersedes the earlier three-level `results/lfwa` run. CelebA uses the merged parallel experiment, which retains completed legacy fits under one scientific signature. Its primary experiment is the `participant_count == 10` subset; LFWA uses `participant_count == 5`. The included `main_p10_results.csv` and `main_p5_results.csv` are exact row filters of their corresponding `raw_results.csv` files. Earlier repository result snapshots belong to earlier manuscript states and must not be mixed with these tables.

## Replication and uncertainty

Both face experiments use deployment seeds 0–4. Each positive noise condition has four realizations per deployment; zero-noise controls have one run per deployment. The ten anchor noise standard deviations are 0.006–0.060 in increments of 0.006, and the ten private-data noise standard deviations are 0.3–3.0 in increments of 0.3. Units are absolute entrywise standard deviation in the reduced representation. CelebA evaluates NAA-GDP at participant counts 2, 5, 10, 20, and 50, with the private-data-noise comparison at 10 participants.

`seed_means.csv` first averages the four realizations within each deployment. `summary.csv` then reports the mean and sample standard deviation (`ddof=1`) across the five deployment means. These standard deviations are not confidence intervals, and the twenty noise realizations are not twenty independent deployments. `balanced_accuracy` is the utility metric used in manuscript plots; the separate `accuracy` field is ordinary accuracy. Rates are stored as fractions and converted to percentages only for display. I-GDP and Local-only linkage is unmeasured and remains missing.

LFWA has 400 positive-noise fits, 15 zero-noise collaboration fits, and 25 participant-local fits. Its 420 result rows comprise the 415 collaboration fits plus five Local-only aggregate rows. CelebA has 1,200 positive-noise fits, 75 zero-noise collaboration fits, and 250 unique participant-local fits. Its 1,300 rows comprise the 1,275 collaboration fits plus 25 Local-only aggregate rows. For CelebA, local aggregates at smaller participant counts pool the first p models from each seed's shared bank of 50 local models; summing `aggregate_model_count` over all p values would count reused models multiple times.

## Attack and solver evidence

The primary noisy-anchor linkage is OP. Private-data-noise linkage uses the known common transform. LFWA `attack_results.csv` contains 830 measured attack rows, including the three attacks on NAA-GDP and noiseless references, plus the private-noise baseline. Its `attack_summary.csv` uses five-seed sample SD, stored under the historical column name `linkage_rate_std`.

CelebA `attack_results.csv` contains 1,400 post-hoc attack records requiring no additional utility fitting: 1,000 participant-count-specific AM records, 200 shared MP records, and 200 shared OP records. The blank participant count on shared MP/OP records is intentional: those attacks depend on the target anchor release, shared across p. `privacy_utility_by_attack.csv` joins those measured linkage values to each matching participant-count-specific utility result. It contains 3,250 rows and is the source for `attack_seed_means.csv` and `attack_summary.csv`. This reuse is not additional independent attack replication.

LFWA `alignment_diagnostics.csv` records all 200 NAA-GDP solver runs. All 100 runs at anchor noise at most 0.030 converged; all 100 at 0.036 or above reached the 500-iteration cap. Those runs remain in the reported results. CelebA `celeba_gpm_diagnostics.csv` contains the 200 detailed NAA-GDP records at p=10, copied from the saved manuscript figure inputs. It includes the relative fixed-point residual in addition to the absolute last-step norm; these are different measures.

`VALIDATION.json` records checks against the saved scalar evidence: exact row and fit counts, complete seed/realization groups, recomputed seed means and sample SDs, attack summaries, OP linkage agreement, and agreement of all 200 detailed CelebA GPM records with the main results. The equivalence evidence contains 800 paired matrix comparisons within 160 deployments. It is a float64 implementation check against a prespecified numerical tolerance, not a statistical equivalence test.

## Portability and external inputs

Source reports are retained verbatim; any generic provisional-run instructions in them are superseded by the complete progress records and validated row counts above. Execution hashes/signatures identify the historical code that generated these results. A cleaned notebook's file hash can differ because outputs, paths, or documentation were edited after execution.

The compact evidence excludes face images, image/identity membership lists, saved projections and releases, model weights, and heavy caches. Obtain input datasets and any required pretrained weights separately using the notebook setup instructions and their applicable access terms. Full experimental reruns reconstruct the needed caches and manifests in the chosen output directory. The LFWA protocol omits three absolute runtime paths; the equivalence manifest omits a BLAS installation path. `celeba_parallel/shared/canonical_public.json` retains scientific configuration, software versions, and deployment/cache integrity hashes while omitting image/sample identifiers and absolute runtime paths. It is a public provenance excerpt, not a resumable cache manifest. `SOURCE_MANIFEST.json` identifies its original canonical hash and every metadata transformation.
