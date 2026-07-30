# Frozen experimental results

The files in this directory are the complete result records used by the current paper figures. JSON records are gzip-compressed without changing their serialized contents.

## MNIST

- `mnist/mnist_combined_609_results.json.gz`: 609 records, consisting of the 309-record main stage and 300 supplemental records.
- `mnist/manifests/`: fixed configuration, data split, linkage gallery, noise schedule, run schedule, controls, completion state, and runtime provenance.

## CelebA

- `celeba/p010_combined_407_results.json.gz`: 207 core records plus 200 supplemental records.
- `celeba/p030_core_207_results.json.gz` and `celeba/p050_core_207_results.json.gz`: the completed participant-count core stages.
- `celeba/p030_low_v_100_results.json.gz` and `celeba/p050_low_v_100_results.json.gz`: focused \(0<v<0.05\) anchor-noise stages.
- `celeba/manifests/`: the locally retained participant-count-specific configuration, allocation, linkage, completion, and environment records. The focused \(p=30,50\) low-\(v\) stages retain completion summaries but not separate run manifests or runtime locks; see `docs/PROVENANCE.md`.

The result tables include deterministic controls that are retained for provenance but excluded from the random-draw spline fits, bootstrap resampling, and quantitative scatter plots.
