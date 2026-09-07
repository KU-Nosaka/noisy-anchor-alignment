> **Historical package.** This file documents the earlier MNIST/CelebA experiments. For the current LFWA/CelebA manuscript, see [`current/`](../current/).

# Frozen experimental results

The files in this directory are the complete result records used by the current paper figures. JSON records are gzip-compressed without changing their serialized contents.

## MNIST

- `mnist/mnist_combined_609_results.json.gz`: 609 records, consisting of the 309-record main stage and 300 supplemental records.
- `mnist/manifests/`: fixed configuration, data split, linkage gallery, noise schedule, run schedule, controls, completion state, and runtime provenance.

## CelebA

- `celeba/p010_combined_407_results.json.gz`: 207 core records plus 200 supplemental records.
- `celeba/participant_replay/p002_results.csv.gz` through `p050_results.csv.gz`: the revised Section 5.3.3 replay for \(p\in\{2,5,10,20,50\}\). Each file contains 100 paired anchor-noise observations with \(v\sim\mathrm{Uniform}(0,0.1)\) and three deterministic controls.
- `celeba/manifests/`: the locally retained configuration, allocation, linkage, completion, and environment records for the primary \(p=10\) experiment.

The result tables include deterministic controls that are retained for provenance but excluded from the random-draw spline fits, bootstrap resampling, and quantitative scatter plots.

