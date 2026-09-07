> **Historical package.** This file documents the earlier MNIST/CelebA experiments. For the current LFWA/CelebA manuscript, see [`current/`](../current/).

# Provenance and source-snapshot status

## Authoritative artifacts

The compressed result tables are the authoritative records of the completed quantitative experiments. The lossless qualitative-input archives are the authoritative inputs for the reconstruction grids. Before rendering, the repository validator checks the distributed files against `ARTIFACT_MANIFEST.json` and verifies result counts, schedule roles, participant counts, and reconstruction-archive structure.

The notebooks contain the protocol, attacks, data preparation, model training, evaluation, and checkpoint logic needed for an independent rerun. The available provenance records are distributed with the results, but their coverage differs by stage:

- the MNIST main and supplemental stages retain their principal configuration, data, linkage, noise, run, completion, and runtime records;
- the primary CelebA \(p=10\) core and supplemental stages retain their locally archived configuration, allocation, linkage, run, completion, control, and runtime records;
- the CelebA calibration manifest referenced by the completed stages was not present in the local archive;
- the revised participant-count replay retains the complete 103-record aggregate table for every \(p\in\{2,5,10,20,50\}\). Each row carries its run, data-design, anchor, deployment, training, attack, GPM, and runtime fields; the three exact standalone notebooks retain the checkpoint and resume implementation.

These omissions do not remove any observation used by the paper figures, but they preclude claiming a complete byte-for-byte provenance chain for every historical stage. For the revised participant-count study, the validator additionally checks that the five files share the paired noise vector, anchor hash, and deployment-bank hash.

## MNIST supplemental source snapshot

The embedded `core_protocols.py`, `mnist_runner.py`, and `mnist_orchestrator.py` sources in Notebook 02 match the corresponding hashes recorded by the completed supplemental stage. The executed stage records supplemental-runner SHA-256

```text
7d5c217c1f9fbe00b33af35d3ec55cd0a5e7fc451efb464126b7edf8736b8a8f
```

whereas the available notebook embeds

```text
a31297d5bb3f077b5b26cdd5c89d17a52cc330bf160c6e2c7bd3871af157aa37
```

The available runner implements the same supplemental schedule, but it is not the byte-identical historical source. The frozen 609-record table remains sufficient to reproduce every reported MNIST analysis and figure.

## CelebA qualitative replay source snapshot

The mixed-identity archive records historical replay-module SHA-256

```text
72a4a52df9dbe9002e1d13227bb4f46f45089c9d6f126b36a5777fb4b01ce371
```

whereas Notebook 06 embeds

```text
59c7cbb1d7367fe4b67c1bbf675b19aca92947d4d4d53600836ddc059bcd9e19
```

The identity-specific archive records the latter hash. Notebook 06 is sufficient to rerun the qualitative design, while the frozen lossless archives reproduce the exact mixed and identity-specific paper grids without relying on byte-identical historical replay code.

## Excluded artifacts

The repository intentionally excludes complete source datasets, model-download caches, optimizer checkpoints, redundant CSV/JSON copies, rendered PNG/SVG previews, obsolete pilot designs, the superseded \(p=10,30,50\) participant-count analysis, the failed pre-correction \(p=5,10\) replay shard, removed digit-leakage plots, \(p=70,90\) experimental results, Overleaf snapshots, and local dependency directories. The compact qualitative archives retain only the selected source and recovered tiles needed for the published grids.

