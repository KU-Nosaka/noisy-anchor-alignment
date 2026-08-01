# CelebA participant-count replay

These five gzip-compressed CSV files are the complete records for the revised Section 5.3.3 experiment. The design fixes one deterministic 50-participant master allocation, GDP-parameter bank, centered unit-isometric anchor, public-SVD reducer, model procedure, and frozen face auditor.

For each \(p\in\{2,5,10,20,50\}\), the corresponding file contains 103 records:

- 100 AA-I-GDP anchor-noise observations, with draw indices 0 through 99 and \(v\sim\mathrm{Uniform}(0,0.1)\);
- one noiseless C-GDP control;
- one noiseless I-GDP control; and
- one zero-anchor-noise AA-I-GDP control.

The 100 scalar noise draws are paired across participant counts. The validator also checks the shared anchor and deployment-bank hashes, record schedules, unique run IDs, and GPM stopping counts. The CSV serialization is retained because each row contains the complete attack, utility, optimization, runtime, and provenance fields produced by its Colab shard.
