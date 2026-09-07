# Parallel CelebA progress

Completed fitted models: 1525/1525.
Main result rows: 1300/1300.
Post-hoc attack rows: 1400/1400; these require no utility retraining.

All completed legacy fits are reused without editing their files. Local models and aggregates belong to p50; private-noise runs and the OP/MP/AM replay belong to p5_10. Main p=10 rows are an exact filter of the merged results. Seed SD is computed after averaging four noise realizations within each deployment. Additional attack results are kept separate from the original OP utility records. attack_seed_means.csv first averages four realizations; attack_summary.csv reports their mean and SD across the five deployment seeds. Incomplete cells remain provisional. privacy_utility_three_attacks.pdf compares MP, AM, and OP; the known-common-transform private-noise baseline and noiseless references are shared across panels. Rerun the owner replay/merge cell after peer jobs finish if this report is provisional.
