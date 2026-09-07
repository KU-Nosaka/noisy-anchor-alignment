# LFWA run status

Completed fitted models: 440/440.
Completed main result rows: 420/420.
Completed individual local fits: 25/25.
Completed positive-noise NAA fits by participant count: {5: 200}.

All rates in CSV files are fractions. SD is across outer-seed means, after averaging
four realizations within each seed and noise level. A summary row is final only when
all five seeds have all required realizations. Partial figures are labelled provisional.

The main p=5 CSV is an exact filter of raw_results.csv. No rerun or new deployment
is used for the participant-count figure. The same target participant and fixed
100-reference gallery pool are retained across participant counts.

Local-only fits one MLP per participant on its own unperturbed public-SVD records.
Each local model is fitted once per seed and reused in nested collaborations.
Local-only result rows pool confusion counts across the first p local models, so
their balanced accuracy uses the same pooled test cohort as collaboration methods.
They are aggregate rows, not additional fits. Their fit_seconds is the sum of the
constituent local fitting times. Privacy is unmeasured, not zero.

This evaluates MP/AM/OP reconstructions followed by a frozen public face auditor; it does
not establish privacy against all admissible attacks. I-GDP and Local-only have no linkage estimate.
