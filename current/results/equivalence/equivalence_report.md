# Numerical validation of C-GDP / AA-GDP equivalence

In float64 synthetic implementation checks spanning dimensions 10, 50, 100, and 400, participant counts 2, 10, and 50, and private-data-noise scales 0, 0.01, 0.1, 1, and 10, the 800 paired C-GDP and AA-GDP collaboration matrices agreed to a maximum relative Frobenius discrepancy of 3.797e-14. This includes 600 comparisons with centered orthonormal anchors (r=4d) and 200 additional comparisons with 10 participants and shifted full-rank anchors at r=d+1 whose centered condition number is 10. All scored residuals were below the prespecified 10^-10 numerical tolerance. These checks validate the implementation of Proposition III.1 under its prescribed noise coupling; they are not a statistical test of equivalence.

| Anchor configuration | Dimension | Comparisons | Maximum matrix residual | Maximum block residual | Maximum rotation residual |
|---|---:|---:|---:|---:|---:|
| orthonormal | 10 | 150 | 7.831e-16 | 9.999e-16 | 9.398e-16 |
| orthonormal | 50 | 150 | 1.367e-15 | 1.476e-15 | 1.418e-15 |
| orthonormal | 100 | 150 | 1.650e-15 | 1.749e-15 | 1.657e-15 |
| orthonormal | 400 | 150 | 2.462e-15 | 2.542e-15 | 2.430e-15 |
| minimal_full_rank | 10 | 50 | 3.264e-15 | 5.043e-15 | 2.919e-15 |
| minimal_full_rank | 50 | 50 | 8.513e-15 | 1.044e-14 | 7.507e-15 |
| minimal_full_rank | 100 | 50 | 1.387e-14 | 1.644e-14 | 1.306e-14 |
| minimal_full_rank | 400 | 50 | 3.797e-14 | 4.289e-14 | 3.720e-14 |

The 800 comparisons share five noise levels within each of 160 deployments. They must not be described as 800 independent deployments. No image data, labels, model, accuracy measure, linkage auditor, or privacy claim is used. The tolerance and grid were specified before execution. The uncoupled-noise control produces different matrices while preserving the same marginal Gaussian law.
