# Paper figure map

Running `python scripts/reproduce_figures.py` regenerates the following manuscript-ready PDFs.

## Main manuscript

| Figure | Output file | Renderer |
|---:|---|---|
| 1 | `mnist_exact_source_linkage_four_panel.pdf` | `figures/render_mnist_quantitative.py` |
| 2 | `mnist_reconstructions_mixed_exact_source.pdf` | `figures/render_mnist_reconstructions.py` |
| 3 | `mnist_utility_balanced_accuracy_two_panel.pdf` | `figures/render_mnist_utility.py` |
| 4 | `mnist_balanced_accuracy_versus_exact_source_linkage_op.pdf` | `figures/render_mnist_quantitative.py` |
| 5 | `celeba_cross_identity_leakage_private_and_anchor_methods_two_panel.pdf` | `figures/render_celeba_p010_quantitative.py` |
| 6 | `celeba_smiling_balanced_accuracy_two_panel.pdf` | `figures/render_celeba_p010_quantitative.py` |
| 7 | `celeba_smiling_balanced_accuracy_vs_cross_identity_linkage_op.pdf` | `figures/render_celeba_p010_quantitative.py` |
| 8 | `celeba_gpm_convergence_diagnostics.pdf` | `figures/render_celeba_p010_quantitative.py` |
| 9 | `celeba_p010_reconstruction_leakage_ordered.pdf` | `figures/render_celeba_reconstructions.py` |
| 10 | `celeba_privacy_utility_by_participant_count.pdf` | `figures/render_celeba_participant_count.py` |

## Supplement

- `mnist_reconstructions_digit_0_exact_source.pdf` through `mnist_reconstructions_digit_9_exact_source.pdf`
- `celeba_reconstructions_identity_01.pdf` through `celeba_reconstructions_identity_10.pdf`
- `celeba_participant_gpm_stopping.pdf`

The quantitative curves are fixed cubic regression-spline fits with five equally spaced interior knots. Pointwise 95% bands use 10,000 percentile-bootstrap resamples. Endpoint controls are preserved in the frozen result records but excluded from the fits and displayed random observations.
