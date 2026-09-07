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
| 10 | `celeba_privacy_utility_by_participant_count.pdf` | `figures/render_celeba_participant_count.py` (200 paired draws for each \(p\in\{2,5,10,20,50\}\): the 100 published draws over \(0<v<0.1\) and the 100 supplemental draws at half their amplitude; the I-GDP reference is shown for every participant count and the C-GDP and noiseless anchor-aligned controls are omitted from the figure) |
| 10, layered | `celeba_privacy_utility_by_participant_count_layers.pdf` + `celeba_participant_count_layers.tex` | same renderer; see below |

## Supplement

- `mnist_reconstructions_digit_0_exact_source.pdf` through `mnist_reconstructions_digit_9_exact_source.pdf`
- `celeba_reconstructions_identity_01.pdf` through `celeba_reconstructions_identity_10.pdf`

The quantitative curves are fixed cubic regression-spline fits with five equally spaced interior knots. Pointwise 95% bands use 10,000 percentile-bootstrap resamples. Endpoint controls are preserved in the frozen result records but excluded from the fits and displayed random observations.

## Layered (interactive) participant-count figure

Besides the flat figure, `render_celeba_participant_count.py` writes `celeba_privacy_utility_by_participant_count_layers.pdf`, a 16-page PDF whose pages share the flat figure's page box: page 1 carries the axes, chance levels, and legends, and each further page carries one participant count's bootstrap band, observations (with its I-GDP reference), or fitted spline. `figures/latex/celeba_participant_count_layers.tex` stacks these pages as PDF optional-content groups with the `ocgx2` package and typesets one check box per participant count and one for the spline fits; a spline layer is nested in its participant count's group and in the spline-fits group, so it is visible only when both boxes are ticked. Every layer starts visible, so printing or a viewer without layer support shows the flat figure. To use it in a LaTeX manuscript, load `graphicx`, `amssymb`, and `ocgx2` (after `hyperref`), place the layered PDF and the `.tex` body next to the document, and `\input` the body inside a `figure` environment; `figures/latex/celeba_participant_count_interactive.tex` compiles the same body as a standalone PDF (`pdflatex`, twice). Layer switching works in viewers that implement PDF optional content and `SetOCGState` link actions (Adobe Acrobat and Reader, Foxit, PDF-XChange, Okular, and Firefox's viewer); other viewers show the default state.
