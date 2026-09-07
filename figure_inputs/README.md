# Qualitative figure inputs

These archives contain only the lossless image tiles and metadata needed to compose the qualitative reconstruction figures.

- `mnist_reconstruction_cells.zip`: the mixed-digit grid and ten fixed-digit supplementary grids.
- `celeba_mixed_reconstructions.zip`: the mixed-identity CelebA grid.
- `celeba_identity_reconstructions.zip`: ten fixed-identity CelebA supplementary grids.

They are compact, dataset-derived figure inputs rather than complete source datasets. They include selected original/source tiles alongside recovered tiles, because both are required to reproduce the exact qualitative grids. Their use and redistribution remain subject to the applicable dataset terms. `scripts/reproduce_figures.py` validates archive paths and extracts them under the ignored `build/inputs/` directory.
