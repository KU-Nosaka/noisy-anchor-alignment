# Current CelebA and VGGFace2 experiments

The four notebooks in `notebooks/current/` reproduce the September 2026 experiment recipes used by the current manuscript. They start from downloaded dataset files inside this repository. They do not require a previous study, Google Drive, prepared representations, historical tensor hashes, or a training checkpoint.

Each fresh run reads the raw images and annotations, constructs the identity allocation, fits the public SVD feature mapping, and builds the anchor, participant transformations, and linkage galleries. Generated artifacts are fingerprinted and cached locally for resume. A cache is an output of the notebook, not an input prerequisite.

## Choose a notebook

| Notebook | Dataset | Experiment | Default target |
|---|---|---|---|
| [10 — CelebA privacy–utility](../notebooks/current/10_celeba_casia_privacy_utility.ipynb) | CelebA | Both noise placements at all five participant counts | 1,097 CNN fits if every quota fills |
| [11 — VGGFace2 privacy–utility](../notebooks/current/11_vggface2_casia_privacy_utility.ipynb) | VGGFace2 | Both noise placements at all five participant counts | 1,097 CNN fits if every quota fills |
| [12 — CelebA attack comparison](../notebooks/current/12_celeba_casia_attack_comparison.ipynb) | CelebA | MP, AM, and OP at ten participants | 500 paired noise draws; no utility training |
| [13 — VGGFace2 attack comparison](../notebooks/current/13_vggface2_casia_attack_comparison.ipynb) | VGGFace2 | MP, AM, and OP at ten participants | 500 paired noise draws; no utility training |

Use a separate runtime and output directory for each dataset. On Colab, select a Tesla T4 GPU before running the cells. The first cells locate or clone the repository and install the Python dependencies. Put the downloaded files at the paths below before the raw-input validation cell. On a local machine, clone the repository first and select an equivalent CUDA runtime.

Colab's `/content` disk is temporary. Copy outputs and generated caches to storage you control before ending the runtime if you want to resume later. The notebooks do not automatically mount Drive. Read the disk and RAM preflight before beginning: VGGFace2 preparation and the large participant-count fits are substantial even though the CNN is small.

## Downloaded inputs

Data are excluded from Git. Obtain the datasets and annotations under their original distribution terms; the notebooks do not download facial datasets or require a dataset-service credential.

### CelebA

```text
data/raw/celeba/
  img_align_celeba/
    000001.jpg
    ...
  identity_CelebA.txt
  list_attr_celeba.txt
```

The Kaggle attribute file `list_attr_celeba.csv` may replace the official attribute TXT file. Both carry the released `Smiling` labels. The identity metadata are required to exclude shared identities between the public feature-fitting images and participant cohorts, reserve references, and split by identity. The aligned JPEG directory should contain the complete CelebA image set. Dataset partition, landmark, and bounding-box files are not used by this recipe.

### VGGFace2

```text
data/raw/vggface2/
  data/vggface2_train.tar.gz
  meta/bb_landmark.tar.gz
  meta/identity_meta.csv
  meta/train_list.txt
  maad/MAAD_Face.csv
```

The five downloaded files occupy approximately 38.6 GB. The image archive is streamed; a full extracted copy is unnecessary. Use the official VGGFace2 training archive and its matching metadata. The Smiling labels come from **MAAD-Face**, so the image archive alone is insufficient for this experiment. Released unknown labels are excluded, and the defined labels are mapped to the binary task. The input validator checks the actual files, rather than requiring private download receipts.

## Shared feature and evaluation recipe

- Construct a fixed 50-participant allocation with 100 identities per participant. Assign 70, 15, and 15 identities to training, validation, and test within each participant; reserve one distinct reference image per identity first. Keep all remaining eligible images with their identity's participant and split. Nested collaborations use the first `p` participants for `p = 2, 5, 10, 20, 50`.
- Fit an **uncentered** public SVD mapping from 10,000 identity-disjoint public RGB images. Images are cropped and resized to 64×64, scaled to [0,1], and flattened to 12,288 entries. `torch.pca_lowrank` uses `q=420`, `center=False`, and `niter=4`; the leading 400 vectors are orthogonalized by CPU float64 QR. The frozen final mapping has 400 columns.
- CelebA uses center-square cropping. VGGFace2 uses its official face boxes, expanded by 1.3 and clipped, followed by center-square cropping and resizing. VGGFace2 cohort selection also excludes invalid and exact duplicate images; identities require at least 51 valid, distinct images with defined Smiling labels and both classes before reference reservation. CelebA identities require at least two images and exclude public identities and the deterministic legacy test quarantine.
- Generate a 1,600×400 centered, sign-corrected QR anchor with orthonormal columns. Sample independent Haar orthogonal rotations and standard-normal translations for the 50-participant bank. The reference/colluder is participant 0 and the protected participant is participant 1.
- Reconstruct query rows in the public SVD subspace. Decode queries and references, clip to [0,1], bilinearly resize to 160×160, and standardize as `(255*x - 127.5)/128`. The frozen CASIA-WebFace InceptionResnetV1 outputs normalized 512-dimensional embeddings for cosine matching. The model is downloaded automatically from the official `facenet-pytorch` source and checked against SHA-256 `7a67afdbbc995fce5e10128675e318799a70698c2f433ba75dd7eb9a2f096e7d`.
- Each protected identity supplies one query and ten fixed ten-way lineups, giving 1,000 decisions and a 10% random-guessing rate. Distractors are drawn from the active collaboration and held fixed across methods and noise realizations at a given `p`. The model is not fine-tuned on either dataset; identity-disjoint auditor pretraining is not assumed.
- Alignment uses the CPU float64 spectral estimator and participant-block polar factors. These current notebooks do not run the historical GPM iteration workflow.

The dataset preparation seed is 20260713 for CelebA and 20260928 for VGGFace2. CelebA's legacy quarantine uses 20260712. Both datasets use the shared design seed 20260928. Named seeds are preserved in the source. Every output records the freshly generated input/design fingerprints and source/environment details. Randomized SVD and floating-point kernels can differ across Torch, CUDA, and LAPACK versions; these notebooks reproduce the recipe and create verifiable new artifacts, rather than requiring byte-identical historical tensors.

## Main privacy–utility studies

Each noise placement targets 20 accepted realizations in each measured linkage interval `[0,15)`, `[15,25)`, `[25,35)`, `[35,45)`, and `[45,100]` percent, for 100 accepted observations per family at each `p`. Candidate scales are adapted using linkage alone, within anchor-noise scale `v ∈ [0,0.08]` and private-data-noise scale `sigma ∈ [0,3]`. Every proposal uses fresh Gaussian noise. The first 20 accepted realizations in each bin are retained. Calibration is separate at each `p`, with at most 1,500 proposals per family; utility is trained after acceptance.

The clean C-GDP and I-GDP controls are also fitted. A zero-noise alignment diagnostic verifies the equivalence to C-GDP; it does not add a separate utility fit. One Local model is fitted for each active participant. The Local utility reference pools the active participants' local test predictions. The per-`p` totals are `200 + p + 2`: 204, 207, 212, 222, and 252 unique fits, adding to 1,097. A prefitted C-GDP model reused during the pipeline is counted once.

Utility is test Smiling balanced accuracy. The CNN reshapes each 400-dimensional row to 1×20×20, then uses valid 5×5 convolutions with 10 and 20 channels, ReLU and 2×2 max pooling, an 80-entry flattening, a 50-unit ReLU layer, and one output logit. Each model trains afresh for 15 epochs with unweighted binary cross-entropy, Adam at 0.001, batch size 256, and no weight decay. Selection uses highest validation balanced accuracy, keeping the earliest epoch in a tie. Test probabilities use sigmoid and threshold 0.5.

Completion requires all bin quotas and all unique scheduled fits. If calibration reaches its cap before a quota fills, the run retains its real observations and reports incomplete bins; it does not synthesize observations. Pointwise participant-count plots show mean utility and one sample standard deviation of the accepted realizations, not paired confidence intervals across participant counts.

## Supplementary attack comparison

At `p=10`, draw 500 unconditional uniform scales from `[0,0.08)` with seed 20260930 and the original named namespace. Generate a fresh Gaussian anchor-noise realization for each draw, then run MP, AM, and OP on the **same uploads and galleries**. No draw is selected by linkage, rejected for a quota, or used for utility training. A separate zero-noise diagnostic is excluded from the 500 observations and fitted curves.

Resume verifies the frozen schedule and committed per-draw records before skipping them. The default comparison summarizes all 500 draws. Descriptive spline lines and bootstrap bands describe noise-draw variation conditional on the fixed allocation and auditor; they do not provide a privacy guarantee.

## Review and validation

The implementation is readable under `reproduction/casia/`; notebooks call those checked-in modules instead of embedding compressed source. The notebook builder is `scripts/build_current_casia_notebooks.py`.

```bash
python scripts/build_current_casia_notebooks.py --check
python -m unittest discover -s reproduction/casia -p 'test_*.py' -v
python scripts/validate_repository.py
```

The small tests and notebook checks do not rerun the full GPU studies. A new full dataset run is needed to produce an independent set of numerical outcomes. Full-study export produces numeric summaries, the p=10 grouped privacy–utility bars, the anchor-noise participant-count figure, and a diagnostic all-p plot. The manuscript-style figures use sample SD, include the pooled Local control, omit a chance-utility reference line and plot titles, and use the requested crop below 50% when every displayed outcome fits. A fresh below-chance outcome expands the axis rather than being hidden. Empty bins remain absent and their actual counts appear in the CSV summaries. The historical notebooks 01–09 and frozen MNIST/CelebA tables remain available for their earlier recipes; they are not the current CASIA configuration.
