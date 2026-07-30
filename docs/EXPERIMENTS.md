# Experiment reproduction

## Common principles

All random noisy fits use a fixed deployment—data allocation, dimensional-reduction map, anchor, model architecture, and auditor—and vary the prescribed noise realization and its scale. Deterministic controls are retained separately. Training and attack seeds are recorded in every result record.

The experiment notebooks write one atomic checkpoint per fit to Google Drive. A stage first inventories its expected run IDs, skips valid completed records, and writes a completion summary only after all planned records pass validation. This permits a disconnected Colab job to resume on another runtime.

## MNIST

The standard 60,000-image training split is divided into 50,000 training and 10,000 validation images; the official 10,000-image test split is used only for final evaluation. Images are normalized, flattened to 784 dimensions, and projected by one fixed data-independent Gaussian projection to \(h=100\). A fixed stratified-IID allocation divides each split among \(p=10\) participants.

The main stage contains:

- three noiseless controls: C-GDP, AA-I-GDP, and I-GDP;
- 100 C-GDP private-noise draws;
- 100 AA-I-GDP private-noise draws coupled to the corresponding C-GDP draws;
- 100 AA-I-GDP anchor-noise draws;
- six deterministic zero/upper-endpoint controls.

This gives 309 records. The supplemental stage adds 100 paired private-noise observations over \(35\leq\sigma\leq50\) for each private-noise protocol and 100 anchor-noise observations over \(0\leq v\leq0.2\), yielding the final 609-record table. Across the combined paper analysis, each noisy protocol therefore contributes 200 random observations; the private-noise display covers \(0\leq\sigma\leq50\), and the anchor-noise display covers \(0\leq v\leq0.5\).

Utility is held-out digit-classification balanced accuracy. Privacy is ten-way exact-source linkage in the frozen MNIST auditor embedding space.

## CelebA

The recorded deployment reserves 10,000 public CelebA images for fitting a common truncated-SVD reducer to \(h=400\). Participant data are partitioned by identity, with 100 identities per participant and identity-disjoint training, validation, and test allocations. One image per eligible identity is reserved as a linkage query, and a different image of the same identity is used in the ten-way gallery.

The core notebook uses one frozen 90-participant master allocation and executes the \(p=10,30,50\) prefixes used in the paper. Each participant-count stage contains:

- two clean controls: C-GDP and I-GDP;
- 100 C-GDP private-noise draws;
- 100 AA-I-GDP anchor-noise draws;
- four deterministic zero/upper-endpoint controls;
- one high-\(v\) I-GDP-equivalence diagnostic.

This gives 207 records per participant count. The \(p=10\) supplemental stage adds 100 private-noise observations over \(1.5<\sigma\leq3.0\) and 100 focused anchor-noise observations over \(0<v<0.05\). The \(p=30\) and \(p=50\) supplemental notebook adds 100 focused \(0<v<0.05\) anchor-noise observations to each condition. The participant-count comparison therefore uses 200 anchor-noise observations for each of \(p=10,30,50\).

Utility is held-out balanced accuracy for the Smiling attribute. Privacy is ten-way cross-image identity linkage using normalized 512-dimensional embeddings from a frozen InceptionResnetV1 auditor pretrained on VGGFace2.

## Recommended runtime

The reported long-running stages used Google Colab Pro+ high-memory A100 runtimes. Figure rendering is CPU-only. The exact package, CUDA, GPU, deterministic-execution, and stage timing metadata are retained in the result manifests.

## Data persistence

Use a Google Drive folder for both the dataset/model cache and stage outputs. Do not write the only copy of results to `/content`: Colab's local disk is erased when a runtime disconnects.
