# Experiment notebooks

The notebooks are self-contained Colab workflows: the reviewed protocol and runner modules are written from notebook cells at runtime. Assign one new Google Drive result root for an independent reproduction and reuse it in every stage.

| Notebook | Purpose | Expected completed records |
|---|---|---:|
| `01_mnist_main.ipynb` | MNIST clean controls, 100 private-noise draws for each matched protocol, 100 anchor-noise draws, and endpoint controls | 309 |
| `02_mnist_supplemental_noise.ipynb` | Adds 100 paired private-noise draws and 100 focused anchor-noise draws | 300 additional; 609 combined |
| `03_celeba_p010_main.ipynb` | Primary CelebA \(p=10\) controls and noisy protocols | 207 |
| `04_celeba_p010_supplemental_noise.ipynb` | Adds the focused \(p=10\) private- and anchor-noise strata | 200 |
| `05_celeba_p010_reconstruction_replay.ipynb` | Replays the selected mixed and identity-specific qualitative reconstructions | reconstruction artifacts |
| `06_celeba_participant_replay_p050.ipynb` | Revised participant-count replay shard for \(p=50\) | 103 |
| `07_celeba_participant_replay_p002_p020.ipynb` | Revised participant-count replay shard for \(p=2,20\) | 103 per participant count |
| `08_celeba_participant_replay_p005_p010.ipynb` | Corrected revised participant-count replay shard for \(p=5,10\) | 103 per participant count |

## Exact execution controls

1. In `01_mnist_main.ipynb`, leave `RUN_MODE="pilot"` and `CONFIRM_FULL_RUN=False` for the validation-only calibration. After inspecting the frozen pilot manifest, rerun the notebook with `RUN_MODE="full"`, `CONFIRM_FULL_RUN=True`, and the same Drive project path, results subfolder, and pilot-manifest path. The full stage must complete before Notebook 02 is started.
2. In `02_mnist_supplemental_noise.ipynb`, keep `EXECUTE_SUPPLEMENTAL_SWEEP=True` and point it to the completed Notebook 01 results subfolder.
3. In `03_celeba_p010_main.ipynb`, keep `EXECUTE_FULL_SWEEP=True`. The notebook preserves the original 90-participant master allocation for exact reproduction but executes only its \(p=10\) prefix.
4. In `04_celeba_p010_supplemental_noise.ipynb`, keep `EXECUTE_SUPPLEMENTAL_SWEEP=True` and reuse the CelebA results/cache roots created by Notebook 03.
5. In `05_celeba_p010_reconstruction_replay.ipynb`, keep both `EXECUTE_RECONSTRUCTION_SWEEP=True` and `EXECUTE_IDENTITY_SWEEPS=True`. The mixed replay runs first and creates metadata needed by the ten identity-specific replays.
6. Run Notebooks 06--08 on independent A100 runtimes. Each notebook writes to a distinct Google Drive result root, constructs the same deterministic 50-participant deployment bank and anchor, and resumes from atomic per-fit records. Together they execute 100 paired draws with \(v\sim\mathrm{Uniform}(0,0.1)\) for each \(p\in\{2,5,10,20,50\}\), plus three controls. Do not substitute the failed, non-`r2` \(p=5,10\) notebook that preceded Notebook 08.

Every stage is resumable. Rerunning a notebook against the same Drive root inventories the planned run IDs and skips valid completed fits.

For code review, `scripts/celeba_participant_replay_runner.py` is the exact replay overlay embedded in all three participant-count notebooks. The notebooks also embed its reviewed CelebA base and unit-isometric-anchor modules, so the notebooks—not the extracted overlay alone—are the executable Colab artifacts.

The experiment environment is intentionally managed inside the notebooks because CUDA-enabled Colab packages are runtime-specific. See `docs/EXPERIMENTS.md` and the runtime locks under `results/`.
