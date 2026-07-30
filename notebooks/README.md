# Experiment notebooks

The notebooks are self-contained Colab workflows: the reviewed protocol and runner modules are written from notebook cells at runtime. Assign one new Google Drive result root for an independent reproduction and reuse it in every stage.

| Notebook | Purpose | Expected completed records |
|---|---|---:|
| `01_mnist_main.ipynb` | MNIST clean controls, 100 private-noise draws for each matched protocol, 100 anchor-noise draws, and endpoint controls | 309 |
| `02_mnist_supplemental_noise.ipynb` | Adds 100 paired private-noise draws and 100 focused anchor-noise draws | 300 additional; 609 combined |
| `03_celeba_participant_sweep.ipynb` | Core CelebA stages for \(p=10,30,50\) | 207 per participant count |
| `04_celeba_p010_supplemental_noise.ipynb` | Adds the focused \(p=10\) private- and anchor-noise strata | 200 |
| `05_celeba_p030_p050_low_v.ipynb` | Adds focused low-\(v\) anchor-noise strata for \(p=30,50\) | 100 per participant count |
| `06_celeba_p010_reconstruction_replay.ipynb` | Replays the selected mixed and identity-specific qualitative reconstructions | reconstruction artifacts |

## Exact execution controls

1. In `01_mnist_main.ipynb`, leave `RUN_MODE="pilot"` and `CONFIRM_FULL_RUN=False` for the validation-only calibration. After inspecting the frozen pilot manifest, rerun the notebook with `RUN_MODE="full"`, `CONFIRM_FULL_RUN=True`, and the same Drive project path, results subfolder, and pilot-manifest path. The full stage must complete before Notebook 02 is started.
2. In `02_mnist_supplemental_noise.ipynb`, keep `EXECUTE_SUPPLEMENTAL_SWEEP=True` and point it to the completed Notebook 01 results subfolder.
3. In `03_celeba_participant_sweep.ipynb`, keep `EXECUTE_FULL_SWEEP=True` and `EXECUTE_EXPLORATORY_P70_P90=False`. Running all cells then executes calibration and the \(p=10\), \(p=30\), and \(p=50\) paper stages while assigning explicit skipped states to \(p=70,90\), so the final notebook summary remains valid. The 90-participant master allocation is still constructed exactly as in the recorded design.
4. In `04_celeba_p010_supplemental_noise.ipynb`, keep `EXECUTE_SUPPLEMENTAL_SWEEP=True` and reuse the CelebA results/cache roots created by Notebook 03.
5. In `05_celeba_p030_p050_low_v.ipynb`, keep `EXECUTE_SUPPLEMENTAL_SWEEP=True`, `RUN_P30=True`, and `RUN_P50=True`, using the same roots.
6. In `06_celeba_p010_reconstruction_replay.ipynb`, keep both `EXECUTE_RECONSTRUCTION_SWEEP=True` and `EXECUTE_IDENTITY_SWEEPS=True`. The mixed replay runs first and creates metadata needed by the ten identity-specific replays.

Every stage is resumable. Rerunning a notebook against the same Drive root inventories the planned run IDs and skips valid completed fits.

The experiment environment is intentionally managed inside the notebooks because CUDA-enabled Colab packages are runtime-specific. See `docs/EXPERIMENTS.md` and the runtime locks under `results/`.
