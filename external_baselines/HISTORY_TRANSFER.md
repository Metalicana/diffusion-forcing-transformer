# DFoT Native Observed-History Transfer Pilot

This adds an inference-only KEEPSAKE history selector to the existing frozen
DFoT_RE10K checkpoint. It does not train, fine-tune, run the full RealEstate10K
download, launch MemCam generation, or use Weights & Biases. CUDA execution has
not been verified on Newton until the smoke and pilot actually run there.

## Already Allocated H100

From your existing compute-node shell (not the login node), after transferring
these files into the clone:

```bash
cd "$HOME/diffusion-forcing-transformer"
bash external_baselines/run_history_transfer.sh
```

Setup creates the separate `~/.conda/envs/dfot` Python 3.10 environment with
constrained dependencies and CUDA 12.8 PyTorch wheels, then checks imports, a real
CUDA kernel, torchvision NMS and MP4 I/O. Do not substitute a physical GPU index:
Slurm's existing CUDA_VISIBLE_DEVICES is preserved. This needs a compatible
driver; an H100 model name alone does not establish driver compatibility.

The launcher runs setup, tests and the pilot, saving a timestamped log under
`~/memcam_results/dfot_history_transfer_mini_n5`. Its default total time budget is
12,600 seconds including setup. Set `DFOT_TOTAL_SECONDS` to the remaining time
minus a margin if your four-hour allocation has already been running. This is a
between-cell budget, not a hard timeout or a prediction of completion time.

For a manual resume after the environment is already installed:

```bash
"$HOME/.conda/envs/dfot/bin/python" -m unittest discover -s tests -p 'test_history_transfer.py' -v
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled
"$HOME/.conda/envs/dfot/bin/python" -u -m external_baselines.native_history_transfer \
  --output "$HOME/memcam_results/dfot_history_transfer_mini_n5" \
  --scenes 5 --max-seconds 7200
```

There is no nested srun. Do not submit another sbatch if using an allocation
already running. For a future allocation with dependencies installed, use
`sbatch slurm/newton_dfot_history_transfer.sbatch`. It requests one highgpu H100
for four hours. Environment setup and downloads are not a completion guarantee.
The runner time budget includes its preparation and checks between generation
cells; it cannot extend the Slurm allocation or interrupt a cell exactly on time.

## Protocol

- Native `realestate10k_mini`, upstream deterministic evaluation clip selection,
  five source clips by default. The dataset loader uses its existing shuffle
  seed 0, not a quality-based example search. Clip IDs and index maps are frozen.
- Twenty sampled observations, stride 10 in the released processed data: the
  first 16 are observed historical inputs; the last four are unseen target views.
  The packaged camera conditions and 256x256 images use upstream preprocessing,
  including its documented historical intrinsic/crop convention.
- Four context slots and four jointly generated target slots per invocation;
  native RE10K pixel-space checkpoint, 50 sampling steps, vanilla history guidance
  scale 2, float16 autocast, no interpolation, compilation or fine-tuning.
- FIFO chooses the last four historical observations. Uniform chooses indices
  0, 5, 10, 15. KEEPSAKE streams all 16 past observations through B4 using MemCam's
  actual `compute_slam_covisibility_scores` with default parameters. DINOv2-base
  float32 normalized features are extracted only from observed history.
- All contexts include the newest observation. KEEPSAKE temporarily protects
  only the incoming observation; no permanent initial anchor is imposed because
  all 16 input observations here are observed data, not generated initial state.
  The score formula is unchanged. Camera vectors are converted from w2c to c2w.
  Selected images and their camera vectors are reordered together chronologically.
- Identical target indices/poses and seed `2026 + clip_index` across policies.
  The sampler receives only selected historical RGB and target cameras, never
  target RGB. Targets are accessed by the evaluator after generation.
- Uniform is an offline history-subset baseline, not an online constant-memory
  algorithm. B4 is the generator conditioning budget. No total-RAM/VRAM bound or
  unchanged native memory retriever is claimed: DFoT's temporal context selection
  itself is the integration point. The proposed full persistent-archive transfer
  and generated-history feedback experiments remain separate.

## Evaluation and Resumption

PSNR/SSIM/LPIPS are computed on four targets, then averaged per clip and equally
across clips. LPIPS uses AlexNet and inputs scaled to [-1,1]; SSIM uses Gaussian
sigma 1.5 and population covariance. PSNR uses per-frame RGB MSE, with an explicit
1e-12 floor only for exact/near-exact matches. Unquantized predictions are scored;
PNG quantization is visualization only. No FVD, VBench aggregate or inferred
confidence interval is produced for this small pilot.

Files: `config.json`, `cohort.json`, `encoder.json`, `checkpoint.json`,
`software.json`, `metrics.json`, per-case `selection.json`, generated/context/GT
PNGs, `prediction.npy`, `frame_metrics.csv` and generation/quality receipts.
`progress.json` records completed cells. Final `scores.csv`, `per_video.csv` and
`paired_differences.csv` require all 15 cells. Incomplete work is explicitly named
`partial_per_video.csv`; no silently intersected or favorable-only cohort.

Rerun the same command to resume. Verified generation can be reused if metrics
fail. Inputs, source code, scorer, software, weights, features and artifacts are
hashed or frozen; mismatches require a new output directory. Concurrent writers
are excluded by a filesystem lock. A changed host/GPU is recorded but does not
imply numerical identity across hardware. Do not update source code mid-run.

This is a native-data observed-history transfer pilot, not a reproduction of the
paper's separate 32-case OOD benchmark and not evidence of long-rollout stability.
Report every prespecified clip and all three methods, including null/reversed
outcomes. Do not tune selection parameters against these pilot targets.
