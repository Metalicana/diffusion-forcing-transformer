# External baseline study (local code, cluster execution)

No SSH, model downloads, environment modification, training, MemCam generation,
budget sweeps, or 60-second suite. The default is the official pose-conditioned
**DFoT_RE10K** model evaluated on the exact 15-row, nominal 180-second manifest.
This is a cross-domain pretrained baseline, not the internally trained CaM baseline.

## Create the DFoT environment first

After pulling these files on the cluster, run on an allocated GPU:

```bash
bash external_baselines/setup_dfot.sh 0
```

This creates `$HOME/.conda/envs/dfot` with Python 3.10 and ffmpeg, installs
upstream requirements under compatibility constraints, runs `pip check`, then
checks actual DFoT imports, CUDA matrix/NMS kernels, MP4 encode/decode and the
contract tests. It needs no `study.json`, dataset, checkpoint or calibration.
Existing MemCam/VBench environments are not modified. Override `DFOT_ENV_PREFIX`
with an absolute path for another dedicated DFoT environment. An existing prefix
is reused and its packages are updated by this command.

The bootstrap uses the official PyTorch 2.5.1/torchvision 0.20.1 CUDA 12.4 wheels:
https://pytorch.org/get-started/previous-versions/#v251
This is a conservative starting environment, not a locally GPU-tested lockfile;
the cluster smoke establishes compatibility with the installed GPU and driver.
A newer GPU architecture may require a newer matching PyTorch/torchvision build.
The resolved packages and smoke report are saved in `outputs/dfot_environment/`.

To repeat only the environment smoke:

```bash
export PATH="$HOME/.conda/envs/dfot/bin:$PATH"
"$HOME/.conda/envs/dfot/bin/python" -u -m external_baselines.environment_smoke --gpu 0
```

Only after that passes, configure the manifest-driven generation smoke below.

If an earlier setup fails with `No module named 'pkg_resources'`, repair the
existing environment and repeat the smoke without reinstalling the model stack:

```bash
"$HOME/.conda/envs/dfot/bin/python" -m pip install 'setuptools==80.9.0'
export PATH="$HOME/.conda/envs/dfot/bin:$PATH"
"$HOME/.conda/envs/dfot/bin/python" -u -m external_baselines.environment_smoke --gpu 0
```

TorchMetrics 0.11.4 requires this legacy setuptools module. Setup now explicitly
installs the compatible version; a constraint alone does not install a dependency.

## Configure once on the cluster

Copy `external_baselines/config.example.json` to your own JSON config. Set:

- Absolute Python executables for the existing separate DFoT, MemCam, and VBench
  environments. The launcher Python needs NumPy; workers need upstream dependencies
  plus imageio/ffmpeg. Both `ffmpeg` and `ffprobe` must be on PATH.
- The actual DFoT_RE10K checkpoint and the **same local I3D detector** used by MemCam.
- The canonical manifest, dataset, MemCam/VBench repositories, and fresh study root.
- `models.dfot.intrinsics`: verified **normalized source-image** `[fx, fy, cx, cy]`.
  Set `intrinsics_source` to the calibration/export metadata establishing them.
  No calibration is invented. This adapter currently requires constant intrinsics.
- `path_map`, if necessary: a JSON mapping of old path roots to current roots.
  Only filesystem fields are remapped; the original manifest and hash are preserved.

From this checkout, on an allocated GPU, first run:

```bash
bash external_baselines/run.sh --config /absolute/path/study.json --gpu 0 --phase smoke
```

This validates all 15 rows and runs a separate 73-frame smoke video per requested
model. Review `smoke/`, converted camera arrays, frame maps in `work/smoke/`, and
logs. It prints a rough linear full-cohort cost estimate; long-history cost may
be greater. Inspect output geometry and chronology before approving full work.

Then run **one resumable generation + evaluation + export command**:

```bash
bash external_baselines/run.sh --config /absolute/path/study.json --gpu 0 --phase run --approved-gpu-hours 24
```

Replace `24` with your allocated time. It prevents starting another generation job
after that window, but does not interrupt an in-progress video or metric job.
Use `STUDY_PYTHON=/absolute/path/python` if `python3` lacks NumPy. Rerun the same
command to resume. No GPU work is performed by `--phase preflight`.

## FramePack opt-in

The inspected local checkout is **original FramePack**, not F1. It has no numeric
camera input. To include its native image-to-video comparison, add the object
from `framepack.example.json` under `models.framepack` and pass
`--allow-native-framepack` to both commands. Provision its upstream
`repo/hf_download` cache first; runtime is offline. The worker source is pinned by
SHA256; a differing installed worker must be inspected before updating this pin.

The wrapper executes the pinned upstream model initialization and worker without
launching Gradio. It propagates the worker's caught errors and uses its final file,
retaining the original backward section construction and overlap logic. A nominal
180s request rounds to **5401 native frames**, rather than DFoT's 5397. Untouched
native FramePack video receives standard VBench only. LPIPS/FVD remain missing;
its crop, duration, and unconditioned path do not establish paired-view fidelity.
The upstream worker writes intermediate previews in `work/`, never in the metric
input directory. Disk budgeting must account for those retained intermediate files.

## DFoT contract

Uses the upstream single-image long-video RE10K configuration: continuous diffusion,
50 sampling steps, eight-frame model windows, 1/16 keyframe density, stabilized
vanilla history guidance 4.0/0.02 and interpolation guidance 1.5. Only generated
keyframes condition interpolation. Prompts are recorded but unsupported by DFoT.
The initial image is included as output index zero. Future RGB/GT paths are excluded
from worker jobs. All future poses are permitted conditioning.

Pose conversion follows MemCam's `Rz @ Ry @ Rx`, centimeters-to-meters translation,
then maps UE forward/right/up to OpenCV right/down/forward and inverts c2w to w2c.
DFoT applies its upstream first-camera normalization. A full-view 256-square resize
preserves normalized intrinsics and avoids a hidden crop. `camera_conditions.npy`
and explicit pose-key/dataset-index/timestamp maps accompany each DFoT job.

The launcher checks complete decode, exact frame count/FPS, checkpoint/config/input
hashes and video hash before resuming. Outputs and receipts are installed atomically;
smoke receipts cannot stand in for a full video. A lock prevents concurrent study
writers. Generation and metrics run sequentially on the explicitly assigned GPU.
Changing code/config/inputs requires a new study root.

## Metrics and existing MemCam results

DFoT uses the existing MemCam prefix evaluator at 180s with LPIPS stride 30/224px and
StyleGAN-V I3D FVD (16 frames, 4 clips/video, stride 4, 224px). FVD is cohort-level.
Standard VBench receives exactly the intended native videos and all six dimensions;
imaging-quality per-video values are divided by 100 once. CSV scores use [0,1].

To reuse historical rows, optionally add `reuse_results` objects with:
`run_name`, `manifest`, `generation_config`, `checkpoint`, `quality_dir`, and
`vbench_dir`. Point to actual saved artifacts. The launcher checks path-independent
cohort identity, 180s metric coverage/configuration, and all six VBench identities.
It records source paths/hashes. You must inspect the saved generation config for
one observed frame, camera conditioning, seed, and checkpoint provenance before
adding it; these cannot be inferred reliably from arbitrary legacy config formats.
Unconfigured historical rows are omitted, never replaced by headline constants.
VBench-Long is not included in this launcher.

Outputs: `tables/coverage.csv`, `tables/scores.csv`, `tables/comparison.tex`,
`provenance.json`, canonical videos under each model directory, and per-job configs,
camera maps and resources in `work/`. Failures return nonzero and preserve logs.
DFoT resource receipts separate startup, synchronized generation, and writing.
FramePack reports its upstream generation-and-writing boundary together. Process RSS
and peak allocated/reserved CUDA bytes include native output/history buffers; they
are not estimates of a discrete retrieval archive. Retrieval latency is unavailable.

## Local verification and limits

```bash
python3 -m unittest discover -s tests -p 'test_external_baselines.py' -v
```

Contract tests cover remapping, known translation/rotation, no future-GT fields,
resume identity/frame/decode validation, exact metric cohorts/scaling, and subprocess
failure propagation. GPU model loading, long-video memory use, image geometry, and
cluster environment compatibility require the smoke run; they were not run locally.
