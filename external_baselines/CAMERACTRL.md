# CameraCtrl SVD-XT replacement baseline: first smoke

This path uses the official **svd branch**, pinned at
`1f9d8baeff66f6c60bb9848dfafcc667b3a22a28`, with
`hehao13/CameraCtrl_SVD_ckpts/CameraCtrl_svdxt.ckpt` and
`stabilityai/stable-video-diffusion-img2vid-xt`. It accepts an initial image and
numeric camera poses, with no text-prompt conditioning. It is a pretrained external
baseline, not a Context-as-Memory-trained checkpoint or a MemCam variant.

Official sources:
- https://github.com/hehao13/CameraCtrl/tree/svd
- https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts

On the cluster, after pushing/pulling this checkout:

```bash
bash external_baselines/setup_cameractrl.sh
export PATH="$HOME/.conda/envs/cameractrl/bin:$PATH"
python -u -m external_baselines.cameractrl_smoke --config study.json --gpu 1 \
  --output /data/ab575577/MemCam/outputs/cameractrl_smoke_01 --download
```

The setup creates a separate environment and upstream checkout at `~/CameraCtrl`.
Set `CAMERA_ENV` and `CAMERA_REPO` to override; pass `--repo` to smoke for a custom
checkout. It uses Torch 2.7.1 CUDA 12.8 for Blackwell and disables optional xformers.
The upstream 2024 xformers/Torch pins must not replace the Blackwell build. Exact
pipeline dependencies are diffusers 0.24, transformers 4.39.3, huggingface-hub 0.25.2.
The GPU environment has not been tested locally; the first cluster smoke is the
integration check. `--download` explicitly permits model downloads; without it,
only locally cached models are used. If Hugging Face requests authorization,
accept the base model's terms with your own account and authenticate before retrying.

Default output is **576x320**, the official inference default. To test **640x352
native generation**, add `--width 640 --height 352`; the receipt labels this as an
experimental resolution. It is not a post-generation upscale. Both sizes use an
explicit full-view resize, without a crop, preserving normalized intrinsics.
The paper-based 52.67-degree nominal horizontal FOV assumptions remain recorded.

This smoke produces **25 frames at 7 FPS**. It samples the manifest's numerical
trajectory at nearest source-frame times and records both target/source timestamps
and pose keys in `frame_map.json`. Extrinsics use the inspected CaM UE-to-OpenCV
conversion, then upstream first-camera normalization and Pluecker ray encoding.
Only the initial PNG is opened; no future GT image is loaded. The first generated
frame is preserved as generated, not substituted with GT.

Outputs include the native MP4, resized input, poses, frame map, resolved model
config and receipt with checkpoint/input/video hashes, inference settings, native
sizes and hardware/timing/memory. The empty-output-directory requirement preserves
previous attempts; use a new output directory for retries. Partial videos never
receive a success receipt.

This is a short engineering smoke, **not a 180-second cohort run**. Upstream supports
14 frames for SVD or 25 for SVD-XT. It does not expose a native 180-second history
rollout. Chaining outputs as initial images would be an additional continuation
protocol that needs boundary/time mapping and validation before long comparison.
No paired LPIPS/FVD or legacy MemCam scores are exported for this smoke. The existing
DFoT/FramePack launcher remains available but is not invoked by this path.
