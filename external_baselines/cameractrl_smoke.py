"""One native 25-frame CameraCtrl SVD-XT CaM smoke; never a 180s evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import resource
from external_baselines.common import digest, identity, probe, remap, save
from external_baselines.camera import convert, nominal_intrinsics

COMMIT = '1f9d8baeff66f6c60bb9848dfafcc667b3a22a28'


def sample_indices(source_fps, fps=7, count=25):
    if source_fps < fps or count < 1:
        raise ValueError('Smoke requires source FPS >= output FPS and a positive frame count')
    return [int(k * source_fps / fps + .5) for k in range(count)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='study.json', type=Path)
    p.add_argument('--repo', type=Path, default=Path.home() / 'CameraCtrl')
    p.add_argument('--gpu', default='1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--width', type=int, default=576)
    p.add_argument('--height', type=int, default=320)
    p.add_argument('--download', action='store_true', help='Explicitly provision official SVD-XT and camera weights')
    args = p.parse_args()
    if ',' in args.gpu or args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8:
        p.error('Select one GPU and positive dimensions divisible by eight')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = json.loads(args.config.read_text())
    rows = remap([json.loads(x) for x in Path(cfg['manifest']).read_text().splitlines() if x.strip()], cfg.get('path_map', {}))
    row = rows[args.row]
    repo = args.repo.expanduser().resolve()
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if actual != COMMIT:
        raise ValueError(f'Expected inspected SVD revision {COMMIT}, got {actual}')
    if subprocess.check_output(['git', 'diff', 'HEAD', '--', '*.py', '*.yaml'], cwd=repo):
        raise ValueError('Tracked upstream CameraCtrl source/config files have local modifications')
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Choose an empty smoke output directory to preserve prior artifacts')
    save(args.output / 'request.json', dict(row=row, args={k: str(v) for k, v in vars(args).items()}))
    from huggingface_hub import snapshot_download, hf_hub_download
    print('Resolving SVD-XT base and CameraCtrl weights', flush=True)
    base = snapshot_download('stabilityai/stable-video-diffusion-img2vid-xt',
        allow_patterns=['model_index.json', 'scheduler/*', 'feature_extractor/*',
                        'image_encoder/config.json', 'image_encoder/model.safetensors',
                        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
                        'unet/config.json', 'unet/diffusion_pytorch_model.safetensors'],
        local_files_only=not args.download)
    checkpoint = hf_hub_download('hehao13/CameraCtrl_SVD_ckpts', 'CameraCtrl_svdxt.ckpt', local_files_only=not args.download)
    os.environ['HF_HUB_OFFLINE'] = '1'
    print('Importing CameraCtrl and Torch', flush=True)
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    import imageio.v2 as imageio
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('cameractrl_upstream_inference', repo / 'inference.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    with Image.open(row['input_image']) as original:
        source_size = original.size
        intrinsics = nominal_intrinsics(*source_size)['normalized']
        image = original.convert('RGB').resize((args.width, args.height), Image.Resampling.LANCZOS)
    image.save(args.output / 'initial_resized.png')
    indices = sample_indices(row['fps'])
    if indices[-1] >= row['num_frames']:
        raise ValueError('Requested smoke exceeds manifest trajectory')
    conditions, keys = convert(row['pose_path'], row['start_frame'], indices[-1] + 1, intrinsics)
    selected = conditions[indices]
    cameras = [upstream.Camera([k, *v[:4], 0, 0, *v[4:]]) for k, v in enumerate(selected)]
    relative = upstream.get_relative_pose(cameras, zero_first_frame_scale=True)
    K = selected[:, :4] * np.array([args.width, args.height, args.width, args.height])
    embedding = upstream.ray_condition(torch.tensor(K, dtype=torch.float32)[None],
        torch.tensor(relative, dtype=torch.float32)[None], args.height, args.width, device='cpu')
    embedding = embedding.permute(0, 1, 4, 2, 3).contiguous().to('cuda')
    np.save(args.output / 'relative_c2w.npy', relative)
    save(args.output / 'frame_map.json', [dict(output_index=k, output_timestamp=k / 7,
        dataset_index=row['start_frame'] + index, dataset_timestamp=index / row['fps'],
        pose_key=keys[index]) for k, index in enumerate(indices)])
    settings = OmegaConf.load(repo / 'configs/train_cameractrl/svd_320_576_cameractrl.yaml')
    save(args.output / 'model_config.json', OmegaConf.to_container(settings, resolve=True))
    start = time.monotonic()
    print('Loading official SVD-XT and camera adapter', flush=True)
    pipeline = upstream.get_pipeline(base, settings['unet_subfolder'], settings['down_block_types'],
        settings['up_block_types'], settings['pose_encoder_kwargs'], settings['attention_processor_kwargs'],
        checkpoint, False, 'cuda')
    torch.cuda.synchronize()
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device='cuda').manual_seed(cfg.get('seed', 42))
    with torch.inference_mode():
        output = pipeline(image=image, pose_embedding=embedding, height=args.height, width=args.width,
            num_frames=25, num_inference_steps=25, min_guidance_scale=1., max_guidance_scale=3.,
            fps=7, do_image_process=True, decode_chunk_size=8, generator=generator, output_type='pt').frames[0]
    torch.cuda.synchronize()
    generated = time.monotonic()
    with imageio.get_writer(args.output / 'partial.mp4', fps=7, codec='libx264', macro_block_size=1) as writer:
        for frame in output:
            writer.append_data((frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    info = probe(args.output / 'partial.mp4')
    if (info['frames'], info['width'], info['height'], info['fps']) != (25, args.width, args.height, 7):
        raise ValueError(f'Unexpected video shape: {info}')
    video = args.output / 'cameractrl_svdxt_smoke.mp4'
    (args.output / 'partial.mp4').replace(video)
    save(args.output / 'receipt.json', dict(model='CameraCtrl SVD-XT', code_commit=actual,
        checkpoint=checkpoint, checkpoint_sha256=digest(checkpoint), base_snapshot=base,
        base_weight_hashes={str(p.relative_to(base)): digest(p) for p in Path(base).rglob('*.safetensors')},
        input_sha256=digest(row['input_image']), pose_sha256=digest(row['pose_path']),
        seed=cfg.get('seed', 42), native_size=source_size, output=info, normalized_intrinsics=intrinsics,
        calibration='nominal: published 52.67deg assumed horizontal; square native pixels; centered principal point; no distortion',
        preprocessing='full-view resize; image-edge coordinates; normalized intrinsics preserved',
        task='single initial image + numeric poses; no text conditioning; no future GT images',
        temporal_protocol='25 frames at 7 FPS; nearest 30-FPS dataset poses; frame zero is generated, not guaranteed identical to input',
        scope='engineering smoke only, no 180s continuation or matched-cohort metrics',
        experimental_resolution=(args.width, args.height) != (576, 320),
        startup_seconds=loaded-start, generation_seconds=generated-loaded,
        writing_and_validation_seconds=time.monotonic()-generated,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gpu=torch.cuda.get_device_name(), python=sys.executable, torch=torch.__version__, video_sha256=digest(video)))
    print(f'CameraCtrl smoke passed: {video}', flush=True)


if __name__ == '__main__':
    main()
