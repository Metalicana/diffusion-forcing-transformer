"""Check dependencies, real CUDA execution and video I/O without data/weights."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='inherit', help='Keep the scheduler mask by default')
    parser.add_argument('--output', type=Path, default=Path('outputs/dfot_environment/smoke.json'))
    args = parser.parse_args()
    if ',' in args.gpu:
        parser.error('Select one allocated GPU')
    if args.gpu != 'inherit':
        if os.environ.get('SLURM_JOB_ID'):
            parser.error('Inside Slurm use --gpu inherit; do not replace its GPU mask')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    start = time.monotonic()
    report = {'python': sys.executable, 'python_version': sys.version, 'selected_gpu': args.gpu,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'job': os.environ.get('SLURM_JOB_ID')}
    try:
        for name in ('pkg_resources', 'torch', 'torchvision', 'lightning', 'hydra', 'transformers',
                     'diffusers', 'imageio', 'algorithms.dfot.dfot_video_pose'):
            print(f'Importing {name} ...', flush=True)
            module = importlib.import_module(name)
            report[name] = getattr(module, '__version__', 'import OK')
        import torch
        import torchvision
        import numpy as np
        import imageio.v2 as imageio
        from external_baselines.common import probe
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: check GPU allocation, visibility and driver compatibility')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['cuda_runtime'] = torch.version.cuda
        report['compute_capability'] = list(torch.cuda.get_device_capability(0))
        report['compiled_architectures'] = torch.cuda.get_arch_list()
        print(f"CUDA runtime: {report['cuda_runtime']}; capability: {report['compute_capability']}; compiled: {report['compiled_architectures']}", flush=True)
        print(f"Testing CUDA kernels on {report['gpu']} ...", flush=True)
        with torch.inference_mode():
            x = torch.randn(256, 256, device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                result = x @ x.T
            if not torch.isfinite(result).all():
                raise RuntimeError('Non-finite CUDA matrix result')
            torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]], device='cuda'),
                                torch.tensor([1.], device='cuda'), 0.5)
            torch.cuda.synchronize()
        print('Testing MP4 encode/decode and ffprobe ...', flush=True)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'smoke.mp4'
            with imageio.get_writer(video, fps=30, codec='libx264', macro_block_size=1) as writer:
                for k in range(8):
                    writer.append_data(np.full((64, 64, 3), k * 30, dtype=np.uint8))
            info = probe(video)
            if info['frames'] != 8 or info['fps'] != 30:
                raise RuntimeError(f'Video I/O mismatch: {info}')
            report['video_io'] = info
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Smoke report: {args.output.resolve()}', flush=True)
    print('Environment smoke passed. No model weights loaded or study videos generated.', flush=True)


if __name__ == '__main__':
    main()
