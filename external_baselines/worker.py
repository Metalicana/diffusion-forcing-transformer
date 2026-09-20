"""One isolated process per video; all model imports occur after progress output."""
import argparse
import ast
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from external_baselines.common import save, digest


def dfot(job, torch):
    import numpy as np
    from PIL import Image
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from algorithms.dfot.dfot_video_pose import DFoTVideoPose
    from external_baselines.camera import convert
    root = Path(__file__).resolve().parents[1]
    overrides = ['dataset=realestate10k_mini', 'algorithm=dfot_video_pose',
        'dataset.context_length=1', 'dataset.frame_skip=1',
        'algorithm.compile=false', 'algorithm.logging.metrics=[]',
        'algorithm.diffusion.is_continuous=true', '++algorithm.diffusion.precond_scale=0.125',
        '++algorithm.backbone.use_fourier_noise_embedding=true',
        'algorithm.tasks.prediction.keyframe_density=0.0625',
        'algorithm.tasks.interpolation.max_batch_size=4',
        'algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla',
        '+algorithm.tasks.prediction.history_guidance.guidance_scale=4.0',
        '+algorithm.tasks.prediction.history_guidance.stabilization_level=0.02',
        'algorithm.tasks.interpolation.history_guidance.name=vanilla',
        '+algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5']
    with initialize_config_dir(version_base=None, config_dir=str(root / 'configurations')):
        cfg = compose(config_name='config', overrides=overrides)
    cfg.dataset.n_frames = job['input']['num_frames']
    save(Path(job['work']) / 'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    print('Constructing DFoT and loading checkpoint', flush=True)
    model = DFoTVideoPose(cfg.algorithm)
    checkpoint = torch.load(job['model']['checkpoint'], map_location='cpu', weights_only=False)
    state = {k.replace('diffusion_model._orig_mod.', 'diffusion_model.'): v
             for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    model.eval().to('cuda')
    if model.is_latent_diffusion:
        raise ValueError('This adapter targets the official pixel-space DFoT_RE10K checkpoint')
    row = job['input']
    conds, keys = convert(row['pose_path'], row['start_frame'], row['num_frames'],
                          job['model']['intrinsics'], job['model'].get('translation_scale', 100.0))
    save(Path(job['work']) / 'intrinsics.json', {
        'status': job['model'].get('intrinsics_status', 'user_supplied'),
        'normalized': job['model']['intrinsics'],
        'source': job['model']['intrinsics_source']})
    np.save(Path(job['work']) / 'camera_conditions.npy', conds)
    save(Path(job['work']) / 'frame_map.json', [
        {'output_index': k, 'dataset_index': row['start_frame'] + k,
         'pose_key': keys[k], 'timestamp_sec': k / row['fps']} for k in range(len(keys))])
    # Full-view anisotropic resize: normalized intrinsics unchanged; no crop.
    image = Image.open(row['input_image']).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS)
    initial = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().to('cuda') / 255
    xs = torch.zeros((1, row['num_frames'], 3, 256, 256), device='cuda')
    xs[:, 0] = initial
    torch.cuda.synchronize()
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        output = model._unnormalize_x(model._predict_videos(
            model._normalize_x(xs), torch.from_numpy(conds)[None].to('cuda')))
        output[:, 0] = initial
    torch.cuda.synchronize()
    generated = time.monotonic()
    import imageio.v2 as imageio
    with imageio.get_writer(job['output'], fps=30, codec='libx264', macro_block_size=1) as writer:
        for frame in output[0]:
            writer.append_data((frame.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy())
    return {'loaded_at': loaded, 'generation_seconds': generated - loaded,
            'writing_seconds': time.monotonic() - generated, 'prompt_supported': False,
            'camera_input': True, 'observed_frames': 1, 'resolution': [256, 256],
            'spatial_transform': 'full-view resize, normalized intrinsics unchanged',
            'precision': '16-mixed', 'training_domain': 'RealEstate10K', 'retrieval_latency': None}


def framepack(job, torch):
    """Execute the pinned upstream initializer and worker, excluding its web UI."""
    import numpy as np
    from PIL import Image
    source = Path(job['model']['repo']) / 'demo_gradio.py'
    if digest(source) != job['model']['worker_sha256']:
        raise ValueError('FramePack worker changed; inspect it and update worker_sha256')
    tree = ast.parse(source.read_text())
    nodes = []
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'worker':
            # The GUI swallows errors; batch mode must propagate them.
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    child.body.append(ast.Raise())
            nodes.append(node)
            found = True
            break
        text = ast.unparse(node)
        if text.startswith(('parser', 'args =', 'print(args)', 'outputs_folder =', 'os.makedirs(outputs_folder')):
            continue
        nodes.append(node)
    if not found:
        raise ValueError('Unsupported FramePack variant: original demo_gradio worker required')
    sys.path.insert(0, str(source.parent))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    ns = {'__file__': str(source), '__name__': 'framepack_batch', 'outputs_folder': job['work']}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), ns)
    class Queue:
        def __init__(self):
            self.last = None
        def top(self):
            return None
        def push(self, item):
            if item[0] == 'file':
                self.last = item[1]
            if item[0] == 'progress':
                print(str(item[1][1:]), flush=True)
    class Stream:
        input_queue = Queue()
        output_queue = Queue()
    ns['stream'] = Stream()
    torch.cuda.synchronize()
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    ns['worker'](np.array(Image.open(job['input']['input_image']).convert('RGB')),
        job['input'].get('prompt', ''), '', job['seed'], job['input']['num_frames'] / 30,
        9, 25, 1.0, 10.0, 0.0, 6.0, False, 16)
    torch.cuda.synchronize()
    last = ns['stream'].output_queue.last
    if not last:
        raise RuntimeError('FramePack produced no final video')
    shutil.copyfile(last, job['output'])
    return {'loaded_at': loaded, 'generation_and_writing_seconds': time.monotonic() - loaded,
            'camera_input': False, 'observed_frames': 1, 'prompt_supported': True,
            'alignment': 'native unconditioned trajectory; no paired LPIPS/FVD comparison',
            'retrieval_latency': None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job')
    args = p.parse_args()
    job = json.loads(Path(args.job).read_text())
    start = time.monotonic()
    print('Importing Torch (no import timeout)', flush=True)
    import torch
    import numpy as np
    import random
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    torch.manual_seed(job['seed'])
    torch.cuda.manual_seed_all(job['seed'])
    stats = globals()[job['kind']](job, torch)
    stats['startup_seconds'] = stats.pop('loaded_at') - start
    stats.update(total_seconds=time.monotonic() - start,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                 peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                 gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
    save(Path(job['work']) / 'resources.json', stats)


if __name__ == '__main__':
    main()
