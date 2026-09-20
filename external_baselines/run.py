"""Local/cluster entry point. No SSH, scheduler submission, training, or downloads."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from external_baselines.common import (DIMENSIONS, cohort_identity, digest, generation_input,
    identity, probe, remap, resumable, save, validate_rows)

ROOT = Path(__file__).resolve().parents[1]


def run(command, log, cwd, env):
    print('+', ' '.join(map(str, command)), flush=True)
    with Path(log).open('a') as output:
        proc = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='', flush=True)
            output.write(line)
            output.flush()
        proc.stdout.close()
        if proc.wait():
            raise RuntimeError(f'Command failed ({proc.returncode}); see {log}')


def csv_write(path, rows, fields):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def collect_quality(directory, rows, run_name):
    summary = json.loads((directory / 'summary.json').read_text())
    records = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines() if s.strip()]
    expected = {r['output_prefix'] + 'custom.mp4' for r in rows}
    names = [Path(r['output']).name for r in records]
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError('Quality evaluator returned a different cohort')
    if any(r['status'] != 'completed' or r['run_name'] != run_name or
           float(r['duration_sec']) != 180 or r['num_frames_expected'] != 5397 for r in records):
        raise ValueError('Incomplete/mixed quality results')
    config = summary['metric_config']
    required = dict(frame_stride=30, learned_image_size=224, fvd_clip_length=16,
                    fvd_clips_per_video=4, fvd_frame_stride=4, fvd_image_size=224,
                    fvd_backend='styleganv_i3d')
    if any(config.get(k) != v for k, v in required.items()) or config.get('max_frames') not in (None, 0):
        raise ValueError('Incompatible metric protocol')
    group = summary['by_duration']['180']
    values = [r['lpips_alex'] for r in records]
    if group['fvd_clips'] != 60 or not all(math.isfinite(x) for x in values + [group['fvd']]):
        raise ValueError('Missing cohort FVD or LPIPS')
    return {'LPIPS': sum(values) / len(values), 'FVD': group['fvd']}


def collect_bench(directory, rows):
    files = list(directory.glob('*_eval_results.json'))
    if len(files) != 1:
        raise ValueError('Expected exactly one VBench result')
    payload = json.loads(files[0].read_text())
    expected = {r['output_prefix'] + 'custom.mp4' for r in rows}
    result = {}
    for dim in DIMENSIONS:
        records = payload[dim][1]
        names = [Path(r['video_path']).name for r in records]
        if len(names) != 15 or set(names) != expected:
            raise ValueError(f'{dim}: wrong cohort')
        values = [float(r['video_results']) / (100 if dim == 'imaging_quality' else 1) for r in records]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError(f'{dim}: invalid scores')
        result[dim] = sum(values) / len(values)
    return result


def evaluate(cfg, model_name, rows, study, env):
    output = study / model_name
    metric_root = study / 'metrics' / model_name
    metric_root.mkdir(parents=True, exist_ok=True)
    scores = {}
    # Stock FramePack is native I2V; never run paired quality metrics against trajectory GT.
    if model_name == 'dfot':
        directory = metric_root / 'quality' / model_name
        try:
            scores.update(collect_quality(directory, rows, model_name))
        except (OSError, ValueError, KeyError):
            command = [cfg['metric_python'], '-u', 'utils/evaluate_context_memory_prefix_curves.py',
                '--manifest', study / 'inputs/manifest.jsonl', '--dataset_root', cfg['dataset_root'],
                '--model_output_dir', output, '--metrics_dir', metric_root / 'quality',
                '--run_name', model_name, '--source_duration', '180', '--eval_durations', '180',
                '--learned_metrics', 'lpips,fvd', '--metric_device', 'cuda', '--metric_batch_size', '8',
                '--frame_stride', '30', '--learned_image_size', '224', '--fvd_backend', 'styleganv_i3d',
                '--fvd_clip_length', '16', '--fvd_clips_per_video', '4', '--fvd_frame_stride', '4',
                '--fvd_image_size', '224', '--fvd_detector_path', cfg['fvd_detector'], '--strict']
            run(command, study / 'logs' / f'{model_name}.quality.log', cfg['memcam_repo'], env)
            scores.update(collect_quality(directory, rows, model_name))
    directory = metric_root / 'vbench'
    directory.mkdir(exist_ok=True)
    try:
        scores.update(collect_bench(directory, rows))
    except (OSError, ValueError, KeyError):
        # output directory holds only canonical completed mp4 files and JSON receipts.
        run([cfg['vbench_python'], '-u', 'evaluate.py', '--videos_path', output,
             '--output_path', directory, '--mode', 'custom_input', '--dimension', *DIMENSIONS],
             study / 'logs' / f'{model_name}.vbench.log', cfg['vbench_repo'], env)
        scores.update(collect_bench(directory, rows))
    return scores


def export(study, coverage, scores):
    csv_write(study / 'tables/coverage.csv', coverage,
              ['model', 'row', 'scene', 'start_frame', 'status', 'frames', 'metrics_complete', 'error'])
    fields = ['model', 'checkpoint', 'camera_input', 'observed_frames', 'nominal_duration',
              'actual_duration', 'N', 'LPIPS', 'FVD', *DIMENSIONS, 'generation_wall_seconds',
              'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'peak_process_rss_bytes']
    csv_write(study / 'tables/scores.csv', scores, fields)
    def tex(v):
        if v is None or v == '':
            return '--'
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v).replace('_', r'\_').replace('&', r'\&')
    columns = ['model', 'N', 'LPIPS', 'FVD', *DIMENSIONS]
    lines = [r'\begin{tabular}{l' + 'r' * (len(columns) - 1) + '}',
             ' & '.join(map(tex, columns)) + r' \\']
    lines.extend(' & '.join(tex(row.get(k)) for k in columns) + r' \\' for row in scores)
    lines.append(r'\end{tabular}')
    (study / 'tables/comparison.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--gpu', required=True, help='One explicitly allocated CUDA GPU index/UUID')
    parser.add_argument('--phase', choices=['preflight', 'smoke', 'run'], default='smoke')
    parser.add_argument('--approved-gpu-hours', type=float, default=0)
    parser.add_argument('--allow-native-framepack', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    study = Path(cfg['study_root']).expanduser().resolve()
    for d in ('inputs', 'logs', 'tables', 'metrics', 'work', 'smoke'):
        (study / d).mkdir(parents=True, exist_ok=True)
    lock = (study / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    coverage, scores = [], []
    try:
        source = Path(cfg['manifest']).expanduser()
        original = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        rows = remap(original, cfg.get('path_map', {}))
        validate_rows(rows)
        for row in rows:
            gt_dir = Path(cfg['dataset_root']) / 'frames' / row['scene']
            if gt_dir.resolve() != Path(row['gt_frames_dir']).resolve():
                raise ValueError('Manifest GT directory differs from evaluator dataset_root/frames/scene')
            for index in range(row['start_frame'], row['start_frame'] + row['num_frames']):
                if not (gt_dir / f'{index:04d}.png').is_file():
                    raise FileNotFoundError(gt_dir / f'{index:04d}.png')
        for binary in ('ffmpeg', 'ffprobe'):
            if shutil.which(binary) is None:
                raise FileNotFoundError(f'{binary} must be on PATH')
        for model in cfg['models']:
            if model not in ('dfot', 'framepack'):
                raise ValueError(f'Unsupported model: {model}')
        if 'framepack' in cfg['models'] and not args.allow_native_framepack:
            raise ValueError('FramePack has no poses. Explicitly pass --allow-native-framepack for native I2V only')
        if ',' in args.gpu:
            raise ValueError('Use one explicitly allocated GPU; jobs run sequentially')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        model_hashes = {}
        for name, model in cfg['models'].items():
            if not Path(model['python']).is_file():
                raise FileNotFoundError(model['python'])
            if name == 'dfot':
                if not model.get('intrinsics') or not model.get('intrinsics_source'):
                    raise ValueError('Set verified normalized intrinsics AND intrinsics_source in config')
                model_hashes[name] = digest(model['checkpoint'])
                from external_baselines.camera import convert
                for r in rows:
                    convert(r['pose_path'], r['start_frame'], r['num_frames'], model['intrinsics'], model.get('translation_scale', 100))
            else:
                worker = Path(model['repo']) / 'demo_gradio.py'
                if digest(worker) != model['worker_sha256']:
                    raise ValueError('FramePack worker hash mismatch')
                weights = list((Path(model['repo']) / 'hf_download').rglob('*.safetensors'))
                if not weights:
                    raise ValueError('No FramePack weights in repo/hf_download; provision upstream cache first')
                model_hashes[name] = {str(p): digest(p) for p in weights}
        for key in ('metric_python', 'vbench_python', 'fvd_detector'):
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(cfg[key])
        if shutil.disk_usage(study).free < cfg.get('minimum_free_gib', 50) * 2**30:
            raise ValueError('Insufficient study disk space')
        hardware = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
            '--query-gpu=name,memory.total,memory.free', '--format=csv'], text=True)
        input_hashes = [{k: digest(r[k]) for k in ('input_image', 'pose_path')} for r in rows]
        code_hashes = {str(p.relative_to(ROOT)): digest(p)
                       for folder in ('external_baselines', 'algorithms', 'configurations', 'utils')
                       for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml')}
        for filename in ('evaluate_context_memory_prefix_curves.py', 'evaluate_context_memory.py'):
            path = Path(cfg['memcam_repo']) / 'utils' / filename
            code_hashes[str(path)] = digest(path)
        code_hashes[cfg['fvd_detector']] = digest(cfg['fvd_detector'])
        signature = identity(dict(config=cfg, cohort=cohort_identity(rows), inputs=input_hashes,
                                  checkpoints=model_hashes, code=code_hashes))
        provenance_file = study / 'provenance.json'
        if provenance_file.exists() and json.loads(provenance_file.read_text())['signature'] != signature:
            raise ValueError('Study inputs/config/code changed. Choose a new study_root to preserve old work')
        provenance = dict(signature=signature, config=cfg, source_sha256=digest(source),
            cohort_identity=cohort_identity(rows), input_hashes=input_hashes, checkpoints=model_hashes,
            code_hashes=code_hashes, hardware=hardware, cpu=platform.processor(), platform=platform.platform(),
            dfot_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            protocols={'dfot': '30 FPS, single GT initial frame, full-view resize 256 square, UE -> OpenCV w2c',
                       'framepack': 'native I2V, no camera, upstream chronology and native crop, VBench only'})
        save(provenance_file, provenance)
        shutil.copyfile(source, study / 'inputs/manifest.original.jsonl')
        (study / 'inputs/manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        save(study / 'inputs/config.json', cfg)
        print(hardware, f'Validated {len(rows)} trajectories. Signature: {signature}', flush=True)
        if args.phase == 'preflight':
            return 0
        if args.phase == 'run' and args.approved_gpu_hours <= 0:
            raise ValueError('Full generation requires --approved-gpu-hours after reviewing smoke results')
        deadline = time.monotonic() + args.approved_gpu_hours * 3600
        failed = False
        for name, model in cfg['models'].items():
            if args.phase == 'run':
                smoke_receipt = study / 'smoke' / name / 'receipt.json'
                smoke = json.loads(smoke_receipt.read_text())
                if smoke['signature'] != signature:
                    raise ValueError('Run a successful smoke with this config first')
            selected = rows[:1] if args.phase == 'smoke' else rows
            run_dir = study / ('smoke/' + name if args.phase == 'smoke' else name)
            run_dir.mkdir(parents=True, exist_ok=True)
            expected_names = {r['output_prefix'] + 'custom.mp4' for r in selected}
            if set(p.name for p in run_dir.glob('*.mp4')) - expected_names:
                raise ValueError('Unrelated videos in model output directory')
            resources = []
            model_coverage = []
            for index, row in enumerate(selected):
                record = dict(model=name, row=index, scene=row['scene'], start_frame=row['start_frame'],
                              status='pending', frames=0, metrics_complete=False)
                coverage.append(record)
                model_coverage.append(record)
                work = study / 'work' / args.phase / name / str(index)
                work.mkdir(parents=True, exist_ok=True)
                inp = generation_input(row)
                if args.phase == 'smoke':
                    inp['num_frames'] = 73
                # Original FramePack: sections of 36 frames plus initial frame.
                frames = inp['num_frames'] if name == 'dfot' else round(inp['num_frames'] / 36) * 36 + 1
                video = run_dir / (row['output_prefix'] + 'custom.mp4')
                receipt = video.with_suffix('.json')
                try:
                    if not resumable(video, receipt, signature, frames):
                        if args.phase == 'run' and time.monotonic() >= deadline:
                            raise RuntimeError('Approved GPU-time window exhausted; resume with another allocation')
                        job = dict(kind=name, model=model, input=inp, seed=cfg.get('seed', 42),
                                   work=str(work), output=str(work / 'partial.mp4'))
                        save(work / 'job.json', job)
                        print(f'{name}: trajectory {index + 1}/{len(selected)}; target {frames} frames', flush=True)
                        run([model['python'], '-u', '-m', 'external_baselines.worker', work / 'job.json'],
                            study / 'logs' / f'{args.phase}.{name}.{index}.log', ROOT, env)
                        info = probe(work / 'partial.mp4')
                        if info['frames'] != frames or info['fps'] != 30:
                            raise ValueError(f'Output frame contract mismatch: {info}; expected {frames}')
                        (work / 'partial.mp4').replace(video)
                        stats = json.loads((work / 'resources.json').read_text())
                        save(receipt, dict(signature=signature, video_sha256=digest(video), info=info, resources=stats))
                    data = json.loads(receipt.read_text())
                    resources.append(data['resources'])
                    record.update(status='completed', frames=frames)
                    if args.phase == 'smoke':
                        estimate = data['resources']['total_seconds'] * 5397 / inp['num_frames'] * 15 / 3600
                        save(run_dir / 'receipt.json', dict(signature=signature, estimated_gpu_hours=estimate,
                             note='Linear short-run estimate; long-history cost may differ', video=str(video)))
                        print(f'{name}: rough 15-video estimate {estimate:.2f} GPU-hours. Inspect {video}', flush=True)
                except Exception as exc:
                    record.update(status='failed', error=str(exc))
                    traceback.print_exc()
                    failed = True
                export(study, coverage, scores)
            score = dict(model=name, checkpoint=model.get('checkpoint', 'FramePackI2V_HY'),
                         camera_input=name == 'dfot', observed_frames=1, nominal_duration=180,
                         actual_duration=frames / 30, N=len(resources))
            if resources:
                score['generation_wall_seconds'] = sum(r['total_seconds'] for r in resources)
                for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'peak_process_rss_bytes'):
                    score[key] = max(r[key] for r in resources)
            if args.phase == 'run' and all(r['status'] == 'completed' for r in model_coverage):
                try:
                    score.update(evaluate(cfg, name, rows, study, env))
                    for record in model_coverage:
                        record['metrics_complete'] = True
                except Exception:
                    traceback.print_exc()
                    failed = True
            scores.append(score)
            export(study, coverage, scores)
        if args.phase == 'run':
            for legacy in cfg.get('reuse_results', []):
                try:
                    legacy_rows = [json.loads(s) for s in Path(legacy['manifest']).read_text().splitlines() if s.strip()]
                    if cohort_identity(legacy_rows) != cohort_identity(rows):
                        raise ValueError('Historical cohort differs')
                    # Reuse requires the actual saved generation config, not filename seed labels.
                    saved_config = Path(legacy['generation_config'])
                    digest(saved_config)
                    values = collect_quality(Path(legacy['quality_dir']), rows, legacy['run_name'])
                    values.update(collect_bench(Path(legacy['vbench_dir']), rows))
                    historical = dict(model=legacy['run_name'], checkpoint=legacy['checkpoint'],
                        camera_input=True, observed_frames=1, nominal_duration=180,
                        actual_duration=5397 / 30, N=15, **values)
                    scores.append(historical)
                    for i, r in enumerate(rows):
                        coverage.append(dict(model=legacy['run_name'], row=i, scene=r['scene'],
                            start_frame=r['start_frame'], status='reused_verified_metrics',
                            frames=5397, metrics_complete=True))
                    provenance.setdefault('reused_results', []).append(dict(
                        source=legacy, generation_config_sha256=digest(saved_config),
                        quality_summary_sha256=digest(Path(legacy['quality_dir']) / 'summary.json')))
                except Exception as exc:
                    failed = True
                    coverage.append(dict(model=legacy['run_name'], status='reuse_failed', error=str(exc)))
                    traceback.print_exc()
            save(provenance_file, provenance)
            export(study, coverage, scores)
        return int(failed)
    except Exception as exc:
        save(study / 'logs/failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        export(study, coverage, scores)
        raise


if __name__ == '__main__':
    sys.exit(main())
