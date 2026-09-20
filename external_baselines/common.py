"""Dependency-light study contracts; generation never receives a GT directory."""
import hashlib
import json
import subprocess
from pathlib import Path

PATH_FIELDS = ('input_image', 'pose_path', 'gt_frames_dir', 'overlap_dir')
DIMENSIONS = ('subject_consistency', 'background_consistency', 'motion_smoothness',
              'dynamic_degree', 'aesthetic_quality', 'imaging_quality')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def remap(rows, mappings):
    result = []
    for row in rows:
        row = dict(row)
        for key in PATH_FIELDS:
            if row.get(key):
                for source, target in sorted(mappings.items(), key=lambda p: -len(p[0])):
                    path = Path(row[key])
                    try:
                        suffix = path.relative_to(source)
                    except ValueError:
                        continue
                    row[key] = str(Path(target) / suffix)
                    break
        result.append(row)
    return result


def cohort_identity(rows):
    # The original hash is retained too; input content hashes protect remapped paths.
    return identity([{k: v for k, v in r.items() if k not in PATH_FIELDS} for r in rows])


def validate_rows(rows, expected=15):
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} manifest rows, got {len(rows)}')
    names = []
    for r in rows:
        for key in ('scene', 'start_frame', 'duration_sec', 'fps', 'num_frames',
                    'input_image', 'pose_path', 'gt_frames_dir', 'output_prefix', 'split_id', 'split_seed'):
            if key not in r:
                raise ValueError(f'Missing manifest field: {key}')
        name = r['output_prefix'] + 'custom.mp4'
        if Path(name).name != name or name.startswith('.'):
            raise ValueError(f'Unsafe output name: {name}')
        names.append(name)
        if r['fps'] != 30 or r['num_frames'] != 5397 or r['duration_sec'] != 180:
            raise ValueError('This protocol requires the canonical 5397-frame, 30-FPS, nominal 180s cohort')
        for key in PATH_FIELDS:
            if r.get(key) and not Path(r[key]).exists():
                raise FileNotFoundError(r[key])
    if len(set(names)) != len(names):
        raise ValueError('Duplicate output identity')


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                   check=True, capture_output=True)
    stream['frames'] = int(stream['nb_read_frames'])
    n, d = map(int, stream['r_frame_rate'].split('/'))
    stream['fps'] = n / d
    return stream


def resumable(video, receipt, signature, frames):
    try:
        r = json.loads(Path(receipt).read_text())
        if r['signature'] != signature or r['video_sha256'] != digest(video):
            return False
        info = probe(video)
        return info['frames'] == frames and info['fps'] == 30
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


def generation_input(row):
    return {k: row[k] for k in ('input_image', 'pose_path', 'start_frame', 'num_frames', 'fps', 'prompt') if k in row}
