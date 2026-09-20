"""Create a separate, explicitly nominal camera configuration from native PNG sizes."""
import argparse
import json
from pathlib import Path
from PIL import Image
from external_baselines.camera import nominal_intrinsics
from external_baselines.common import remap, save


def prepare(config):
    cfg = json.loads(Path(config).read_text())
    rows = [json.loads(line) for line in Path(cfg['manifest']).expanduser().read_text().splitlines() if line.strip()]
    rows = remap(rows, cfg.get('path_map', {}))
    if len(rows) != 15:
        raise ValueError('Expected the canonical 15-row manifest')
    records = []
    for row in rows:
        with Image.open(row['input_image']) as image:
            records.append(dict(input_image=row['input_image'], **nominal_intrinsics(*image.size)))
    if len({tuple(r['native_size']) for r in records}) != 1:
        raise ValueError('Native dimensions vary: per-image intrinsics are required before using this adapter')
    model = cfg['models']['dfot']
    model['intrinsics'] = records[0]['normalized']
    model['intrinsics_status'] = 'nominal_published_fov_smoke_only'
    model['intrinsics_source'] = {
        'paper': 'https://arxiv.org/html/2506.03141v2#A2',
        'published_focal_length_mm': 24, 'published_fov_deg': 52.67, 'published_aperture': 10,
        'assumptions': ['FOV is horizontal', 'native PNG covers full stated FOV',
                        'native square pixels', 'centered principal point', 'no lens distortion'],
        'preprocessing': 'PIL LANCZOS full-view anisotropic resize to 256x256; no crop or padding',
        'images': records,
    }
    cfg['study_root'] = str(Path(cfg['study_root']).with_name(Path(cfg['study_root']).name + '_nominal_smoke'))
    cfg['models'] = {'dfot': model}
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='study.json')
    parser.add_argument('--output', type=Path, default=Path('study.nominal-smoke.json'))
    args = parser.parse_args()
    if args.output.resolve() == Path(args.config).resolve() or args.output.exists():
        parser.error('Choose a new output config; existing configs are preserved')
    cfg = prepare(args.config)
    save(args.output, cfg)
    print('NOMINAL INTRINSICS — engineering smoke only; not measured calibration')
    print(json.dumps(cfg['models']['dfot']['intrinsics_source']['images'][0], indent=2))
    print(f'Config: {args.output}; study root: {cfg["study_root"]}')


if __name__ == '__main__':
    main()
