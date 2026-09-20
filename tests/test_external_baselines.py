import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from external_baselines.camera import convert
from external_baselines.common import cohort_identity, generation_input, remap, resumable, identity, save, digest
from external_baselines.run import collect_bench, run


class Contracts(unittest.TestCase):
    def test_remap_preserves_extras_and_boundaries(self):
        rows = [dict(scene='s', start_frame=7, prompt='/old/a', input_image='/old/a/i.png',
                     pose_path='/older/p.json', extra={'keep': True})]
        moved = remap(rows, {'/old': '/new'})
        self.assertEqual(moved[0]['input_image'], '/new/a/i.png')
        self.assertEqual(moved[0]['pose_path'], '/older/p.json')
        self.assertEqual(moved[0]['prompt'], '/old/a')
        self.assertEqual(cohort_identity(rows), cohort_identity(moved))
        self.assertNotEqual(cohort_identity(rows), cohort_identity([dict(rows[0], start_frame=8)]))

    def test_camera_translation_and_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'poses.json'
            p.write_text(json.dumps({'CineCameraActor': {
                '0': {'position': [0, 0, 0], 'rotation': [0, 0, 0]},
                '1': {'position': [100, 0, 0], 'rotation': [0, 0, 0]},
                '2': {'position': [0, 0, 0], 'rotation': [0, 0, 90]}}}))
            v, keys = convert(p, 0, 3, [1, 1, .5, .5])
            np.testing.assert_allclose(v[0, 4:].reshape(3, 4), np.eye(4)[:3])
            np.testing.assert_allclose(v[1, 4:].reshape(3, 4)[:, 3], [0, 0, -1])
            # After 90-degree UE yaw, world-right (+OpenCV x) is camera-forward.
            np.testing.assert_allclose(v[2, 4:].reshape(3, 4)[:, :3] @ [1, 0, 0], [0, 0, 1], atol=1e-6)
            self.assertEqual(keys, ['0', '1', '2'])
            with self.assertRaises(ValueError):
                convert(p, 1, 3, [1, 1, .5, .5])

    def test_no_future_gt_in_generation_job(self):
        row = dict(input_image='initial', pose_path='poses', gt_frames_dir='secret',
                   overlap_dir='secret', future_images=['secret'], start_frame=10,
                   num_frames=73, fps=30, prompt='room')
        self.assertNotIn('secret', json.dumps(generation_input(row)))

    def test_resume_requires_content_config_count_and_decode(self):
        with tempfile.TemporaryDirectory() as d:
            video, receipt = Path(d) / 'v.mp4', Path(d) / 'v.json'
            video.write_bytes(b'test')
            save(receipt, {'signature': 'sig', 'video_sha256': digest(video)})
            with patch('external_baselines.common.probe', return_value={'frames': 73, 'fps': 30}):
                self.assertTrue(resumable(video, receipt, 'sig', 73))
                self.assertFalse(resumable(video, receipt, 'different', 73))
                self.assertFalse(resumable(video, receipt, 'sig', 5397))
            with patch('external_baselines.common.probe', side_effect=ValueError('decode failed')):
                self.assertFalse(resumable(video, receipt, 'sig', 73))

    def test_exact_metric_cohort_and_scaling(self):
        from external_baselines.common import DIMENSIONS
        rows = [{'output_prefix': str(i)} for i in range(15)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x_eval_results.json'
            payload = {dim: [0, [{'video_path': f'{i}custom.mp4',
                'video_results': 50 if dim == 'imaging_quality' else .5} for i in range(15)]] for dim in DIMENSIONS}
            save(p, payload)
            self.assertEqual(collect_bench(Path(d), rows)['imaging_quality'], .5)
            payload['subject_consistency'][1][-1]['video_path'] = '0custom.mp4'
            save(p, payload)
            with self.assertRaises(ValueError):
                collect_bench(Path(d), rows)

    def test_failure_propagation(self):
        import sys, os
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                run([sys.executable, '-c', 'raise SystemExit(7)'], Path(d) / 'log', d, os.environ)


if __name__ == '__main__':
    unittest.main()
