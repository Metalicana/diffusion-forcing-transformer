import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from external_baselines.history_stress import (
    REGIMES, cache_case, export_regimes, find_revisit, fixed_plan, pose_differences,
    prepare_stress_cases,
)
from external_baselines.native_history_transfer import predict
from test_history_transfer import cameras


def returning_cameras():
    cond = cameras(200)
    position = np.zeros(200)
    position[30:120] = 10.
    position[120:142] = np.linspace(10, 0, 22)
    cond[:, 7] = -position
    return cond


class StressPlans(unittest.TestCase):
    def test_fixed_plans_share_targets_and_only_change_history_or_gap(self):
        small, large, gap = [fixed_plan(r, 7) for r in REGIMES[:3]]
        self.assertEqual(small["targets"], large["targets"])
        self.assertEqual(small["history"], large["history"][-16:])
        self.assertEqual(large["history"], gap["history"])
        self.assertEqual(large["targets"][0] - large["history"][-1], 2)
        self.assertEqual(gap["targets"][0] - gap["history"][-1], 32)
        self.assertEqual(len(large["history"]), 64)
        self.assertLess(gap["targets"][-1], 7 + 191)

    def test_pose_revisit_requires_match_excursion_and_four_unseen_targets(self):
        plan = find_revisit(returning_cameras())
        self.assertIsNotNone(plan)
        self.assertEqual(plan["targets"], [142, 144, 146, 148])
        self.assertLess(plan["history"][-1], plan["targets"][0])
        self.assertGreaterEqual(plan["targets"][0] - max(plan["matched_history"]), 64)
        self.assertTrue(all(a <= 5 for a in plan["match_rotation_deg"]))
        self.assertGreater(plan["excursion_position"], 0)
        self.assertIsNone(find_revisit(cameras(200)))
        stationary = cameras(200)
        stationary[:, 7] = 0
        self.assertIsNone(find_revisit(stationary))
        self.assertIsNone(find_revisit(returning_cameras()[:148]))
        only_first_target_matches = returning_cameras()[:149]
        only_first_target_matches[144:, 7] = -100
        self.assertIsNone(find_revisit(only_first_target_matches))

    def test_rotation_only_return_supported_and_large_rotation_rejected(self):
        cond = cameras(200)
        cond[:, 7] = 0
        angle = np.zeros(200)
        angle[30:142] = np.pi / 2
        for i, a in enumerate(angle):
            rot = np.array([[np.cos(a), 0, np.sin(a), 0], [0, 1, 0, 0], [-np.sin(a), 0, np.cos(a), 0]])
            cond[i, 4:] = rot.flatten()
        self.assertIsNotNone(find_revisit(cond))
        cond[142:] = cond[100]
        self.assertIsNone(find_revisit(cond))

    def test_camera_distance_and_angle_have_expected_units(self):
        a = np.eye(4)[None]
        b = a.copy()
        b[0, 0, 3] = 2
        b[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        dist, angle = pose_differences(a, b)
        self.assertAlmostEqual(dist.item(), 2.)
        self.assertAlmostEqual(angle.item(), 90.)

    def test_predict_64_history_never_passes_target_pixels(self):
        import torch
        class Model:
            device = torch.device("cpu")
            def _normalize_x(self, x):
                return x
            def _unnormalize_x(self, x):
                return x
            def _predict_videos(self, x, cond):
                self.x, self.cond = x.clone(), cond.clone()
                return x
        model = Model()
        images = np.stack([np.full((3, 256, 256), i / 64, np.float32) for i in range(64)])
        ids = [0, 21, 42, 63]
        predict(model, images, cameras(68)[:64], cameras(68)[64:], ids, 7)
        np.testing.assert_array_equal(model.x[0, :4].numpy(), images[ids])
        self.assertEqual(float(model.x[0, 4:].abs().sum()), 0)
        np.testing.assert_array_equal(model.cond[0].numpy(), cameras(68)[ids + [64, 65, 66, 67]])

    def test_real_cache_adapter_exact_source_indices_and_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_poses").mkdir()
            source = root / "clip.mp4"
            source.write_bytes(b"video")
            (root / "test_poses/clip.pt").write_bytes(b"poses")
            class Dataset:
                save_dir = root
                def video_path_to_preprocessed_path(self, p):
                    return p
                def video_length(self, m):
                    return 200
                def load_video_and_cond(self, metadata, start, end):
                    frames = torch.arange(start, end).float().view(-1, 1, 1, 1).expand(-1, 3, 256, 256) / 200
                    return frames, torch.as_tensor(cameras(200)[start:end])
                def transform(self, x):
                    return x
                def _process_external_cond(self, raw, frame_skip):
                    return raw[::frame_skip]
            plan = fixed_plan("gap32_h64", 0)
            case = cache_case(Dataset(), dict(video_paths=source), plan, root / "out", "gap32_h64", 0)
            with np.load(Path(case["directory"]) / "observations.npz") as saved:
                np.testing.assert_allclose(saved["images"][:, 0, 0, 0], np.array(plan["history"] + plan["targets"]) / 200)
                self.assertEqual(saved["conditions"].shape, (68, 16))
            self.assertEqual(cache_case(Dataset(), dict(video_paths=source), plan, root / "out", "gap32_h64", 0), case)
            with self.assertRaises(ValueError):
                cache_case(Dataset(), dict(video_paths=source), dict(history=plan["history"], targets=[0, 2, 4, 6]), root, "invalid", 0)

    def test_regimes_not_pooled_and_absent_revisits_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [dict(scene=f"{r}_{i}", regime=r, history_count=16 if r == REGIMES[0] else 64)
                     for r in REGIMES[:3] for i in range(2)]
            data = {(c["scene"], p): [dict(psnr_db=20., ssim=.8, lpips=.1)] * 4
                    for c in cases for p in ("fifo", "uniform", "keepsake")}
            export_regimes(root, cases, data, True)
            with (root / "scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(r["videos"] == "2" for r in rows))
            with (root / "regime_status.csv").open() as handle:
                coverage = list(csv.DictReader(handle))
            self.assertEqual(coverage[-1]["status"], "unavailable")
            del data[(cases[0]["scene"], "fifo")]
            with self.assertRaises(ValueError):
                export_regimes(root, cases, data, True)

    def test_pose_audit_and_fixed_cohort_are_not_output_selected(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_poses").mkdir()
            previous = []
            metadata = []
            for i in range(5):
                previous.append(dict(scene=f"clip{i}", start_frame=0))
                metadata.append(dict(video_paths=Path(f"clip{i}.mp4")))
                (root / f"test_poses/clip{i}.pt").write_bytes(str(i).encode())
            (root / "cohort.json").write_text(json.dumps(previous))
            class Dataset:
                save_dir = root
                def __len__(self):
                    return 5
                def get_clip_location(self, i):
                    return i, 0
                def video_length(self, m):
                    return 200
                def load_cond(self, m, start, end):
                    return torch.as_tensor(returning_cameras())
                def _process_external_cond(self, raw, frame_skip):
                    return raw
            ds = Dataset()
            ds.metadata = metadata
            args = SimpleNamespace(scenes=5, revisit_scenes=2, seed=2026, pilot_root=root, output=root / "out")
            def fake_cache(dataset, meta, plan, output, regime, index):
                return dict(scene=Path(meta["video_paths"]).stem, regime=regime, plan=plan)
            with patch("external_baselines.history_stress.cache_case", side_effect=fake_cache):
                cases = prepare_stress_cases(ds, args)
            self.assertEqual(len(cases), 17)
            coverage = json.loads((args.output / "coverage.json").read_text())
            self.assertEqual(coverage["eligible_revisit_clips"], 5)
            self.assertEqual(coverage["selected_revisit_clips"], 2)
            self.assertEqual(len([c for c in cases if c["regime"] == REGIMES[3]]), 2)


if __name__ == "__main__":
    unittest.main()
