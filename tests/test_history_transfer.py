from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from external_baselines.history_selection import (
    camera_to_c2w, choose_history, conditioning_indices, load_scorer, validate_selection,
)
from external_baselines.native_history_transfer import (
    configuration, export, freeze, predict, prepare_cases, save, verify_receipt,
)
from external_baselines.common import digest


def cameras(count):
    result = np.zeros((count, 16), dtype=np.float32)
    result[:, :4] = [1, 1, .5, .5]
    for i in range(count):
        w2c = np.eye(4)[:3]
        w2c[0, 3] = -i * .1
        result[i, 4:] = w2c.flatten()
    return result


class HistorySelectionTests(unittest.TestCase):
    def test_camera_inverse_and_validation(self):
        pose = camera_to_c2w(cameras(16))
        np.testing.assert_allclose(pose[:, 0, 3], np.arange(16) * .1, atol=1e-7)
        for bad in (np.zeros((4, 12)), np.full((4, 16), np.nan), np.zeros((4, 16))):
            with self.assertRaises(ValueError):
                camera_to_c2w(bad)
        reflected = cameras(4)
        reflected[:, 4] = -1
        with self.assertRaises(ValueError):
            camera_to_c2w(reflected)

    def test_baseline_history_and_invalid_ids(self):
        self.assertEqual(choose_history("fifo", cameras(16), None, 4, None)[0], [12, 13, 14, 15])
        self.assertEqual(choose_history("uniform", cameras(16), None, 4, None)[0], [0, 5, 10, 15])
        self.assertEqual(conditioning_indices([0, 4, 9, 15], 16, 4, 4), [0, 4, 9, 15, 16, 17, 18, 19])
        for ids in ([0, 1, 2, 16], [0, 0, 1, 2], [2, 1, 0, 3], [0, 1, 2], [0., 1, 2, 3]):
            with self.assertRaises(ValueError):
                validate_selection(ids, 16, 4)

    def test_streaming_causal_bank_latest_protection_and_ties(self):
        calls = []
        def scorer(ids, c2ws, dino_features):
            self.assertEqual(len(c2ws), max(ids) + 1)
            self.assertEqual(set(dino_features), set(ids))
            self.assertEqual(len(ids), 5)
            calls.append(ids)
            return {i: 1. for i in ids}
        features = np.eye(16)
        chosen, trace = choose_history("keepsake", cameras(16), features, 4, scorer)
        self.assertEqual(chosen, [12, 13, 14, 15])
        self.assertEqual(len(calls), 12)
        self.assertEqual(trace[4]["evicted"], [0])
        for update in trace:
            self.assertIn(update["incoming"], update["retained"])
            self.assertLessEqual(len(update["retained"]), 4)
        with self.assertRaises(ValueError):
            choose_history("keepsake", cameras(16), features * 2, 4, scorer)

    def test_production_formula_used_without_reimplementation(self):
        repo = Path(__file__).resolve().parents[2] / "MemCam"
        if not repo.exists():
            self.skipTest("Sibling MemCam checkout needed for production parity")
        scorer, _ = load_scorer(repo)
        rng = np.random.default_rng(23)
        features = rng.normal(size=(16, 24))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        chosen, trace = choose_history("keepsake", cameras(16), features, 4, scorer)
        for update in trace[4:]:
            ids = update["candidates"]
            direct = scorer(ids, camera_to_c2w(cameras(16))[:update["incoming"] + 1],
                            dino_features={i: features[i] for i in ids})
            self.assertEqual(update["scores"], direct)
        self.assertEqual(len(chosen), 4)


class ArtifactsTests(unittest.TestCase):
    def test_native_configuration_composes(self):
        try:
            from omegaconf import OmegaConf
            import hydra
        except ImportError:
            self.skipTest("Hydra required for native configuration integration test")
        cfg = configuration(SimpleNamespace(scenes=5, data=Path("/tmp/mini")))
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        self.assertEqual(cfg.dataset.n_frames, 20)
        self.assertEqual(cfg.dataset.context_length, 4)
        self.assertEqual(cfg.algorithm.tasks.prediction.history_guidance.guidance_scale, 2.0)
        self.assertFalse(cfg.algorithm.tasks.prediction.history_guidance.visualize)
        self.assertFalse(cfg.algorithm.tasks.interpolation.enabled)

    def test_frozen_configs_and_receipts_reject_changed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze(root / "config.json", {"seed": 1})
            freeze(root / "config.json", {"seed": 1})
            with self.assertRaises(ValueError):
                freeze(root / "config.json", {"seed": 2})
            prediction = root / "prediction.npy"
            np.save(prediction, [1])
            save(root / "generation.json", dict(signature="same", artifacts={prediction.name: digest(prediction)}))
            self.assertIsNotNone(verify_receipt(root, "generation", "same"))
            with self.assertRaises(ValueError):
                verify_receipt(root, "generation", "other")
            np.save(prediction, [2])
            with self.assertRaises(ValueError):
                verify_receipt(root, "generation", "same")

    def test_complete_table_requires_all_methods_and_equal_clip_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [{"scene": "a"}, {"scene": "b"}]
            results = {(scene, policy): [dict(psnr_db=v, ssim=v, lpips=v)] * 4
                       for scene, v in (("a", 1.), ("b", 3.)) for policy in ("fifo", "uniform", "keepsake")}
            export(root, cases, results, True)
            import csv
            with (root / "scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[0]["lpips"]), 2.)
            del results["b", "fifo"]
            with self.assertRaises(ValueError):
                export(root, cases, results, True)

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save(Path(tmp) / "bad.json", {"score": float("nan")})

    def test_real_adapter_prepares_native_frame_mapping_and_detects_source_changes(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test_poses").mkdir()
            video = root / "test.mp4"
            video.write_bytes(b"fixture")
            torch.save(torch.zeros(200, 18), root / "test_poses/test.pt")
            class Dataset:
                save_dir = root
                metadata = [{"video_paths": video}]
                def __len__(self):
                    return 1
                def get_clip_location(self, i):
                    return 0, 3
                def video_path_to_preprocessed_path(self, path):
                    return path
                def __getitem__(self, i):
                    return dict(videos=torch.zeros(20, 3, 256, 256), conds=torch.as_tensor(cameras(20)),
                                nonterminal=torch.ones(20, dtype=torch.bool))
            cases = prepare_cases(Dataset(), root / "result", 1)
            self.assertEqual(cases[0]["frame_indices"], list(range(3, 194, 10)))
            self.assertEqual(prepare_cases(Dataset(), root / "result", 1), cases)
            video.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                prepare_cases(Dataset(), root / "result", 1)

    def test_real_predict_adapter_never_receives_target_pixels_and_reorders_poses(self):
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
        history = np.stack([np.full((3, 256, 256), i / 16, dtype=np.float32) for i in range(16)])
        selection = [0, 5, 10, 15]
        with patch("torch.cuda.manual_seed_all"):
            actual = predict(model, history, cameras(20)[:16], cameras(20)[16:], selection, 7)
        np.testing.assert_array_equal(actual, np.zeros((4, 3, 256, 256)))
        np.testing.assert_array_equal(model.x[0, :4].numpy(), history[selection])
        np.testing.assert_array_equal(model.cond[0].numpy(), cameras(20)[selection + [16, 17, 18, 19]])

    def test_slurm_gpu_mask_is_preserved(self):
        from external_baselines.environment_smoke import main
        with patch.dict("os.environ", {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "GPU-allowed"}):
            with patch("sys.argv", ["smoke", "--gpu", "0"]), self.assertRaises(SystemExit):
                main()
            import os
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-allowed")


if __name__ == "__main__":
    unittest.main()
