import unittest
from external_baselines.cameractrl_smoke import sample_indices


class CameraCtrlTiming(unittest.TestCase):
    def test_native_time_map(self):
        indices = sample_indices(30)
        self.assertEqual(len(indices), 25)
        self.assertEqual(indices[0], 0)
        self.assertEqual(len(set(indices)), 25)
        self.assertEqual(indices[-1], 103)
        for k, index in enumerate(indices):
            self.assertLessEqual(abs(index / 30 - k / 7), 1 / 60)

    def test_no_duplicate_pose_upsampling(self):
        with self.assertRaises(ValueError):
            sample_indices(5)


if __name__ == '__main__':
    unittest.main()
