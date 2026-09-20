"""Run upstream rollout control flow with NumPy tensors and a stub sampler."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


class Tensor(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def to(self, *_):
        return self

    def long(self):
        return self.astype(np.int64).view(Tensor)


def tensor(value):
    return np.asarray(value).view(Tensor)


class Progress:
    def __init__(self, **kwargs):
        pass

    def close(self):
        pass


def make_model(causal=False):
    source = Path(__file__).resolve().parents[1] / 'algorithms/dfot/dfot_video.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DFoTVideo')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_predict_sequence', '_pad_to_max_tokens')]
    cls.body, cls.bases = methods, []
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef):
            node.returns = None
            for arg in node.args.args + node.args.kwonlyargs:
                arg.annotation = None
        if isinstance(node, ast.AnnAssign):
            node.annotation = ast.Name(id='object', ctx=ast.Load())
    torch = SimpleNamespace(zeros=lambda shape, **kw: tensor(np.zeros(shape)),
                            ones=lambda shape, **kw: tensor(np.ones(shape)),
                            cat=lambda items, dim: tensor(np.concatenate(items, axis=dim)), long=np.int64)
    ns = dict(torch=torch, tqdm=Progress,
              repeat=lambda x, pattern, t: tensor(np.repeat(x, t, axis=1)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), ns)
    model = ns['DFoTVideo']()
    model.max_tokens, model.chunk_size, model.x_shape = 8, -1, [1]
    model.sampling_timesteps, model.device, model.use_causal_mask = 50, 'cpu', causal
    model.calls = []
    def sample(batch_size, length, context, conditions, **kwargs):
        if conditions is not None:
            assert conditions.shape[1] == (length if causal else 8)
        model.calls.append(conditions)
        return context, None
    model._sample_sequence = sample
    return model


class PredictionPadding(unittest.TestCase):
    def rollout(self, length, causal=False, conditioned=True):
        model = make_model(causal)
        conditions = tensor(np.arange(length).reshape(1, length, 1)) if conditioned else None
        result, _ = model._predict_sequence(tensor([[[99.]]]), length=length,
                                            conditions=conditions, sliding_context_len=4)
        self.assertEqual(result.shape, (1, length, 1))
        self.assertEqual(result[0, 0, 0], 99)
        return model.calls

    def test_five_keyframes_pad_to_eight(self):
        calls = self.rollout(5)
        np.testing.assert_array_equal(calls[0][0, :, 0], [0, 1, 2, 3, 4, 4, 4, 4])

    def test_final_rollout_window(self):
        calls = self.rollout(13)
        np.testing.assert_array_equal(calls[-1][0, :, 0], [8, 9, 10, 11, 12, 12, 12, 12])

    def test_full_window_unchanged(self):
        np.testing.assert_array_equal(self.rollout(8)[0][0, :, 0], np.arange(8))

    def test_causal_and_unconditioned(self):
        self.assertEqual(self.rollout(5, causal=True)[0].shape[1], 5)
        self.assertEqual(self.rollout(5, conditioned=False), [None])


if __name__ == '__main__':
    unittest.main()
