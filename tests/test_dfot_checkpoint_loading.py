"""Exercise the real upstream checkpoint hook without allocating GPU weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from external_baselines.worker import load_dfot_checkpoint


class Parent:
    def on_load_checkpoint(self, checkpoint):
        pass


def lightweight_model():
    source = Path(__file__).resolve().parents[1] / 'algorithms/dfot/dfot_video.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DFoTVideo')
    names = {'_compile_checkpoint', '_should_include_in_checkpoint', 'on_load_checkpoint'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = [ast.Name(id='Parent', ctx=ast.Load())]
    namespace = dict(Parent=Parent, Dict=dict, Any=object, rank_zero_print=lambda *a: None, cyan=lambda x: x)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), namespace)
    model = namespace['DFoTVideo']()
    model.cfg = SimpleNamespace(compile=False, checkpoint=SimpleNamespace(strict=True, reset_optimizer=False))
    model.state_dict = lambda: {'data_mean': 'configured mean', 'data_std': 'configured std',
                                'diffusion_model.model.weight': 'initial weight'}
    def load(state, strict):
        if not strict or set(state) != set(model.state_dict()):
            raise AssertionError('Final load must remain strict and complete')
        model.loaded = state
    model.load_state_dict = load
    return model


class CheckpointLoading(unittest.TestCase):
    def test_omitted_buffers_restored_by_upstream_hook(self):
        model = lightweight_model()
        load_dfot_checkpoint(model, {'state_dict': {'diffusion_model._orig_mod.model.weight': 'trained'}})
        self.assertEqual(model.loaded['data_mean'], 'configured mean')
        self.assertEqual(model.loaded['data_std'], 'configured std')
        self.assertEqual(model.loaded['diffusion_model.model.weight'], 'trained')

    def test_missing_learned_weight_still_fails(self):
        with self.assertRaises(ValueError):
            load_dfot_checkpoint(lightweight_model(), {'state_dict': {}})

    def test_unexpected_learned_weight_still_fails(self):
        with self.assertRaises(ValueError):
            load_dfot_checkpoint(lightweight_model(), {'state_dict': {'diffusion_model.model.extra': 'bad'}})


if __name__ == '__main__':
    unittest.main()
