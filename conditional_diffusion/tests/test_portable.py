"""CPU checks using synthetic arrays only; no experiment data or GPU needed."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import read, write, sha
from dataset import Store, prepare, fixed_noise
from models import build, sample
from objectives import generative_loss
from sampler_objective import differentiable_sample, sampled_forecast_loss
from runtime import train, load_model, base_for, evaluate_model, selection

METHOD = read(Path(__file__).resolve().parents[1] / 'model.json')['method']


def tiny_cache(root):
    times = np.concatenate([np.datetime64(date) + np.arange(31) * np.timedelta64(6, 'm')
                            for date in ['2023-05-01', '2023-07-01', '2025-05-01', '2025-07-01']])
    rng = np.random.default_rng(123)
    np.save(root / 'times.npy', times)
    np.save(root / 'radar_u8.npy', rng.integers(0, 256, (124, 66, 70), dtype=np.uint8))
    np.save(root / 'pwv_u8.npy', rng.integers(0, 256, (124, 66, 70), dtype=np.uint8))
    splits = {'R1': {'train': [0], 'valid': [62], 'test': [93]}, 'R2': {'train': [62], 'valid': [0], 'test': [31]}}
    for rnd, parts in splits.items():
        for split, ids in parts.items(): np.save(root / f'{rnd}_{split}.npy', np.array(ids, dtype=np.int64))
        np.save(root / f'{rnd}_screen.npy', np.array(parts['valid'], dtype=np.int64))
    write(root / 'manifest.json', {'dataset_id': 'SYNTHETIC-TEST-ONLY'})

class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_noise_identity_independent_of_batching(self):
        a = fixed_noise([2, 9], 0); b = fixed_noise([9], 0)
        self.assertTrue(torch.equal(a[1], b[0])); self.assertFalse(torch.equal(a[0], fixed_noise([2], 1)[0]))

    def test_native_and_actual_sampler_gradients_and_parity(self):
        torch.manual_seed(31); m = build(METHOD); m.set_normalization(.04, .14)
        x = torch.rand(1, 20, 66, 70); y = torch.rand_like(x)
        base = build('deterministic').eval().requires_grad_(False) if METHOD == 'resdiff' else None
        bp = base(x).detach() if base is not None else None
        native = generative_loss(m, x, y, METHOD, 'skill', bp); native.backward()
        self.assertTrue(torch.isfinite(native)); self.assertGreater(m.out.weight.grad.abs().sum().item(), 0.)
        m.zero_grad(set_to_none=True); noise = fixed_noise([3], 0)
        expected = sample(m, x, noise, METHOD, bp, 16)
        actual = differentiable_sample(m, x, noise, METHOD, bp, 16, 4)
        self.assertTrue(torch.allclose(expected, actual.detach(), atol=1e-6, rtol=1e-6))
        del actual, expected, native
        import gc
        gc.collect()
        aux = sampled_forecast_loss(m, x, y, METHOD, .14, bp, 4, torch.Generator().manual_seed(9)); aux.backward()
        self.assertTrue(torch.isfinite(aux)); self.assertGreater(m.out.weight.grad.abs().sum().item(), 0.)

    def test_prepare_decode_gaps_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); black = root / 'black.png'; white = root / 'white.png'
            Image.fromarray(np.zeros((66, 70), dtype=np.uint8)).save(black)
            Image.fromarray(np.full((66, 70), 255, dtype=np.uint8)).save(white)
            csvfile = root / 'frames.csv'
            with csvfile.open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['time', 'radar_path', 'pwv_path'])
                for i in range(31): w.writerow([str(np.datetime64('2023-05-01') + i * np.timedelta64(6, 'm')), 'black.png', 'white.png'])
            counts = prepare(csvfile, root / 'cache'); self.assertEqual(counts['R1_train'], 2)
            self.assertEqual(np.load(root / 'cache/radar_u8.npy').min(), 255)
            self.assertEqual(np.load(root / 'cache/pwv_u8.npy').max(), 0)
            with self.assertRaises(FileExistsError): prepare(csvfile, root / 'cache')

    def test_split_normalization_training_resume_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root / 'cache'; cache.mkdir(); tiny_cache(cache)
            store = Store(cache); normal = store.normalization('R1')
            before = np.load(cache / 'radar_u8.npy').copy(); before[62:] = 0; np.save(cache / 'radar_u8.npy', before)
            self.assertEqual(normal, Store(cache).normalization('R1'))
            # Restore data; one synthetic epoch tests the portable I/O loop only.
            tiny_cache(cache); output = root / 'run'
            cfg = {'method': METHOD, 'stage': 'native', 'round': 'R1', 'seed': 20261019, 'epochs': 1,
                'recipe': 'standard', 'learning_rate': .0003, 'eta_min': .00001, 'weight_decay': .0001,
                'batch_size': 1, 'bf16': False, 'validation_every': 1, 'keep_candidates': 1, 'include_final': True,
                'minimum_epochs': 1, 'patience': 0, 'minimum_free_gib': 0., 'output_budget_gib': 1.}
            base_path = None
            if METHOD == 'resdiff':
                base_path = root / 'base.pt'; torch.save({'model': build('deterministic').state_dict(), 'method': 'deterministic'}, base_path)
            result = train(cfg, cache, output, 'cpu', base_path=base_path)
            self.assertFalse(result['test_used_for_selection']); self.assertEqual(result['trained_epochs'], 1)
            m, ck = load_model(output / 'selected.pt', METHOD, 'cpu'); base = base_for(METHOD, ck, base_path, 'cpu')
            records, means = evaluate_model(m, base, Store(cache), Store(cache).starts('R1', 'valid'), METHOD, 2, 1)
            self.assertEqual(len(records), 120); self.assertEqual(len(means), 60); self.assertTrue(np.isfinite(selection(means)['RMSE_dbz']))
            again = train(cfg, cache, output, 'cpu', base_path=base_path, resume=True)
            self.assertEqual(result['selected_epoch'], again['selected_epoch'])
            with self.assertRaises(FileExistsError): train(cfg, cache, output, 'cpu', base_path=base_path)
            if METHOD == 'resdiff':
                wrong = root / 'wrong.pt'; torch.save({'model': build('deterministic').state_dict()}, wrong)
                with self.assertRaises(ValueError): base_for(METHOD, ck, wrong, 'cpu')
            np.save(cache / 'R1_valid.npy', np.array([60], dtype=np.int64))
            with self.assertRaises(ValueError): Store(cache).starts('R1', 'valid')

if __name__ == '__main__': unittest.main()
