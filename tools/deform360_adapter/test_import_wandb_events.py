"""CPU tests; stdlib only, no SDK, network, or CUDA."""
import json
from pathlib import Path
import tempfile
import unittest

from import_wandb_events import convert


class ImportTests(unittest.TestCase):
    def test_validation_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'events.jsonl'
            events = []
            for it in (1, 2):
                events.extend([
                    dict(event='iteration_start', iteration=it, lr=[0.000404]),
                    dict(event='optimizer_step_complete', iteration=it, epoch=1,
                         seconds=it, grad_norm=1., rollout=3,
                         losses={'loss': 6., 'decode.loss_mse_momentum_c': 1.,
                                 'decode.loss_l2_render_c': 2.,
                                 'decode.loss_ssim_render_c': 3.})])
            def write():
                p.write_text(''.join(json.dumps(e) + '\n' for e in events))
            write()
            rows, report = convert([p], 1, 1, 2)
            self.assertEqual(report['step_count'], 2)
            self.assertEqual(rows[1]['optimization/lr'], .000404)
            self.assertEqual(rows[1]['train/loss_ssim'], 3.)
            with self.assertRaises(ValueError):
                convert([p, p], 1, 1, 2)
            with self.assertRaises(ValueError):
                convert([p], 1, 1, 3)
            events[-1]['losses']['loss'] = float('nan')
            write()
            with self.assertRaises(ValueError):
                convert([p], 1, 1, 2)

    def test_stage2_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'events.jsonl'
            events = [dict(event='cache_provenance', cache_key='frame_70',
                           sha256='fixture', source_window=[183, 195]),
                      dict(event='iteration_start', iteration=13201, lr=[.000404]),
                      dict(event='optimizer_step_complete', iteration=13201,
                           epoch=23, seconds=1., preclip_norm=2., postclip_norm=1.,
                           requested_rollout=69, effective_rollout=12,
                           cache_key='frame_70', window_start=70, source_start=183,
                           target_source=195, losses={'loss': 6.,
                           'decode.loss_mse_momentum_c': 1.,
                           'decode.loss_l2_render_c': 2., 'decode.loss_ssim_render_c': 3.})]
            p.write_text(''.join(json.dumps(e) + '\n' for e in events) + '{partial')
            rows, report = convert([p], 2, 13201, 13201)
            self.assertEqual(rows[0]['stage2/window_local_end_inclusive'], 82)
            self.assertEqual(rows[0]['stage2/dense_target_source_end'], 195)
            self.assertLess(report['sources'][0]['snapshot_bytes'], p.stat().st_size)


if __name__ == '__main__':
    unittest.main()
