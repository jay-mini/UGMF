import unittest

import torch

from scripts.sample_class_grid import (
    build_balanced_inputs,
    compute_scale_metrics,
)


class ClassGridScriptTests(unittest.TestCase):
    def test_balanced_inputs_reuse_each_noise_column_across_classes(self):
        initial_noise, cond = build_balanced_inputs(
            classes=[2, 5, 9],
            samples_per_class=2,
            sample_shape=(1, 4, 4),
            seed=17,
        )

        noise_rows = initial_noise.reshape(3, 2, 1, 4, 4)
        self.assertTrue(torch.equal(noise_rows[0], noise_rows[1]))
        self.assertTrue(torch.equal(noise_rows[1], noise_rows[2]))
        self.assertTrue(torch.equal(cond, torch.tensor([2, 2, 5, 5, 9, 9])))

    def test_metrics_detect_scale_zero_label_invariance_and_cfg_shift(self):
        unconditional = torch.zeros(3, 2, 1, 2, 2)
        conditional = torch.stack(
            [
                torch.full((2, 1, 2, 2), value)
                for value in (0.0, 1.0, 2.0)
            ]
        )

        metrics = compute_scale_metrics({0.0: unconditional, 1.0: conditional})

        self.assertEqual(
            metrics["0"]["max_label_difference_for_fixed_noise"],
            0.0,
        )
        self.assertEqual(metrics["0"]["mean_absolute_label_spread"], 0.0)
        self.assertGreater(metrics["1"]["mean_absolute_label_spread"], 0.0)
        self.assertGreater(metrics["1"]["mean_absolute_shift_from_scale_0"], 0.0)


if __name__ == "__main__":
    unittest.main()
