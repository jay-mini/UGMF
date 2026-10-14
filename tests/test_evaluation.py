from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from scripts.evaluate import build_label_sequence, prepare_export_directory
from ugm.evaluation.fid import (
    calculate_pixel_fd_proxy,
    calculate_standard_metrics,
)
from ugm.evaluation.kid import resolve_kid_subset_size
from ugm.evaluation.precision_recall import validate_prc_settings
from ugm.evaluation.speed_quality import (
    CountingModel,
    benchmark_sampler,
    expected_nfe,
)


def save_constant_png(path: Path, value: int) -> None:
    image = Image.fromarray(
        np.full((8, 8), value, dtype=np.uint8),
        mode="L",
    )
    image.save(path)


class TinyModel(nn.Module):
    def forward(self, x, t, cond=None):
        del t, cond
        return x + 1.0


class EvaluationTests(unittest.TestCase):
    def test_pixel_fd_is_explicit_proxy_and_identical_sets_are_zero(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            generated = root / "generated"
            reference = root / "reference"
            generated.mkdir()
            reference.mkdir()
            for index, value in enumerate((0, 64, 128, 255)):
                save_constant_png(generated / f"{index}.png", value)
                save_constant_png(reference / f"{index}.png", value)

            result = calculate_pixel_fd_proxy(generated, reference)

            self.assertFalse(result["standard_metric"])
            self.assertAlmostEqual(result["value"], 0.0, places=8)
            self.assertIn("must not", result["interpretation"])

    def test_standard_metric_wrapper_preserves_backend_outputs(self):
        calls = []

        def fake_calculate(**kwargs):
            calls.append(kwargs)
            return {
                "frechet_inception_distance": 12.5,
                "kernel_inception_distance_mean": 0.02,
                "kernel_inception_distance_std": 0.003,
                "precision": 0.7,
                "recall": 0.6,
                "f_score": 0.646,
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            generated = root / "generated"
            reference = root / "reference"
            generated.mkdir()
            reference.mkdir()
            for index in range(4):
                save_constant_png(generated / f"{index}.png", index)
                save_constant_png(reference / f"{index}.png", index)

            result = calculate_standard_metrics(
                generated_dir=generated,
                reference_dir=reference,
                device=torch.device("cpu"),
                fid=True,
                kid=True,
                prc=True,
                batch_size=2,
                kid_subsets=3,
                kid_subset_size=1000,
                prc_neighborhood=3,
                prc_batch_size=2,
                seed=9,
                calculate_fn=fake_calculate,
            )

        self.assertEqual(result["fid"]["value"], 12.5)
        self.assertEqual(result["kid"]["subset_size_effective"], 4)
        self.assertEqual(
            result["generative_precision_recall"]["precision"],
            0.7,
        )
        self.assertTrue(calls[0]["fid"])
        self.assertTrue(calls[0]["kid"])
        self.assertTrue(calls[0]["prc"])
        self.assertFalse(calls[0]["cache"])

    def test_kid_and_prc_validation(self):
        self.assertEqual(
            resolve_kid_subset_size(
                requested=1000,
                generated_count=64,
                reference_count=80,
            ),
            64,
        )
        validate_prc_settings(
            neighborhood=3,
            generated_count=4,
            reference_count=5,
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "neighborhood"):
            validate_prc_settings(
                neighborhood=4,
                generated_count=4,
                reference_count=5,
                batch_size=2,
            )

    def test_nfe_accounts_for_solver_and_cfg_branches(self):
        self.assertEqual(
            expected_nfe(
                sampler_name="ddim",
                num_steps=50,
                has_condition=True,
                guidance_scale=3.0,
            ),
            100,
        )
        self.assertEqual(
            expected_nfe(
                sampler_name="heun",
                num_steps=20,
                has_condition=True,
                guidance_scale=1.0,
            ),
            40,
        )
        self.assertEqual(
            expected_nfe(
                sampler_name="euler",
                num_steps=20,
                has_condition=False,
                guidance_scale=5.0,
            ),
            20,
        )

    def test_benchmark_records_warmup_repetitions_and_forward_calls(self):
        model = CountingModel(TinyModel())
        x = torch.zeros(2, 1)
        t = torch.zeros(2)

        def run_once():
            return model(x, t)

        result = benchmark_sampler(
            run_once=run_once,
            counting_model=model,
            device=torch.device("cpu"),
            batch_size=2,
            warmup_runs=2,
            repetitions=3,
            expected_forward_calls=1,
        )

        self.assertEqual(result["warmup_runs"], 2)
        self.assertEqual(result["repetitions"], 3)
        self.assertEqual(
            result["observed_model_forward_calls_per_repeat"],
            [1, 1, 1],
        )
        self.assertTrue(result["nfe_verified"])
        self.assertFalse(result["cuda_memory"]["available"])

    def test_balanced_label_sequence_and_safe_export_directory(self):
        labels = build_label_sequence(
            config={"class_cond": True, "num_classes": 3},
            count=8,
            class_label=None,
            policy="balanced",
            seed=42,
        )
        self.assertTrue(
            torch.equal(labels, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            export_dir = Path(temporary_dir) / "samples"
            prepare_export_directory(export_dir, overwrite=False)
            save_constant_png(export_dir / "000000.png", 0)
            with self.assertRaises(FileExistsError):
                prepare_export_directory(export_dir, overwrite=False)
            prepare_export_directory(export_dir, overwrite=True)
            self.assertFalse(any(export_dir.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
