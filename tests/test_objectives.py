import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.utils.tensor import extract


class ZeroVelocityModel(nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)


class ObjectiveTests(unittest.TestCase):
    def test_flow_matching_compute_loss_returns_scalar_metrics(self):
        objective = FlowMatchingObjective()
        model = ZeroVelocityModel()
        batch = {
            "x": torch.zeros(2, 1, 4, 4),
            "cond": torch.tensor([0, 1]),
        }

        output = objective.compute_loss(model, batch)

        self.assertEqual(output.loss.ndim, 0)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIn("loss_fm", output.metrics)

    def test_ddpm_compute_loss_returns_scalar_metrics(self):
        objective = DDPMobjective(num_timesteps=10)
        model = ZeroVelocityModel()
        batch = {
            "x": torch.zeros(2, 1, 4, 4),
            "cond": torch.tensor([0, 1]),
        }

        output = objective.compute_loss(model, batch)

        self.assertEqual(output.loss.ndim, 0)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIn("loss_ddpm", output.metrics)

    def test_extract_shape(self):
        values = torch.linspace(0, 1, steps=10)
        timesteps = torch.tensor(
            [0, 3, 9],
            dtype=torch.long,
        )

        result = extract(
            values,
            timesteps,
            (3, 1, 28, 28),
        )

        self.assertEqual(result.shape, (3, 1, 1, 1))

    def test_ddpm_schedule_properties(self):
        objective = DDPMobjective(num_timesteps=100,)

        self.assertEqual(objective.betas.shape, (100,))
        self.assertEqual(objective.alphas.shape, (100,))
        self.assertEqual(objective.alpha_bars.shape, (100,))

        self.assertTrue(torch.all(objective.alphas > 0))
        self.assertTrue(torch.all(objective.betas < 1))

        self.assertTrue(torch.all(objective.alpha_bars[1:] <= objective.alpha_bars[:-1]))


    def test_ddpm_q_sample_shape(self):
        objective = DDPMobjective(num_timesteps=100, )

        x0 = torch.zeros(2, 1, 28, 28)
        timesteps = torch.randint(
            0,
            100,
            size=(2,),
            dtype=torch.long,
        )
        xt = objective.q_sample(x0, timesteps, noise=None)

        self.assertEqual(xt.shape, x0.shape)
        self.assertTrue(torch.isfinite(xt).all())


    def test_ddpm_predict_x0_from_noise_roundtrip(self):
        objective = DDPMobjective(num_timesteps=100, )

        x0 = torch.randn(2, 1, 28, 28)
        noise = torch.randn_like(x0)
        timesteps = torch.tensor([10, 50], dtype=torch.long)
        xt = objective.q_sample(x0, timesteps, noise=noise)

        noise_pred = noise
        x0_pred = objective.predict_x0_from_noise(xt, timesteps, noise_pred)

        self.assertTrue(torch.allclose(x0, x0_pred, atol=1e-5, rtol=1e-5))

if __name__ == "__main__":
    unittest.main()
