import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective


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


if __name__ == "__main__":
    unittest.main()
