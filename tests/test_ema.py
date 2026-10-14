import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugm.objectives.ddpm import DDPMobjective
from ugm.models.unet import SimpleUNet
from ugm.utils.ema import EMA

class ZeroVelocityModel(nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)
    
class EMATests(unittest.TestCase):
    def test_ema_initially_matches_model(self):
        model = ZeroVelocityModel()
        ema = EMA(model)

        for model_parameter, ema_parameter in zip(
            model.parameters(),
            ema.ema_model.parameters(),
        ):
            self.assertEqual(model_parameter, ema_parameter)

    def test_ema_parameters_do_not_require_grad(self):
        model = ZeroVelocityModel()
        ema = EMA(model)

        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in ema.ema_model.parameters()
            )
        )

    def test_ema_update(self):
        model = torch.nn.Linear(
            1,
            1,
            bias=False,
        )

        with torch.no_grad():
            model.weight.fill_(1.0)

        ema = EMA(
            model,
            decay=0.5,
        )

        with torch.no_grad():
            model.weight.fill_(3.0)

        ema.update(
            model=model,
            global_step=1,
        )

        expected = torch.tensor([[2.0]])

        self.assertTrue(
            torch.allclose(
                ema.ema_model.weight,
                expected,
                rtol=1e-5,
                atol=1e-8,
            ),
            msg=(
                "EMA weight does not match the expected value.\n"
                f"Actual:\n{ema.ema_model.weight}\n"
                f"Expected:\n{expected}"
            ),
        )

if __name__ == '__main__':
    unittest.main()
