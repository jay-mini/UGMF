import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from ugm.objectives.ddpm import DDPMobjective
from ugm.samplers.ddpm import DDPMAncestralSampler

class ZeroVelocityModel(nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)
    

class ObjectiveTests(unittest.TestCase):
    def test_ddpm_sample_shape(self):
        objective = DDPMobjective(num_timesteps=10, )

        sampler = DDPMAncestralSampler()
        model = ZeroVelocityModel()

        xt = torch.randn(2, 1, 8, 8)
        t = torch.full(
            (2,),
            fill_value=5,
            dtype=torch.long,
        )
        cond = torch.tensor([0, 1])

        x_prev = sampler.sample(model, objective, shape=xt.shape, cond=cond)

        self.assertEqual(x_prev.shape, xt.shape)
        self.assertTrue(torch.isfinite(x_prev).all())

    def test_ddpm_p_sample_shape(self):
        objective = DDPMobjective(num_timesteps=10, )

        sampler = DDPMAncestralSampler()
        model = ZeroVelocityModel()

        xt = torch.randn(2, 1, 8, 8)
        t = torch.full(
            (2,),
            fill_value=5,
            dtype=torch.long,
        )
        cond = torch.tensor([0, 1])

        x_prev = sampler.p_sample(model, objective, xt, t, cond=cond)

        self.assertEqual(x_prev.shape, xt.shape)
        self.assertTrue(torch.isfinite(x_prev).all())


    def test_no_noise_added_at_t0(self):
        objective = DDPMobjective(num_timesteps=10, )

        sampler = DDPMAncestralSampler()
        model = ZeroVelocityModel()

        xt = torch.randn(2, 1, 8, 8)
        t = torch.full(
            (2,),
            fill_value=0,
            dtype=torch.long,
        )
        cond = torch.tensor([0, 1])

        torch.manual_seed(42)  # Set a seed for reproducibility
        x_prev = sampler.p_sample(model, objective, xt, t, cond=cond)

        torch.manual_seed(999)
        x_prev_again = sampler.p_sample(model, objective, xt, t, cond=cond)

        self.assertTrue(torch.allclose(x_prev, x_prev_again), "Noise was added at t=0, but it should not be.")


    def test_ddpm_sampler_smoke(self):
        objective = DDPMobjective(num_timesteps=5, )

        sampler = DDPMAncestralSampler()
        model = ZeroVelocityModel()

        shape = (2, 1, 8, 8)
        cond = torch.tensor([0, 1])

        x_samples = sampler.sample(model, objective, shape=shape, cond=cond)

        self.assertEqual(x_samples.shape, shape)
        self.assertTrue(torch.isfinite(x_samples).all())

if __name__ == "__main__":
    unittest.main()

