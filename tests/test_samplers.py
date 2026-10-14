import sys
import unittest
import inspect
import math
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from ugm.objectives.ddpm import DDPMobjective
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.samplers.ddim import DDIMSampler
from ugm.samplers.base import SamplingResult
from ugm.samplers.ode import ODESampler

class ConstantPredictionModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x, t, cond=None):
        return torch.full_like(x, self.value)

class ZeroVelocityModel(nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)


class ConstantVelocityModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x, t, cond=None):
        return torch.full_like(x, self.value)


class LinearVelocityModel(nn.Module):
    """The analytic ODE dx/dt = x has solution x(t) = exp(t) x(0)."""

    def forward(self, x, t, cond=None):
        return x
    

class ObjectiveTests(unittest.TestCase):
    def test_all_samplers_expose_the_same_sample_parameters(self):
        expected = inspect.signature(ODESampler.sample)

        self.assertEqual(
            inspect.signature(DDPMAncestralSampler.sample),
            expected,
        )
        self.assertEqual(
            inspect.signature(DDIMSampler.sample),
            expected,
        )

    def test_euler_and_heun_share_fixed_noise_and_solve_constant_field(self):
        shape = (2, 3)
        initial_noise = torch.arange(6, dtype=torch.float64).reshape(shape)
        original_noise = initial_noise.clone()
        model = ConstantVelocityModel(value=2.0)

        results = []
        for solver in ("euler", "heun"):
            result = ODESampler(num_steps=2, solver=solver).sample(
                model=model,
                objective=None,
                shape=shape,
                device="cpu",
                initial_noise=initial_noise,
                num_steps=4,
                return_trajectory=True,
            )

            self.assertIsInstance(result, SamplingResult)
            self.assertEqual(result.trajectory.shape, (5, *shape))
            self.assertTrue(torch.equal(result.trajectory[0], initial_noise))
            self.assertTrue(
                torch.allclose(
                    result.timesteps,
                    torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
                )
            )
            self.assertTrue(
                torch.allclose(result.samples, initial_noise + 2.0)
            )
            results.append(result.samples)

        self.assertTrue(torch.allclose(results[0], results[1]))
        self.assertTrue(torch.equal(initial_noise, original_noise))

    def test_heun_is_more_accurate_for_analytic_linear_ode(self):
        shape = (2, 1)
        initial_noise = torch.ones(shape, dtype=torch.float64)
        num_steps = 10
        step_size = 1.0 / num_steps
        model = LinearVelocityModel()

        euler = ODESampler(num_steps=num_steps, solver="euler").sample(
            model=model,
            objective=None,
            shape=shape,
            device="cpu",
            initial_noise=initial_noise,
        )
        heun = ODESampler(num_steps=num_steps, solver="heun").sample(
            model=model,
            objective=None,
            shape=shape,
            device="cpu",
            initial_noise=initial_noise,
        )

        expected_euler = (1.0 + step_size) ** num_steps
        expected_heun = (
            1.0 + step_size + 0.5 * step_size**2
        ) ** num_steps
        analytic_solution = torch.full_like(initial_noise, math.e)

        self.assertTrue(
            torch.allclose(
                euler,
                torch.full_like(initial_noise, expected_euler),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                heun,
                torch.full_like(initial_noise, expected_heun),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        euler_error = torch.max(torch.abs(euler - analytic_solution))
        heun_error = torch.max(torch.abs(heun - analytic_solution))
        self.assertLess(heun_error.item(), euler_error.item())
        self.assertLess(heun_error.item(), 0.01)

    def test_ddim_runtime_steps_and_trajectory_use_initial_noise(self):
        objective = DDPMobjective(num_timesteps=10)
        initial_noise = torch.randn(2, 1, 4, 4)

        result = DDIMSampler(num_sampling_steps=5, eta=0.0).sample(
            model=ZeroVelocityModel(),
            objective=objective,
            shape=tuple(initial_noise.shape),
            device="cpu",
            initial_noise=initial_noise,
            num_steps=3,
            return_trajectory=True,
        )

        self.assertIsInstance(result, SamplingResult)
        self.assertEqual(result.trajectory.shape, (4, *initial_noise.shape))
        self.assertTrue(torch.equal(result.trajectory[0], initial_noise))
        self.assertEqual(result.timesteps.tolist(), [9, 4, 0, -1])

    def test_ddpm_rejects_shortened_ancestral_schedule(self):
        objective = DDPMobjective(num_timesteps=10)

        with self.assertRaisesRegex(ValueError, "Use DDIM"):
            DDPMAncestralSampler().sample(
                model=ZeroVelocityModel(),
                objective=objective,
                shape=(2, 1, 4, 4),
                device="cpu",
                num_steps=5,
            )

    def test_ddpm_trajectory_starts_from_supplied_noise(self):
        initial_noise = torch.randn(2, 1, 4, 4)
        result = DDPMAncestralSampler(clip_x0=False).sample(
            model=ZeroVelocityModel(),
            objective=DDPMobjective(num_timesteps=1),
            shape=tuple(initial_noise.shape),
            device="cpu",
            initial_noise=initial_noise,
            return_trajectory=True,
        )

        self.assertIsInstance(result, SamplingResult)
        self.assertEqual(result.trajectory.shape, (2, *initial_noise.shape))
        self.assertTrue(torch.equal(result.trajectory[0], initial_noise))
        self.assertTrue(torch.equal(result.trajectory[-1], result.samples))

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

    def test_ddim_timesteps_cover_training_schedule(self):
        sampler = DDIMSampler(
            num_sampling_steps=50,
            eta=0.0,
        )

        timesteps = sampler.build_timesteps(
            num_train_timesteps=1000,
        )

        self.assertEqual(timesteps.numel(), 50)
        self.assertEqual(timesteps[0].item(), 999)
        self.assertEqual(timesteps[-1].item(), 0)
        self.assertTrue(
            torch.all(timesteps[:-1] > timesteps[1:])
        )

    def test_ddim_sampler_smoke(self):
        objective = DDPMobjective(
            num_timesteps=10,
        )

        sampler = DDIMSampler(
            num_sampling_steps=5,
            eta=0.0,
            clip_x0=True,
        )

        model = ZeroVelocityModel()
        shape = (2, 1, 8, 8)
        initial_noise = torch.randn(shape)

        samples = sampler.sample(
            model=model,
            objective=objective,
            shape=shape,
            device="cpu",
            initial_noise=initial_noise,
        )

        self.assertEqual(samples.shape, shape)
        self.assertTrue(torch.isfinite(samples).all())

    def test_ddim_eta_zero_is_deterministic(self):
        objective = DDPMobjective(
            num_timesteps=10,
        )

        sampler = DDIMSampler(
            num_sampling_steps=5,
            eta=0.0,
            clip_x0=True,
        )

        model = ZeroVelocityModel()
        shape = (2, 1, 8, 8)
        initial_noise = torch.randn(shape)

        samples_1 = sampler.sample(
            model=model,
            objective=objective,
            shape=shape,
            device="cpu",
            initial_noise=initial_noise,
        )

        samples_2 = sampler.sample(
            model=model,
            objective=objective,
            shape=shape,
            device="cpu",
            initial_noise=initial_noise,
        )

        self.assertTrue(
            torch.allclose(samples_1, samples_2)
        )

    def test_ddpm_clipped_x0_is_used_in_posterior_mean(self):
        objective = DDPMobjective(
            num_timesteps=10,
            beta_start=0.1,
            beta_end=0.1,
            prediction_type="x0",
        )

        sampler = DDPMAncestralSampler(
            clip_x0=True,
        )

        # The model predicts x0 = 10, which must be clipped to 1.
        model = ConstantPredictionModel(value=10.0)

        xt = torch.zeros(2, 1, 4, 4)
        timesteps = torch.zeros(
            2,
            dtype=torch.long,
        )

        x_prev = sampler.p_sample(
            model=model,
            objective=objective,
            xt=xt,
            timesteps=timesteps,
        )

        expected = torch.ones_like(xt)

        self.assertTrue(
            torch.allclose(
                x_prev,
                expected,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_ddim_single_step_uses_last_training_timestep(self):
        sampler = DDIMSampler(
            num_sampling_steps=1,
            eta=0.0,
        )

        timesteps = sampler.build_timesteps(
            num_train_timesteps=1000,
        )

        self.assertEqual(timesteps.numel(), 1)
        self.assertEqual(timesteps[0].item(), 999)

if __name__ == "__main__":
    unittest.main()

