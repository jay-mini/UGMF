import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugm.models.conditioning import (
    UNCONDITIONAL_CLASS_LABEL,
    drop_class_condition,
    guided_prediction,
)
from ugm.models.unet import SimpleUNet
from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.samplers.ddim import DDIMSampler
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.samplers.ode import ODESampler
from ugm.trainers.base_trainer import BaseTrainer


class FixedBranchModel(nn.Module):
    """Return visibly different unconditional and conditional predictions."""

    def forward(self, x, t, cond=None):
        value = 2.0 if cond is None else 5.0
        return torch.full_like(x, value)


class ClassifierFreeGuidanceTests(unittest.TestCase):
    def test_guidance_scale_zero_and_one_match_defined_branches(self):
        model = FixedBranchModel()
        x = torch.zeros(2, 1)
        t = torch.zeros(2)
        cond = torch.tensor([0, 1])

        unconditional = model(x, t, cond=None)
        conditional = model(x, t, cond=cond)

        scale_zero = guided_prediction(
            model,
            x,
            t,
            cond,
            guidance_scale=0.0,
        )
        scale_one = guided_prediction(
            model,
            x,
            t,
            cond,
            guidance_scale=1.0,
        )
        scale_two = guided_prediction(
            model,
            x,
            t,
            cond,
            guidance_scale=2.0,
        )

        self.assertTrue(torch.equal(scale_zero, unconditional))
        self.assertTrue(torch.equal(scale_one, conditional))
        self.assertTrue(torch.equal(scale_two, torch.full_like(x, 8.0)))

    def test_ddpm_and_ddim_steps_apply_cfg_boundary_scales(self):
        objective = DDPMobjective(
            num_timesteps=1,
            beta_start=0.1,
            beta_end=0.1,
            prediction_type="x0",
        )
        model = FixedBranchModel()
        xt = torch.zeros(2, 1)
        timesteps = torch.zeros(2, dtype=torch.long)
        cond = torch.tensor([0, 1])

        ddpm = DDPMAncestralSampler(clip_x0=False)
        ddim = DDIMSampler(num_sampling_steps=1, eta=0.0, clip_x0=False)

        for scale, expected_value in ((0.0, 2.0), (1.0, 5.0)):
            with self.subTest(sampler="ddpm", guidance_scale=scale):
                result = ddpm.p_sample(
                    model=model,
                    objective=objective,
                    xt=xt,
                    timesteps=timesteps,
                    cond=cond,
                    guidance_scale=scale,
                )
                self.assertTrue(
                    torch.allclose(
                        result,
                        torch.full_like(xt, expected_value),
                    )
                )

            with self.subTest(sampler="ddim", guidance_scale=scale):
                result = ddim.step(
                    model=model,
                    objective=objective,
                    xt=xt,
                    timestep=0,
                    previous_timestep=-1,
                    cond=cond,
                    guidance_scale=scale,
                )
                self.assertTrue(
                    torch.equal(result, torch.full_like(xt, expected_value))
                )

    def test_ode_sampling_applies_cfg_for_euler_and_heun(self):
        shape = (2, 1)
        initial_noise = torch.zeros(shape)
        cond = torch.tensor([0, 1])

        for solver in ("euler", "heun"):
            with self.subTest(solver=solver):
                sampler = ODESampler(num_steps=1, solver=solver)
                unconditional = sampler.sample(
                    model=FixedBranchModel(),
                    objective=None,
                    shape=shape,
                    cond=cond,
                    guidance_scale=0.0,
                    device="cpu",
                    initial_noise=initial_noise,
                )
                conditional = sampler.sample(
                    model=FixedBranchModel(),
                    objective=None,
                    shape=shape,
                    cond=cond,
                    guidance_scale=1.0,
                    device="cpu",
                    initial_noise=initial_noise,
                )

                self.assertTrue(
                    torch.equal(unconditional, torch.full(shape, 2.0))
                )
                self.assertTrue(
                    torch.equal(conditional, torch.full(shape, 5.0))
                )

    def test_condition_dropout_boundaries(self):
        cond = torch.tensor([0, 1, 2, 3])

        kept = drop_class_condition(cond, drop_probability=0.0)
        dropped = drop_class_condition(cond, drop_probability=1.0)

        self.assertTrue(torch.equal(kept, cond))
        self.assertTrue(
            torch.equal(
                dropped,
                torch.full_like(cond, UNCONDITIONAL_CLASS_LABEL),
            )
        )

    def test_null_labels_match_the_unconditional_unet_branch(self):
        torch.manual_seed(7)
        model = SimpleUNet(
            in_channels=1,
            base_channels=4,
            time_dim=8,
            num_classes=3,
        ).eval()
        x = torch.randn(2, 1, 8, 8)
        t = torch.tensor([0.25, 0.75])
        null_cond = torch.full(
            (2,),
            UNCONDITIONAL_CLASS_LABEL,
            dtype=torch.long,
        )

        with torch.no_grad():
            from_none = model(x, t, cond=None)
            from_null_labels = model(x, t, cond=null_cond)

        self.assertTrue(torch.equal(from_none, from_null_labels))

    def test_conditional_training_step_runs_with_random_dropout(self):
        torch.manual_seed(11)
        x = torch.randn(4, 1, 8, 8)
        labels = torch.tensor([0, 1, 2, 1])
        loader = DataLoader(
            TensorDataset(x, labels),
            batch_size=4,
            shuffle=False,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            for objective_name, objective in (
                ("ddpm", DDPMobjective(num_timesteps=2)),
                ("fm", FlowMatchingObjective()),
            ):
                with self.subTest(objective=objective_name):
                    model = SimpleUNet(
                        in_channels=1,
                        base_channels=4,
                        time_dim=8,
                        num_classes=3,
                    )
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=1e-3,
                    )
                    trainer = BaseTrainer(
                        model=model,
                        objective=objective,
                        optimizer=optimizer,
                        train_loader=loader,
                        device="cpu",
                        output_dir=Path(temporary_dir) / objective_name,
                        mixed_precision="no",
                        grad_clip=1.0,
                        cond_drop_prob=0.5,
                    )
                    prepared = trainer._prepare_batch(next(iter(loader)))
                    metrics = trainer._train_step(prepared)

                    self.assertEqual(trainer.global_step, 1)
                    self.assertTrue(
                        torch.isfinite(torch.tensor(metrics["loss"]))
                    )


if __name__ == "__main__":
    unittest.main()
