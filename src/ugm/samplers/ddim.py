from __future__ import annotations

import torch
from torch import nn

from ugm.models.conditioning import guided_prediction, validate_guidance_scale
from ugm.samplers.base import BaseSampler, SampleOutput


class DDIMSampler(BaseSampler):
    def __init__(
        self,
        num_sampling_steps: int = 50,
        eta: float = 0.0,
        clip_x0: bool = True
    ) -> None:
        super().__init__()

        if num_sampling_steps <= 0:
            raise ValueError(
                "num_steps must be positive."
            )

        if eta < 0.0:
            raise ValueError(
                "eta must be non-negative."
            )

        self.num_sampling_steps = num_sampling_steps
        self.eta = eta
        self.clip_x0 = clip_x0

    def build_timesteps(
        self,
        num_train_timesteps: int,
        num_sampling_steps: int | None = None,
    ) -> torch.Tensor:
        steps = self._resolve_num_steps(
            num_sampling_steps,
            self.num_sampling_steps,
        )

        if steps > num_train_timesteps:
            raise ValueError(
                "num_sampling_steps cannot exceed "
                "num_train_timesteps"
            )

        if steps == 1:
            return torch.tensor(
                [num_train_timesteps - 1],
                dtype=torch.long,
            )

        timesteps = torch.linspace(
            0,
            num_train_timesteps - 1,
            steps=steps,
        )

        timesteps = (
            timesteps.round().long().unique(sorted=True)
        )

        if timesteps.numel() != steps:
            raise RuntimeError(
                "Duplicate DDIM timesteps detected."
            )

        return timesteps.flip(0)

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        objective: nn.Module,
        xt: torch.Tensor,
        timestep: int,
        previous_timestep: int,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:

        batch_size = xt.shape[0]


        timesteps = torch.full(
            (batch_size,),
            timestep,
            device=xt.device,
            dtype=torch.long,
        )


        model_time = objective.normalize_timesteps(
            timesteps
        )


        # -------------------------
        # model prediction
        # -------------------------

        model_output = guided_prediction(
            model,
            xt,
            model_time,
            cond,
            guidance_scale,
        )


        # -------------------------
        # convert prediction
        # -------------------------

        x0_pred = objective.predict_x0(
            xt,
            timesteps,
            model_output,
        )


        noise_pred = objective.predict_eps(
            xt,
            timesteps,
            model_output,
        )


        if self.clip_x0:
            x0_pred = x0_pred.clamp(
                -1,
                1,
            )

            # recompute eps after clipping

            noise_pred = objective.predict_eps_from_x0(
                xt,
                timesteps,
                x0_pred,
            )


        if previous_timestep < 0:
            return x0_pred


        alpha_bar_t = objective.alpha_bars[
            timestep
        ].to(
            device=xt.device,
            dtype=xt.dtype,
        )


        alpha_bar_prev = objective.alpha_bars[
            previous_timestep
        ].to(
            device=xt.device,
            dtype=xt.dtype,
        )


        alpha_bar_t = alpha_bar_t.reshape(
            1,
            *((1,) * (xt.ndim-1)),
        )


        alpha_bar_prev = alpha_bar_prev.reshape(
            1,
            *((1,) * (xt.ndim-1)),
        )


        sigma = (
            self.eta
            *
            torch.sqrt(
                (
                    (1-alpha_bar_prev)
                    /
                    (1-alpha_bar_t)
                )
                *
                (
                    1
                    -
                    alpha_bar_t
                    /
                    alpha_bar_prev
                )
            )
        )


        direction = torch.sqrt(
            (
                1
                -
                alpha_bar_prev
                -
                sigma.square()
            ).clamp_min(0)
        ) * noise_pred


        mean = (
            torch.sqrt(alpha_bar_prev)
            *
            x0_pred
            +
            direction
        )


        if self.eta == 0:
            return mean


        return mean + sigma * torch.randn_like(xt)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        objective: nn.Module | None = None,
        shape: tuple[int, ...] | None = None,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        device: torch.device | str = "cuda",
        initial_noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> SampleOutput:
        if objective is None:
            raise ValueError("DDIM sampling requires a DDPM objective.")
        shape = self._require_shape(shape)

        model.eval()

        device = torch.device(device)
        guidance_scale = validate_guidance_scale(guidance_scale)
        objective.to(device)
        x = self._prepare_initial_noise(shape, device, initial_noise)
        cond = self._prepare_condition(cond, shape[0], device)
        states = [x.clone()] if return_trajectory else None

        timesteps = self.build_timesteps(
            objective.num_timesteps,
            num_sampling_steps=num_steps,
        ).tolist()

        previous_timesteps = (
            timesteps[1:] + [-1]
        )

        for timestep, previous_timestep in zip(
            timesteps,
            previous_timesteps,
        ):
            x = self.step(
                model=model,
                objective=objective,
                xt=x,
                timestep=timestep,
                previous_timestep=(
                    previous_timestep
                ),
                cond=cond,
                guidance_scale=guidance_scale,
            )

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    "Non-finite DDIM sample at "
                    f"timestep {timestep}"
                )
            if states is not None:
                states.append(x.clone())

        trajectory_times = torch.tensor(
            [timesteps[0], *previous_timesteps],
            device=device,
            dtype=torch.long,
        )
        return self._format_output(x, states, trajectory_times)
