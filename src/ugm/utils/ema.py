from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(
        self, 
        model: nn.Module,
        decay: float=0.9999,
        update_after_step: int=0,
        update_every: int=1,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(
                f"decay must be in [0, 1], got {decay}"
            )
        
        if update_every <= 0:
            raise ValueError(
                "update_every must be positive"
            )
        
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.num_updates = 0

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        global_step: int,
    ) -> None:
        if global_step < self.update_after_step:
            self.copy_from(model)
            return
        
        if global_step % self.update_every != 0:
            return
        
        self.num_updates += 1

        model_parameters = dict(
            model.named_parameters()
        )

        for name, ema_parameter in (
            self.ema_model.named_parameters()
        ):
            model_parameter = model_parameters[name]

            ema_parameter.lerp_(
                model_parameter.detach(),
                1.0 - self.decay,
            )

        model_bufers = dict(
            model.named_buffers()
        )

        for name, ema_buffer in (
            self.ema_model.named_buffers()
        ):
            ema_buffer.copy_(
                model_bufers[name]
            )

    @torch.no_grad()
    def copy_from(
        self,
        model: nn.Module,
    ) -> None:
        self.ema_model.load_state_dict(
            model.state_dict()
        )

    def state_dict(self) -> dict:
        return {
            "ema_model": self.ema_model.state_dict(),
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
        }
    
    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self.ema_model.load_state_dict(
            state_dict["ema_model"]
        )

        self.decay = state_dict["decay"]
        self.update_after_step = state_dict[
            "update_after_step"
        ]
        self.update_every = state_dict[
            "update_every"
        ]
        self.num_updates = state_dict[
            "num_updates"
        ]