from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn

@dataclass
class LossOutput:
    """
    Standard output returned by all generative training objectives.

    Attributes:
    loss: Scalar tensor used for backpropagation.
    metrics: Detached logging metrics.
    aux: Optional intermediate tensors or metadata for debugging.
    """
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    aux: dict[str, Any] | None = None


class BaseObjective(nn.Module, ABC):
    """
    Base class for all generative objectives.
    
    The trainer should not care whether the objective is DDPM, Flow Matching, 
    Reactified Flow, or Consistency Distillation.
    
    Each objective is responsible for:

    1. sampling time/noise;
    2. constructing x_t;
    3. defining the training target;
    4. computing the loss.
    """

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> LossOutput:
        raise NotImplementedError("Subclasses must implement compute_loss method.")
    
    
