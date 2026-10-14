"""Rectified Flow uses the same ODE solvers as Flow Matching."""

from ugm.samplers.ode import EulerODESampler, HeunODESampler, ODESampler

RectifiedFlowSampler = ODESampler

__all__ = [
    "EulerODESampler",
    "HeunODESampler",
    "ODESampler",
    "RectifiedFlowSampler",
]
