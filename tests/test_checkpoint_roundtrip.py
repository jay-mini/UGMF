# tests/test_checkpoint_roundtrip.py

from __future__ import annotations

import unittest

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ugm.models.unet import SimpleUNet
from ugm.objectives.ddpm import DDPMobjective
from ugm.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


class DictDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> None:
        super().__init__()

        if x.shape[0] != cond.shape[0]:
            raise ValueError(
                "x and cond must have the same first dimension, "
                f"but got {x.shape[0]} and {cond.shape[0]}."
            )

        self.x = x
        self.cond = cond

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "cond": self.cond[index],
        }


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively clone a state_dict.

    Optimizer state dictionaries may contain nested dictionaries, lists,
    tuples, tensors, integers, and floating-point values.
    """
    cloned: dict[str, Any] = {}

    for key, value in state_dict.items():
        cloned[key] = clone_nested_value(value)

    return cloned


def clone_nested_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()

    if isinstance(value, dict):
        return {
            key: clone_nested_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clone_nested_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            clone_nested_value(item)
            for item in value
        )

    return value


def assert_nested_equal(
    actual: Any,
    expected: Any,
) -> None:
    """
    Recursively compare nested optimizer/checkpoint state structures.
    """
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual.cpu(), expected.cpu())
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()

        for key in expected:
            assert_nested_equal(
                actual[key],
                expected[key],
            )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(actual, expected):
            assert_nested_equal(
                actual_item,
                expected_item,
            )
        return

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(actual, expected):
            assert_nested_equal(
                actual_item,
                expected_item,
            )
        return

    assert actual == expected


def assert_model_states_equal(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
) -> None:
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    assert state_a.keys() == state_b.keys()

    for name in state_a:
        assert torch.equal(
            state_a[name].cpu(),
            state_b[name].cpu(),
        ), f"Model state differs at {name!r}."


def build_test_data(
    batch_size: int = 8,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    x0 = torch.randn(
        batch_size,
        1,
        28,
        28,
    )

    time_steps = torch.full(
        (batch_size,),
        50,
        dtype=torch.long,
    )

    cond = torch.ones_like(time_steps)
    noise = torch.randn_like(x0)

    loader = DataLoader(
        DictDataset(
            x=x0,
            cond=cond,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(iter(loader))

    return batch, time_steps, noise


def train_one_fixed_step(
    *,
    model: torch.nn.Module,
    objective: DDPMobjective,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    time_steps: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    """
    Perform exactly one deterministic optimizer update.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = objective.compute_loss_with_fixed_noise(
        model=model,
        batch=batch,
        time_steps=time_steps,
        noise=noise,
    )

    loss = output.loss

    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss: {loss.item()}."
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    return float(loss.detach().cpu())


@pytest.mark.parametrize("num_steps_before_save", [1, 5])
def test_checkpoint_roundtrip_and_resume_consistency(
    tmp_path: Path,
    num_steps_before_save: int,
) -> None:
    """
    Verify:

    1. model state roundtrip;
    2. objective state roundtrip;
    3. optimizer state roundtrip;
    4. global_step continuity;
    5. the next resumed training step matches uninterrupted training.
    """
    seed_everything(42)

    batch, time_steps, noise = build_test_data(
        batch_size=8,
    )

    # ------------------------------------------------------------
    # 1. Build the original training objects.
    # ------------------------------------------------------------
    original_model = SimpleUNet()
    original_objective = DDPMobjective(1000)
    original_optimizer = torch.optim.Adam(
        original_model.parameters(),
        lr=1e-3,
    )

    global_step = 0

    # ------------------------------------------------------------
    # 2. Train several steps before saving.
    # ------------------------------------------------------------
    for _ in range(num_steps_before_save):
        train_one_fixed_step(
            model=original_model,
            objective=original_objective,
            optimizer=original_optimizer,
            batch=batch,
            time_steps=time_steps,
            noise=noise,
        )
        global_step += 1

    checkpoint_model_state = clone_state_dict(
        original_model.state_dict()
    )
    checkpoint_objective_state = clone_state_dict(
        original_objective.state_dict()
    )
    checkpoint_optimizer_state = clone_state_dict(
        original_optimizer.state_dict()
    )

    checkpoint_path = tmp_path / "roundtrip.pt"

    save_checkpoint(
        checkpoint_path,
        model=original_model,
        objective=original_objective,
        optimizer=original_optimizer,
        scheduler=None,
        scaler=None,
        epoch=3,
        global_step=global_step,
        config={
            "model": "SimpleUNet",
            "objective": "DDPM",
        },
        extra={
            "test_name": "checkpoint_roundtrip",
        },
    )

    assert checkpoint_path.is_file()

    # ------------------------------------------------------------
    # 3. Continue the original run for exactly one more step.
    #
    # This is the uninterrupted reference result.
    # ------------------------------------------------------------
    reference_next_loss = train_one_fixed_step(
        model=original_model,
        objective=original_objective,
        optimizer=original_optimizer,
        batch=batch,
        time_steps=time_steps,
        noise=noise,
    )

    reference_global_step = global_step + 1

    reference_model_after_next_step = clone_state_dict(
        original_model.state_dict()
    )
    reference_optimizer_after_next_step = clone_state_dict(
        original_optimizer.state_dict()
    )

    # ------------------------------------------------------------
    # 4. Construct fresh training objects.
    #
    # Their initial states are intentionally unrelated to the saved
    # checkpoint.
    # ------------------------------------------------------------
    seed_everything(999)

    resumed_model = SimpleUNet()
    resumed_objective = DDPMobjective(1000)
    resumed_optimizer = torch.optim.Adam(
        resumed_model.parameters(),
        lr=1e-3,
    )

    # ------------------------------------------------------------
    # 5. Load the checkpoint.
    # ------------------------------------------------------------
    resume_state = load_checkpoint(
        checkpoint_path,
        model=resumed_model,
        objective=resumed_objective,
        optimizer=resumed_optimizer,
        scheduler=None,
        scaler=None,
        map_location="cpu",
        strict_model=True,
        strict_objective=True,
        restore_rng=True,
    )

    # ------------------------------------------------------------
    # 6. Verify checkpoint metadata.
    # ------------------------------------------------------------
    assert resume_state.epoch == 3
    assert resume_state.global_step == global_step

    assert resume_state.config == {
        "model": "SimpleUNet",
        "objective": "DDPM",
    }

    assert resume_state.extra == {
        "test_name": "checkpoint_roundtrip",
    }

    # ------------------------------------------------------------
    # 7. Verify model and objective roundtrip.
    # ------------------------------------------------------------
    assert_nested_equal(
        resumed_model.state_dict(),
        checkpoint_model_state,
    )

    assert_nested_equal(
        resumed_objective.state_dict(),
        checkpoint_objective_state,
    )

    # ------------------------------------------------------------
    # 8. Verify optimizer state roundtrip.
    #
    # For Adam this includes:
    # - step
    # - exp_avg
    # - exp_avg_sq
    # - parameter-group hyperparameters
    # ------------------------------------------------------------
    assert_nested_equal(
        resumed_optimizer.state_dict(),
        checkpoint_optimizer_state,
    )

    # Adam state should no longer be empty after at least one update.
    assert len(
        resumed_optimizer.state_dict()["state"]
    ) > 0

    # ------------------------------------------------------------
    # 9. Resume for exactly one more step.
    # ------------------------------------------------------------
    resumed_global_step = resume_state.global_step

    resumed_next_loss = train_one_fixed_step(
        model=resumed_model,
        objective=resumed_objective,
        optimizer=resumed_optimizer,
        batch=batch,
        time_steps=time_steps,
        noise=noise,
    )

    resumed_global_step += 1

    # ------------------------------------------------------------
    # 10. Verify global-step continuity.
    # ------------------------------------------------------------
    assert resumed_global_step == reference_global_step
    assert resumed_global_step == global_step + 1

    # ------------------------------------------------------------
    # 11. Verify that the next loss is identical.
    # ------------------------------------------------------------
    assert resumed_next_loss == pytest.approx(
        reference_next_loss,
        rel=0.0,
        abs=0.0,
    )

    # ------------------------------------------------------------
    # 12. Verify that the model after the next update is identical.
    # ------------------------------------------------------------
    assert_nested_equal(
        resumed_model.state_dict(),
        reference_model_after_next_step,
    )

    # ------------------------------------------------------------
    # 13. Verify that the optimizer state after the next update is
    #     also identical.
    # ------------------------------------------------------------
    assert_nested_equal(
        resumed_optimizer.state_dict(),
        reference_optimizer_after_next_step,
    )

