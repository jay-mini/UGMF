from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ugm.objectives.consistency import ConsistencyDistillationObjective
from ugm.trainers.distill_trainer import ConsistencyDistillationTrainer


class ConstantVelocity(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, x, t, cond=None):
        del t, cond
        return torch.ones_like(x) * self.weight


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        del index
        return {"x": torch.ones(1, 2, 2)}


def test_boundary_parameterization_is_exact_at_data_time() -> None:
    model = ConstantVelocity(123.0)
    x = torch.randn(3, 1, 2, 2)
    t = torch.ones(3)

    output = ConsistencyDistillationObjective.boundary_prediction(model, x, t)

    assert torch.equal(output, x)


def test_teacher_step_moves_from_noise_time_toward_data_time() -> None:
    teacher = ConstantVelocity(2.0)
    x = torch.zeros(2, 1, 1, 1)
    t_current = torch.tensor([0.0, 0.5])
    t_next = torch.tensor([0.25, 0.75])

    moved = ConsistencyDistillationObjective.teacher_euler_step(
        teacher,
        x,
        t_current,
        t_next,
    )

    assert torch.allclose(moved, torch.full_like(x, 0.5))
    with pytest.raises(ValueError, match="t_next > t_current"):
        ConsistencyDistillationObjective.teacher_euler_step(
            teacher,
            x,
            t_next,
            t_current,
        )


def test_distillation_gradients_only_reach_student() -> None:
    teacher = ConstantVelocity(1.0)
    student = ConstantVelocity(0.0)
    objective = ConsistencyDistillationObjective(
        teacher,
        student,
        num_time_steps=4,
        target_decay=0.5,
    )
    batch = {"x": torch.ones(2, 1, 1, 1)}
    output = objective.compute_loss_with_fixed_inputs(
        student,
        batch,
        noise=torch.zeros_like(batch["x"]),
        time_indices=torch.tensor([0, 3]),
    )

    assert torch.isfinite(output.loss)
    assert torch.all(output.aux["t_next"] > output.aux["t_current"])
    assert output.aux["t_next"][1].item() == pytest.approx(1.0)
    output.loss.backward()

    assert student.weight.grad is not None
    assert torch.isfinite(student.weight.grad)
    assert teacher.weight.grad is None
    assert objective.target.weight.grad is None
    assert not teacher.training
    assert not objective.target.training


def test_ema_target_updates_by_configured_rule() -> None:
    teacher = ConstantVelocity(0.0)
    student = ConstantVelocity(0.0)
    objective = ConsistencyDistillationObjective(
        teacher,
        student,
        target_decay=0.75,
        target_update_every=2,
    )
    student.weight.data.fill_(2.0)

    assert not objective.update_target(student, global_step=1)
    assert objective.target.weight.item() == pytest.approx(0.0)
    assert objective.update_target(student, global_step=2)
    assert objective.target.weight.item() == pytest.approx(0.5)
    assert objective.num_target_updates.item() == 1


def test_trainer_updates_student_and_then_target(tmp_path: Path) -> None:
    teacher = ConstantVelocity(1.0)
    student = ConstantVelocity(0.0)
    objective = ConsistencyDistillationObjective(
        teacher,
        student,
        num_time_steps=2,
        target_decay=0.5,
    )
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    trainer = ConsistencyDistillationTrainer(
        model=student,
        objective=objective,
        optimizer=optimizer,
        train_loader=DataLoader(TinyDataset(), batch_size=2),
        device="cpu",
        output_dir=tmp_path,
        mixed_precision="no",
        log_every=100,
        save_every=100,
    )
    before_student = student.weight.detach().clone()
    before_target = objective.target.weight.detach().clone()
    metrics = trainer._train_step(trainer._prepare_batch(next(iter(trainer.train_loader))))

    assert trainer.global_step == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(student.weight.detach(), before_student)
    assert not torch.equal(objective.target.weight.detach(), before_target)
    expected_target = 0.5 * before_target + 0.5 * student.weight.detach()
    assert torch.allclose(objective.target.weight.detach(), expected_target)
    assert metrics["target_updated"] == 1.0
    assert teacher.weight.grad is None
