from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

import scripts.finetune_rf as finetune_rf
import scripts.generate_rf_pairs as generate_rf_pairs
from ugm.data.rectified_flow import (
    PAIR_DATASET_FORMAT,
    PAIR_DATASET_VERSION,
    RectifiedFlowPairDataset,
    ShardShuffleSampler,
)
from ugm.objectives.rectified_flow import RectifiedFlowObjective
from ugm.trainers.finetune_trainer import initialize_from_checkpoint


class TinyVelocity(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, x, t, cond=None):
        del t, cond
        return torch.ones_like(x) * self.weight


def write_pairs(root: Path, x0: torch.Tensor, x1: torch.Tensor) -> Path:
    root.mkdir(parents=True)
    torch.save({"x0": x0, "x1": x1}, root / "pairs-00000.pt")
    manifest = {
        "format": PAIR_DATASET_FORMAT,
        "version": PAIR_DATASET_VERSION,
        "num_samples": x0.shape[0],
        "sample_shape": list(x0.shape[1:]),
        "dtype": "float32",
        "conditional": False,
        "num_classes": None,
        "dataset": "test",
        "teacher": {"checkpoint": "teacher.pt"},
        "generation": {
            "seed": 1,
            "solver": "euler",
            "num_steps": 2,
            "guidance_scale": 1.0,
            "label_policy": None,
        },
        "shards": [{"file": "pairs-00000.pt", "start": 0, "count": x0.shape[0]}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_pair_dataset_loads_noise_teacher_output_and_negative_index(tmp_path: Path) -> None:
    x0 = torch.arange(24, dtype=torch.float32).reshape(3, 1, 2, 4)
    x1 = x0 + 5
    dataset = RectifiedFlowPairDataset(write_pairs(tmp_path / "pairs", x0, x1))

    assert len(dataset) == 3
    assert torch.equal(dataset[1]["x0"], x0[1])
    assert torch.equal(dataset[-1]["x1"], x1[-1])
    indices = list(
        ShardShuffleSampler(
            dataset,
            generator=torch.Generator().manual_seed(1),
        )
    )
    assert sorted(indices) == [0, 1, 2]


def test_rectified_flow_loss_is_finite_and_updates_parameters() -> None:
    model = TinyVelocity(0.0)
    objective = RectifiedFlowObjective(t_min=0.1, t_max=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {
        "x0": torch.zeros(4, 1, 2, 2),
        "x1": torch.ones(4, 1, 2, 2),
    }
    before = model.weight.detach().clone()
    output = objective.compute_loss_with_fixed_time(
        model,
        batch,
        torch.full((4,), 0.5),
    )
    assert torch.isfinite(output.loss)
    assert output.loss.item() == pytest.approx(1.0)
    output.loss.backward()
    optimizer.step()
    assert not torch.equal(model.weight.detach(), before)


def test_initialization_loads_weights_but_not_training_step(tmp_path: Path) -> None:
    source = TinyVelocity(0.75)
    path = tmp_path / "teacher.pt"
    torch.save({"model": source.state_dict(), "global_step": 123}, path)
    target = TinyVelocity(-1.0)

    state = initialize_from_checkpoint(target, path)

    assert target.weight.item() == pytest.approx(0.75)
    assert state.checkpoint_global_step == 123
    assert state.source == "online"


def test_pair_generation_records_reproducible_teacher_coupling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    output_dir = tmp_path / "generated"
    teacher = TinyVelocity(2.0)
    config = {
        "objective": "fm",
        "dataset": "test",
        "in_channels": 1,
        "image_size": 2,
        "class_cond": True,
        "num_classes": 3,
    }
    args = argparse.Namespace(
        teacher_ckpt=checkpoint,
        output_dir=output_dir,
        num_samples=5,
        batch_size=2,
        shard_size=4,
        num_steps=2,
        solver="euler",
        guidance_scale=1.0,
        label_policy="balanced",
        seed=7,
        device="cpu",
        use_ema=False,
        overwrite=False,
    )
    with mock.patch.object(
        generate_rf_pairs,
        "build_teacher",
        return_value=(teacher, config, {"checkpoint": str(checkpoint), "state_source": "online"}),
    ):
        manifest_path = generate_rf_pairs.generate_pairs(args)

    dataset = RectifiedFlowPairDataset(output_dir)
    assert manifest_path.is_file()
    assert len(dataset) == 5
    for index in range(len(dataset)):
        item = dataset[index]
        assert torch.allclose(item["x1"], item["x0"] + 2.0)
        assert item["cond"].item() == index % 3


def test_finetune_script_initializes_and_saves_updated_checkpoint(tmp_path: Path) -> None:
    pairs_dir = write_pairs(
        tmp_path / "pairs",
        torch.zeros(4, 1, 2, 2),
        torch.ones(4, 1, 2, 2),
    )
    teacher = TinyVelocity(0.25)
    init_checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "model": teacher.state_dict(),
            "global_step": 9,
            "config": {
                "objective": "fm",
                "dataset": "test",
                "in_channels": 1,
                "out_channels": 1,
                "image_size": 2,
                "base_channels": 1,
                "time_dim": 4,
                "class_cond": False,
            },
        },
        init_checkpoint,
    )
    output_dir = tmp_path / "run"
    config = dict(finetune_rf.RF_DEFAULT_CONFIG)
    config.update(
        {
            "pairs_dir": str(pairs_dir),
            "init_checkpoint": str(init_checkpoint),
            "output_dir": str(output_dir),
            "batch_size": 4,
            "num_epochs": 1,
            "max_steps": 1,
            "num_workers": 0,
            "scheduler": "none",
            "lr": 0.1,
            "log_every": 1,
            "save_every": 100,
            "config_path": "test.yaml",
        }
    )
    student = TinyVelocity()
    with mock.patch.object(finetune_rf, "build_model", return_value=student):
        trainer = finetune_rf.run_finetuning(config)

    checkpoint = torch.load(
        output_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert trainer.global_step == 1
    assert checkpoint["global_step"] == 1
    assert torch.isfinite(checkpoint["model"]["weight"])
    assert checkpoint["model"]["weight"].item() != pytest.approx(0.25)
    assert checkpoint["config"]["initialization_checkpoint_global_step"] == 9
