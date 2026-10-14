import sys
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.base import BaseObjective, LossOutput
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.models.unet import SimpleUNet
from ugm.trainers.base_trainer import BaseTrainer
from ugm.utils.ema import EMA

class ZeroVelocityModel(nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)
    
class DictDataset(torch.utils.data.Dataset):
    def __init__(self, x:torch.Tensor, cond:torch.Tensor):
        super().__init__()
        self.x = x
        self.cond = cond

    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "cond": self.cond[index]
        }


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t, cond=None):
        return self.weight * x


class ScalarObjective(BaseObjective):
    def compute_loss(self, model, batch):
        x = batch["x"]
        t = torch.zeros(x.shape[0], device=x.device)
        prediction = model(x, t)
        loss = prediction.square().mean()
        return LossOutput(
            loss=loss,
            metrics={"loss": loss.detach()},
        )


class TinyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond=None):
        return self.conv(x)


class PredictionEnergyObjective(BaseObjective):
    def compute_loss(self, model, batch):
        x = batch["x"]
        t = torch.zeros(x.shape[0], device=x.device)
        prediction = model(x, t, cond=batch.get("cond"))
        loss = prediction.square().mean()
        return LossOutput(
            loss=loss,
            metrics={"loss": loss.detach()},
        )
    

def test_training_step_updates_parameters() -> None:
    batch_size = 8

    batch = {
        "x": torch.randn(
            batch_size,
            1,
            28,
            28,
        ),
        "cond": torch.ones(
            batch_size,
            dtype=torch.long,
        ),
    }

    model = SimpleUNet()
    objective = DDPMobjective()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer.zero_grad(set_to_none=True)

    output = objective.compute_loss(
        model=model,
        batch=batch,
    )

    output.loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter in model.named_parameters()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to exercise fp16 autocast and GradScaler.",
)
def test_cuda_amp_has_finite_loss_and_updates_parameters() -> None:
    model = TinyConvModel()
    objective = PredictionEnergyObjective()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 1, 8, 8)),
        batch_size=4,
        shuffle=False,
    )

    with tempfile.TemporaryDirectory() as temporary_dir:
        trainer = BaseTrainer(
            model=model,
            objective=objective,
            optimizer=optimizer,
            train_loader=loader,
            device="cuda",
            output_dir=temporary_dir,
            mixed_precision="fp16",
            grad_clip=1.0,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.model.named_parameters()
        }
        batch = trainer._prepare_batch(next(iter(loader)))
        metrics = trainer._train_step(batch)

        assert trainer.use_amp
        assert trainer.scaler.is_enabled()
        assert trainer.global_step == 1
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in trainer.model.named_parameters()
        )


def _build_stateful_trainer(output_dir: Path) -> BaseTrainer:
    dataset = DictDataset(
        x=torch.ones(6, 1),
        cond=torch.zeros(6, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    model = ScalarModel()
    objective = ScalarObjective()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )
    ema = EMA(
        model,
        decay=0.5,
        update_after_step=0,
        update_every=1,
    )

    return BaseTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        device="cpu",
        output_dir=output_dir,
        mixed_precision="no",
        log_every=100,
        save_every=100,
        config={"scheduler": "step_test"},
        ema=ema,
    )


def test_resume_preserves_step_lr_and_ema_without_extra_max_step() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        original = _build_stateful_trainer(root / "original")
        original.train(num_epochs=2, max_steps=2)

        checkpoint_path = original.ckpt_dir / "last.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        assert checkpoint["global_step"] == 2
        assert checkpoint["scheduler"] is not None
        assert checkpoint["ema"] is not None
        assert checkpoint["extra"]["current_epoch"] == 0
        assert checkpoint["extra"]["batches_consumed_in_epoch"] == 2
        assert (
            checkpoint["extra"]["resume_behavior"]
            == "restart_current_epoch"
        )

        resumed = _build_stateful_trainer(root / "resumed")
        with pytest.warns(RuntimeWarning, match="mid-epoch"):
            resume_state = resumed.load_checkpoint(checkpoint_path)

        assert resume_state.global_step == 2
        assert resumed.global_step == 2
        assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
        assert resumed.scheduler.last_epoch == 2
        assert resumed.ema.num_updates == 2

        parameter_before_noop = resumed.model.weight.detach().clone()
        ema_before_noop = resumed.ema.ema_model.weight.detach().clone()
        lr_before_noop = resumed.optimizer.param_groups[0]["lr"]
        scheduler_epoch_before_noop = resumed.scheduler.last_epoch
        ema_updates_before_noop = resumed.ema.num_updates

        # Reaching max_steps before train() is called must perform no update.
        resumed.train(num_epochs=2, max_steps=2)

        assert resumed.global_step == 2
        assert torch.equal(resumed.model.weight, parameter_before_noop)
        assert torch.equal(resumed.ema.ema_model.weight, ema_before_noop)
        assert resumed.optimizer.param_groups[0]["lr"] == lr_before_noop
        assert resumed.scheduler.last_epoch == scheduler_epoch_before_noop
        assert resumed.ema.num_updates == ema_updates_before_noop

        # Compare one continued step against the uninterrupted trainer.
        original.train(num_epochs=2, max_steps=3)
        resumed.train(num_epochs=2, max_steps=3)

        assert original.global_step == resumed.global_step == 3
        assert original.scheduler.last_epoch == resumed.scheduler.last_epoch == 3
        assert original.optimizer.param_groups[0]["lr"] == pytest.approx(
            resumed.optimizer.param_groups[0]["lr"]
        )
        assert original.ema.num_updates == resumed.ema.num_updates == 3
        assert torch.equal(original.model.weight, resumed.model.weight)
        assert torch.equal(
            original.ema.ema_model.weight,
            resumed.ema.ema_model.weight,
        )


if __name__ == '__main__':
    batch_size = 8

    x0 = torch.randn([batch_size, 1, 28, 28])

    time_steps = torch.full(
        (batch_size,),
        50,
        dtype=torch.long,
    )

    cond = torch.ones_like(time_steps)

    noise = torch.randn_like(x0)

    loader = DataLoader(
        DictDataset(x0, cond=cond),
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(iter(loader))

    test_training_step_updates_parameters()
