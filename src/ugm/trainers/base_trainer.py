# src/ugm/trainers/base_trainer.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class BaseTrainer:
    """
    Minimal reusable trainer.

    The trainer does not know whether the training objective is DDPM or FM.
    It only calls:

        loss_output = objective.compute_loss(model, batch)

    This is the key abstraction.
    """

    def __init__(
        self,
        model: nn.Module,
        objective: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        device: torch.device | str = "cuda",
        scheduler: Any | None = None,
        output_dir: str | Path = "outputs/default",
        mixed_precision: str = "no",
        grad_clip: float | None = 1.0,
        log_every: int = 100,
        save_every: int = 1000,
        config: dict[str, Any] | None = None,
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.objective = objective.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader

        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.mixed_precision = mixed_precision
        self.grad_clip = grad_clip
        self.log_every = int(log_every)
        self.save_every = int(save_every)

        self.global_step = 0
        self.start_epoch = 0

        self.use_amp = (
            self.device.type == "cuda"
            and self.mixed_precision in {"fp16", "bf16"}
        )

        self.amp_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16
        )

        self.scaler = GradScaler(
            enabled=self.use_amp and self.mixed_precision == "fp16"
        )

        if config is not None:
            self._save_config(config)

    def train(
        self,
        num_epochs: int,
        max_steps: int | None = None,
    ) -> None:
        self.model.train()

        for epoch in range(self.start_epoch, num_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for raw_batch in pbar:
                batch = self._prepare_batch(raw_batch)

                with autocast(
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    loss_output = self.objective.compute_loss(
                        model=self.model,
                        batch=batch,
                    )
                    loss = loss_output.loss

                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()

                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip,
                        )

                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                metrics = {
                    name: float(value.detach().cpu())
                    for name, value in loss_output.metrics.items()
                }

                pbar.set_postfix(metrics)

                if self.global_step % self.log_every == 0:
                    self._print_metrics(epoch, metrics)

                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(
                        self.ckpt_dir / f"step_{self.global_step}.pt",
                        epoch=epoch,
                    )
                    self.save_checkpoint(
                        self.ckpt_dir / "last.pt",
                        epoch=epoch,
                    )

                if max_steps is not None and self.global_step >= max_steps:
                    self.save_checkpoint(
                        self.ckpt_dir / "last.pt",
                        epoch=epoch,
                    )
                    return

        self.save_checkpoint(
            self.ckpt_dir / "last.pt",
            epoch=num_epochs - 1,
        )

    def _prepare_batch(
        self,
        raw_batch,
    ) -> dict[str, torch.Tensor]:
        """
        Convert a torchvision-style batch into the unified dict format.

        torchvision returns:
            (images, labels)

        We convert it into:
            {
                "x": images,
                "cond": labels
            }
        """
        if isinstance(raw_batch, dict):
            batch = raw_batch
        elif isinstance(raw_batch, (tuple, list)):
            if len(raw_batch) == 2:
                x, y = raw_batch
                batch = {
                    "x": x,
                    "cond": y,
                }
            elif len(raw_batch) == 1:
                batch = {
                    "x": raw_batch[0],
                }
            else:
                raise ValueError("Unsupported batch tuple/list format.")
        else:
            batch = {
                "x": raw_batch,
            }

        batch_on_device = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_on_device[key] = value.to(self.device, non_blocking=True)
            else:
                batch_on_device[key] = value

        return batch_on_device

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
    ) -> None:
        path = Path(path)

        checkpoint = {
            "model": self.model.state_dict(),
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "global_step": self.global_step,
            "epoch": epoch,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str | Path,
        strict: bool = True,
    ) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"], strict=strict)
        self.objective.load_state_dict(checkpoint["objective"], strict=strict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

    def _print_metrics(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        metric_str = " | ".join(
            f"{key}: {value:.6f}" for key, value in metrics.items()
        )
        print(
            f"[epoch={epoch + 1} step={self.global_step}] {metric_str}"
        )

    def _save_config(
        self,
        config: dict[str, Any],
    ) -> None:
        config_path = self.output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)