# src/ugm/trainers/base_trainer.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ugm.utils.checkpoint import (
    ResumeState,
    load_checkpoint as load_training_checkpoint,
    save_checkpoint as save_training_checkpoint,
)


class BaseTrainer:
    """
    Minimal reusable trainer.

    The trainer is independent of the concrete objective. It only requires:

        loss_output = objective.compute_loss(model, batch)

    State semantics
    ---------------
    self.epoch:
        Index of the next epoch to train.

    self.global_step:
        Number of completed optimizer updates.
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
    ) -> None:
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

        self.config = dict(config) if config is not None else {}

        # self.epoch means the next epoch to train.
        self.epoch = 0

        # Number of completed optimizer updates.
        self.global_step = 0

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
            enabled=(
                self.use_amp
                and self.mixed_precision == "fp16"
            )
        )

        if self.config:
            self._save_config(self.config)

    def train(
        self,
        num_epochs: int,
        max_steps: int | None = None,
    ) -> None:
        """
        Train until self.epoch reaches num_epochs.

        Parameters
        ----------
        num_epochs:
            Total target number of epochs, not the number of additional
            epochs.

            For example, after resuming at self.epoch == 3,
            train(num_epochs=10) trains epochs 3, ..., 9.

        max_steps:
            Optional upper bound on the total global step.
        """
        if num_epochs < self.epoch:
            raise ValueError(
                f"num_epochs={num_epochs} is smaller than the current "
                f"resume epoch self.epoch={self.epoch}."
            )

        self.model.train()
        self.objective.train()

        for epoch in range(self.epoch, num_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for raw_batch in pbar:
                batch = self._prepare_batch(raw_batch)

                metrics = self._train_step(batch)

                pbar.set_postfix(metrics)

                if self.global_step % self.log_every == 0:
                    self._print_metrics(epoch, metrics)

                if self.global_step % self.save_every == 0:
                    # This checkpoint is taken inside an epoch.
                    # We do not claim that the epoch is completed.
                    self.save_checkpoint(
                        self.ckpt_dir
                        / f"step_{self.global_step}.pt",
                        next_epoch=epoch,
                        extra={
                            "checkpoint_type": "step",
                            "current_epoch": epoch,
                        },
                    )

                if (
                    max_steps is not None
                    and self.global_step >= max_steps
                ):
                    # Since the epoch is incomplete, resume from the
                    # beginning of the same epoch.
                    self.epoch = epoch

                    self.save_checkpoint(
                        self.ckpt_dir / "last.pt",
                        next_epoch=self.epoch,
                        extra={
                            "checkpoint_type": "interrupted",
                            "current_epoch": epoch,
                        },
                    )
                    return

            # The entire epoch has completed.
            self.epoch = epoch + 1

            self.save_checkpoint(
                self.ckpt_dir / "last.pt",
                next_epoch=self.epoch,
                extra={
                    "checkpoint_type": "epoch_end",
                    "completed_epoch": epoch,
                },
            )

        # At this point self.epoch == num_epochs.
        self.save_checkpoint(
            self.ckpt_dir / "last.pt",
            next_epoch=self.epoch,
            extra={
                "checkpoint_type": "training_end",
                "completed_epochs": self.epoch,
            },
        )

    def _train_step(
        self,
        batch: dict[str, Any],
    ) -> dict[str, float]:
        """
        Execute one optimizer update and return scalar metrics.
        """
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(
            enabled=self.use_amp,
            dtype=self.amp_dtype,
        ):
            loss_output = self.objective.compute_loss(
                model=self.model,
                batch=batch,
            )
            loss = loss_output.loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at global_step={self.global_step}: "
                f"{loss.detach().item()}"
            )

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

        # Ensure that the principal loss is always logged.
        metrics.setdefault(
            "loss",
            float(loss.detach().cpu()),
        )

        return metrics

    def _prepare_batch(
        self,
        raw_batch: Any,
    ) -> dict[str, Any]:
        """
        Convert common DataLoader outputs into the unified dict format.

        Supported inputs
        ----------------
        dict:
            Kept as a mapping.

        tuple/list of length 2:
            Converted to {"x": x, "cond": y}.

        tuple/list of length 1:
            Converted to {"x": x}.

        tensor or another object:
            Converted to {"x": raw_batch}.
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
                raise ValueError(
                    "Unsupported batch tuple/list format: "
                    f"length={len(raw_batch)}."
                )

        else:
            batch = {
                "x": raw_batch,
            }

        batch_on_device: dict[str, Any] = {}

        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_on_device[key] = value.to(
                    self.device,
                    non_blocking=True,
                )
            else:
                batch_on_device[key] = value

        return batch_on_device

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        next_epoch: int | None = None,
        extra: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save all states required to resume training.

        Parameters
        ----------
        next_epoch:
            Epoch index that should be used when training resumes.
            Defaults to self.epoch.
        """
        if next_epoch is None:
            next_epoch = self.epoch

        return save_training_checkpoint(
            path=path,
            model=self.model,
            objective=self.objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=next_epoch,
            global_step=self.global_step,
            config=self.config,
            extra=extra,
            is_best=is_best,
            best_path=self.ckpt_dir / "best.pt",
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        strict_model: bool = True,
        strict_objective: bool = True,
        restore_rng: bool = True,
    ) -> ResumeState:
        """
        Restore model, objective, optimizer, scheduler, scaler, RNG, and
        trainer counters.
        """
        resume_state = load_training_checkpoint(
            path=path,
            model=self.model,
            objective=self.objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
            strict_model=strict_model,
            strict_objective=strict_objective,
            restore_rng=restore_rng,
        )

        self.epoch = resume_state.epoch
        self.global_step = resume_state.global_step

        # Prefer the saved checkpoint configuration. This makes the resumed
        # run retain the original experiment metadata.
        if resume_state.config:
            self.config = resume_state.config

        return resume_state

    def _print_metrics(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        metric_str = " | ".join(
            f"{key}: {value:.6f}"
            for key, value in metrics.items()
        )

        print(
            f"[epoch={epoch + 1} "
            f"step={self.global_step}] "
            f"{metric_str}"
        )

    def _save_config(
        self,
        config: dict[str, Any],
    ) -> None:
        config_path = self.output_dir / "config.json"

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                config,
                file,
                indent=2,
                ensure_ascii=False,
            )