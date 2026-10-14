from __future__ import annotations

from typing import Any

from ugm.objectives.consistency import ConsistencyDistillationObjective
from ugm.trainers.base_trainer import BaseTrainer


class ConsistencyDistillationTrainer(BaseTrainer):
    """Base trainer plus an EMA target update after successful student steps."""

    objective: ConsistencyDistillationObjective

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        step_before = self.global_step
        metrics = super()._train_step(batch)
        if self.global_step > step_before:
            updated = self.objective.update_target(
                self.model,
                global_step=self.global_step,
            )
            metrics["target_updated"] = float(updated)
            metrics["target_updates"] = float(
                self.objective.num_target_updates.detach().cpu()
            )
        return metrics
