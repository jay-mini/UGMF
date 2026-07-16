from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@ dataclass(frozen=True)

class ResumeState:
    """
    Training metadata returned after loading a checkpoint.

    Attributes
    ----------

    epoch:
        The epoch stored in the checkpoint.

    global_step:
        The total number of optimizer updates stored in the checkpoint.

    config:
        The configuration dictionary stored in the checkpoint.

    extra:
        Additional user-defined information.

    checkpoint_path:
        Path from which the checkpoint was loaded.
    """

    epoch: int
    global_step: int
    config: dict[str, Any]
    extra: dict[str, Any]
    checkpoint_path: Path

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying model if it is wrapped by DataParallel or DDP.

    For an ordinary nn.Module, the model is returned unchanged.
    """
    if hasattr(model, "module"):
        module = getattr(model, "module")

        if isinstance(module, nn.Module):
            return module
        
    return model

def capture_rng_state() -> dict[str, Any]:
    """
    Capture the current random-number-generator states.

    This includes:
    - Python random
    - Numpy
    - PyTorch CPU
    - all visible CUDA devices
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

def restore_rng_state(
    rng_state: Mapping[str, Any] | None,
) -> None:
    """
    Restore random-number-generator states from a checkpoint.

    Missing RNG fields are ignored to preserve compatibility with older checkpoints.
    """
    if not rng_state:
        return
    
    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())

    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        saved_device_count = len(cuda_state)
        current_device_count = torch.cuda.device_count()

        if saved_device_count == current_device_count:
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            # Restore as many device states as possible when the number of
            # visible GPUs differs between saving and loading.
            for device_index, state in enumerate(
                cuda_state[:current_device_count]
            ):
                torch.cuda.set_rng_state(
                    state,
                    device=device_index,
                )

def build_checkpoint(
    *,
    model: nn.Module,
    objective: nn.Module | None,
    optimizer: Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construct a complete training checkpoint dictionary.

    Parameters
    ----------
    model:
        Any PyTorch nn.Module.

    objective:
        Training objective. It may contain trainable parameters or registered
        buffers. Pass None if the objective has no state to save.

    optimizer:
        Optimizer used during training.

    scheduler:
        Learning-rate scheduler, or None.

    scaler:
        AMP gradient scaler, or None.

    epoch:
        Epoch associated with this checkpoint.

    global_step:
        Number of completed optimizer updates.

    config:
        Serializable experiment configuration.

    extra:
        Additional serializable metadata, such as best validation loss.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}.")

    if global_step < 0:
        raise ValueError(
            f"global_step must be non-negative, got {global_step}."
        )

    raw_model = unwrap_model(model)

    return {
        "format_version": 1,
        "model": raw_model.state_dict(),
        "objective": (
            objective.state_dict()
            if objective is not None
            else None
        ),
        "optimizer": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": dict(config) if config is not None else {},
        "extra": dict(extra) if extra is not None else {},
        "rng_state": capture_rng_state(),
    }


def atomic_torch_save(
    checkpoint: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """
    Save a checkpoint atomically.

    The checkpoint is first written to a temporary file and then moved to the
    destination path. This reduces the chance of leaving a corrupted final
    checkpoint if training is interrupted during saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        torch.save(dict(checkpoint), temporary_path)

        # os.replace is atomic when source and destination are on the same
        # filesystem.
        os.replace(temporary_path, path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    return path

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    objective: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    is_best: bool = False,
    best_path: str | Path | None = None,
) -> Path:
    """
    Build and save a general training checkpoint.

    If is_best=True, the saved checkpoint is also copied to best_path.
    """
    checkpoint = build_checkpoint(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        config=config,
        extra=extra,
    )

    saved_path = atomic_torch_save(checkpoint, path)

    if is_best:
        if best_path is None:
            best_path = saved_path.with_name("best.pt")

        best_path = Path(best_path)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_path, best_path)

    return saved_path

def _load_state_if_available(
    *,
    object_name: str,
    obj: Any | None,
    state: Any | None,
    strict: bool = True,
) -> None:
    """
    Load one component state if both the target object and saved state exist.
    """
    if obj is None:
        return

    if state is None:
        raise KeyError(
            f"The checkpoint does not contain state for {object_name!r}, "
            f"but a {object_name} object was provided."
        )

    if isinstance(obj, nn.Module):
        obj.load_state_dict(state, strict=strict)
    else:
        obj.load_state_dict(state)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    objective: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device | Mapping[str, str] | None = "cpu",
    strict_model: bool = True,
    strict_objective: bool = True,
    restore_rng: bool = True,
) -> ResumeState:
    """
    Load a checkpoint and restore training state.

    The caller must first construct the model, objective, optimizer, scheduler,
    and scaler with architectures/settings compatible with the checkpoint.

    Parameters
    ----------
    path:
        Checkpoint file.

    model:
        Model instance into which parameters are loaded.

    objective:
        Objective instance into which parameters and buffers are loaded.

    optimizer:
        Optimizer instance whose internal momentum/state will be restored.

    scheduler:
        Scheduler instance whose state will be restored.

    scaler:
        AMP GradScaler whose state will be restored.

    map_location:
        Device mapping passed to torch.load. Loading onto CPU first is usually
        the safest default.

    strict_model:
        Passed to model.load_state_dict.

    strict_objective:
        Passed to objective.load_state_dict.

    restore_rng:
        Restore Python, NumPy, PyTorch CPU, and CUDA RNG states.

    Returns
    -------
    ResumeState
        Epoch, global step, configuration, and additional metadata.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    # weights_only=False is required because this general checkpoint contains
    # non-tensor Python objects, including NumPy and Python RNG states.
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Expected checkpoint to be a dictionary, "
            f"but got {type(checkpoint).__name__}."
        )

    if "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint {path} does not contain a 'model' state."
        )

    raw_model = unwrap_model(model)

    _load_state_if_available(
        object_name="model",
        obj=raw_model,
        state=checkpoint["model"],
        strict=strict_model,
    )

    if objective is not None:
        _load_state_if_available(
            object_name="objective",
            obj=objective,
            state=checkpoint.get("objective"),
            strict=strict_objective,
        )

    if optimizer is not None:
        _load_state_if_available(
            object_name="optimizer",
            obj=optimizer,
            state=checkpoint.get("optimizer"),
        )

    if scheduler is not None:
        _load_state_if_available(
            object_name="scheduler",
            obj=scheduler,
            state=checkpoint.get("scheduler"),
        )

    if scaler is not None:
        _load_state_if_available(
            object_name="scaler",
            obj=scaler,
            state=checkpoint.get("scaler"),
        )

    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    return ResumeState(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        config=dict(checkpoint.get("config", {})),
        extra=dict(checkpoint.get("extra", {})),
        checkpoint_path=path,
    )
