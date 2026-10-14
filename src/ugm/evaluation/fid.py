from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .kid import resolve_kid_subset_size
from .precision_recall import validate_prc_settings


STANDARD_BACKEND = "torch-fidelity"
FID_KID_FEATURE_EXTRACTOR = "inception-v3-compat"
PRC_FEATURE_EXTRACTOR = "vgg16"


def image_features(
    images: torch.Tensor,
    feature_size: int = 7,
) -> torch.Tensor:
    """Return the project's lightweight grayscale pixel features."""

    images = images.detach().float().cpu()
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W].")
    if images.shape[1] != 1:
        images = images.mean(dim=1, keepdim=True)
    features = F.adaptive_avg_pool2d(images, (feature_size, feature_size))
    return features.flatten(start_dim=1).double()


def _mean_and_covariance(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.shape[0] < 2:
        raise ValueError("At least two samples are required for a covariance.")
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    return mean, covariance


def _symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def pixel_frechet_distance(
    generated_features: torch.Tensor,
    reference_features: torch.Tensor,
) -> float:
    """Fréchet distance over 7x7 grayscale pixels, not standard FID."""

    generated_mean, generated_cov = _mean_and_covariance(generated_features)
    reference_mean, reference_cov = _mean_and_covariance(reference_features)

    if generated_cov.shape != reference_cov.shape:
        raise ValueError("Generated and reference feature sizes must match.")
    eps = 1e-6
    identity = torch.eye(generated_cov.shape[0], dtype=torch.float64)
    generated_cov = generated_cov + eps * identity
    reference_cov = reference_cov + eps * identity

    sqrt_generated_cov = _symmetric_matrix_sqrt(generated_cov)
    middle = sqrt_generated_cov @ reference_cov @ sqrt_generated_cov
    covariance_mean = _symmetric_matrix_sqrt((middle + middle.T) * 0.5)
    mean_difference = generated_mean - reference_mean
    distance = (
        mean_difference.dot(mean_difference)
        + torch.trace(generated_cov)
        + torch.trace(reference_cov)
        - 2.0 * torch.trace(covariance_mean)
    )
    return float(distance.clamp_min(0.0).item())


def _png_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise ValueError(f"No PNG samples found in {directory}.")
    return paths


def load_pixel_features(
    directory: Path,
    feature_size: int = 7,
    batch_size: int = 512,
) -> torch.Tensor:
    """Load exported PNG files and compute small pixel features in batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paths = _png_paths(directory)
    feature_batches: list[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"), dtype=np.float32).copy()
            tensors.append(torch.from_numpy(array).unsqueeze(0) / 255.0)
        feature_batches.append(
            image_features(torch.stack(tensors), feature_size=feature_size)
        )
    return torch.cat(feature_batches, dim=0)


def calculate_pixel_fd_proxy(
    generated_dir: Path,
    reference_dir: Path,
    feature_size: int = 7,
) -> dict[str, Any]:
    """Calculate the explicitly non-standard low-resolution pixel-FD proxy."""

    generated_features = load_pixel_features(generated_dir, feature_size)
    reference_features = load_pixel_features(reference_dir, feature_size)
    return {
        "name": "pixel_fd_proxy",
        "standard_metric": False,
        "value": pixel_frechet_distance(
            generated_features,
            reference_features,
        ),
        "feature_definition": (
            f"{feature_size}x{feature_size} adaptive-average-pooled grayscale "
            "pixels"
        ),
        "generated_samples": int(generated_features.shape[0]),
        "reference_samples": int(reference_features.shape[0]),
        "interpretation": (
            "Lower is better. This is a fast project-local proxy and must not "
            "be reported as Inception FID."
        ),
    }


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    return value


def calculate_standard_metrics(
    *,
    generated_dir: Path,
    reference_dir: Path,
    device: torch.device,
    fid: bool,
    kid: bool,
    prc: bool,
    batch_size: int,
    kid_subsets: int,
    kid_subset_size: int,
    prc_neighborhood: int,
    prc_batch_size: int,
    seed: int,
    verbose: bool = False,
    calculate_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the validated torch-fidelity implementations in one shared pass."""

    if not any((fid, kid, prc)):
        return {}
    if batch_size <= 0:
        raise ValueError("metric batch_size must be positive.")

    generated_count = len(_png_paths(generated_dir))
    reference_count = len(_png_paths(reference_dir))
    if min(generated_count, reference_count) < 2:
        raise ValueError("Standard metrics require at least two image samples.")

    effective_kid_subset_size = None
    if kid:
        effective_kid_subset_size = resolve_kid_subset_size(
            requested=kid_subset_size,
            generated_count=generated_count,
            reference_count=reference_count,
        )
    if prc:
        validate_prc_settings(
            neighborhood=prc_neighborhood,
            generated_count=generated_count,
            reference_count=reference_count,
            batch_size=prc_batch_size,
        )

    if calculate_fn is None:
        try:
            torch_fidelity = importlib.import_module("torch_fidelity")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Standard FID/KID/PR requires torch-fidelity. Install the "
                "evaluation extra with: pip install -e '.[eval]'"
            ) from error
        calculate_fn = torch_fidelity.calculate_metrics
        backend_version = str(torch_fidelity.__version__)
    else:
        backend_version = "test-double"

    raw = calculate_fn(
        input1=str(generated_dir),
        input2=str(reference_dir),
        cuda=device.type == "cuda",
        batch_size=batch_size,
        isc=False,
        fid=fid,
        kid=kid,
        prc=prc,
        kid_subsets=kid_subsets,
        kid_subset_size=effective_kid_subset_size or kid_subset_size,
        prc_neighborhood=prc_neighborhood,
        prc_batch_size=prc_batch_size,
        samples_shuffle=False,
        cache=False,
        rng_seed=seed,
        save_cpu_ram=True,
        verbose=verbose,
    )
    raw = _to_python(raw)

    result: dict[str, Any] = {
        "standard_metrics": True,
        "backend": STANDARD_BACKEND,
        "backend_version": backend_version,
        "generated_samples": generated_count,
        "reference_samples": reference_count,
        "raw_backend_output": raw,
    }
    if fid:
        result["fid"] = {
            "value": raw["frechet_inception_distance"],
            "feature_extractor": FID_KID_FEATURE_EXTRACTOR,
            "lower_is_better": True,
        }
    if kid:
        result["kid"] = {
            "mean": raw["kernel_inception_distance_mean"],
            "std": raw["kernel_inception_distance_std"],
            "feature_extractor": FID_KID_FEATURE_EXTRACTOR,
            "subsets": kid_subsets,
            "subset_size_requested": kid_subset_size,
            "subset_size_effective": effective_kid_subset_size,
            "lower_is_better": True,
        }
    if prc:
        result["generative_precision_recall"] = {
            "precision": raw["precision"],
            "recall": raw["recall"],
            "f_score": raw["f_score"],
            "feature_extractor": PRC_FEATURE_EXTRACTOR,
            "neighborhood": prc_neighborhood,
            "interpretation": (
                "Precision estimates generated-sample fidelity; recall "
                "estimates coverage of the real-data manifold."
            ),
        }
    return result
