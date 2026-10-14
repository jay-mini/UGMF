from __future__ import annotations


def validate_prc_settings(
    *,
    neighborhood: int,
    generated_count: int,
    reference_count: int,
    batch_size: int,
) -> None:
    """Validate k-NN manifold Precision/Recall settings."""

    if neighborhood <= 0:
        raise ValueError("PR neighborhood must be positive.")
    if neighborhood >= min(generated_count, reference_count):
        raise ValueError(
            "PR neighborhood must be smaller than both sample counts."
        )
    if batch_size <= 0:
        raise ValueError("PR batch size must be positive.")
