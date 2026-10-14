from __future__ import annotations


def resolve_kid_subset_size(
    *,
    requested: int,
    generated_count: int,
    reference_count: int,
) -> int:
    """Return a valid KID subset size while keeping the adjustment explicit."""

    if requested < 2:
        raise ValueError("KID subset size must be at least 2.")
    available = min(generated_count, reference_count)
    if available < 2:
        raise ValueError("KID requires at least two generated and real samples.")
    return min(requested, available)
