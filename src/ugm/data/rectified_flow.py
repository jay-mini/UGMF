from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


PAIR_DATASET_FORMAT = "ugmf.rectified_flow_pairs"
PAIR_DATASET_VERSION = 1


def load_pair_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Rectified-flow manifest not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != PAIR_DATASET_FORMAT:
        raise ValueError(f"Unsupported paired-data format in {path}.")
    if int(manifest.get("version", -1)) != PAIR_DATASET_VERSION:
        raise ValueError(f"Unsupported paired-data version in {path}.")
    return manifest


class RectifiedFlowPairDataset(Dataset):
    """Lazy, manifest-backed dataset of ``(noise, teacher output)`` pairs."""

    def __init__(self, root: str | Path, verify: bool = True) -> None:
        self.root = Path(root).resolve()
        self.manifest = load_pair_manifest(self.root)
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError("The paired dataset contains no shards.")

        self._ends: list[int] = []
        running_total = 0
        expected_start = 0
        for shard in self.shards:
            start = int(shard["start"])
            count = int(shard["count"])
            if start != expected_start or count <= 0:
                raise ValueError("Paired-data shards must be contiguous and non-empty.")
            shard_path = self.root / str(shard["file"])
            if verify and not shard_path.is_file():
                raise FileNotFoundError(f"Paired-data shard not found: {shard_path}")
            running_total += count
            expected_start += count
            self._ends.append(running_total)

        self.num_samples = int(self.manifest.get("num_samples", -1))
        if self.num_samples != running_total:
            raise ValueError(
                "Manifest num_samples does not equal the sum of shard counts."
            )
        self._cached_shard_index: int | None = None
        self._cached_shard: dict[str, torch.Tensor] | None = None

        if verify:
            for shard_index in range(len(self.shards)):
                payload = self._load_shard(shard_index)
                self._validate_payload(payload, int(self.shards[shard_index]["count"]))
            self._cached_shard_index = None
            self._cached_shard = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        payload = self._load_shard(shard_index)
        local_index = index - shard_start
        item = {
            "x0": payload["x0"][local_index],
            "x1": payload["x1"][local_index],
        }
        if "cond" in payload:
            item["cond"] = payload["cond"][local_index]
        return item

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._cached_shard_index != shard_index:
            path = self.root / str(self.shards[shard_index]["file"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a tensor dictionary in {path}.")
            self._cached_shard_index = shard_index
            self._cached_shard = payload
        assert self._cached_shard is not None
        return self._cached_shard

    @staticmethod
    def _validate_payload(payload: dict[str, torch.Tensor], count: int) -> None:
        if "x0" not in payload or "x1" not in payload:
            raise KeyError("Every pair shard must contain x0 and x1.")
        x0, x1 = payload["x0"], payload["x1"]
        if not torch.is_tensor(x0) or not torch.is_tensor(x1):
            raise TypeError("x0 and x1 must be tensors.")
        if x0.shape != x1.shape or x0.shape[0] != count:
            raise ValueError("Shard x0/x1 shapes do not match its declared count.")
        if not x0.is_floating_point() or not x1.is_floating_point():
            raise TypeError("Shard endpoints must be floating-point tensors.")
        if not torch.isfinite(x0).all() or not torch.isfinite(x1).all():
            raise ValueError("Shard endpoints contain non-finite values.")
        cond = payload.get("cond")
        if cond is not None:
            if not torch.is_tensor(cond) or cond.shape != (count,):
                raise ValueError("Shard cond must have shape [count].")
            if cond.is_floating_point() or cond.is_complex():
                raise TypeError("Shard cond must use an integer dtype.")


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shards and their contents while retaining disk locality.

    A fully random index permutation makes a lazy sharded dataset repeatedly
    reload large files.  This sampler still visits every item once per epoch,
    but emits all shuffled indices from one randomly selected shard before
    moving to the next shard.
    """

    def __init__(
        self,
        dataset: RectifiedFlowPairDataset,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.dataset = dataset
        self.generator = generator

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        shard_order = torch.randperm(
            len(self.dataset.shards),
            generator=self.generator,
        ).tolist()
        for shard_index in shard_order:
            start = 0 if shard_index == 0 else self.dataset._ends[shard_index - 1]
            end = self.dataset._ends[shard_index]
            local_order = torch.randperm(
                end - start,
                generator=self.generator,
            ).tolist()
            for local_index in local_order:
                yield start + local_index
