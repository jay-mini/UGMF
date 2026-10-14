import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import scripts.train as train_script


class PickleableDataset:
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform


class TinyTrainingDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "x": torch.zeros(1, 4, 4),
            "cond": torch.tensor(index % 2),
        }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, cond=None):
        return torch.zeros_like(x) + self.bias


class TrainScriptTests(unittest.TestCase):
    def test_parse_config_applies_cli_over_yaml_over_defaults(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "experiment.yaml"
            config_path.write_text(
                "objective: ddpm\n"
                "dataset: mnist\n"
                "batch_size: 64\n"
                "lr: 0.001\n",
                encoding="utf-8",
            )

            config = train_script.parse_config(
                [
                    "--config",
                    str(config_path),
                    "--batch_size",
                    "32",
                    "--cond_drop_prob",
                    "0.2",
                    "--guidance_scale",
                    "3.0",
                ]
            )

        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["lr"], 0.001)
        self.assertEqual(config["cond_drop_prob"], 0.2)
        self.assertEqual(config["guidance_scale"], 3.0)
        self.assertEqual(config["num_epochs"], 20)
        self.assertEqual(
            config["config_path"],
            str(config_path.resolve()),
        )

    def test_experiment_yaml_builds_matching_objective(self):
        for filename, objective_name in (
            ("ddpm_mnist.yaml", "ddpm"),
            ("fm_mnist.yaml", "fm"),
        ):
            with self.subTest(filename=filename):
                config_path = (
                    train_script.PROJECT_ROOT
                    / "configs"
                    / "experiment"
                    / filename
                )
                config = train_script.parse_config(
                    ["--config", str(config_path)]
                )
                objective = train_script.build_objective(config)

                self.assertEqual(config["objective"], objective_name)

                if objective_name == "ddpm":
                    self.assertEqual(
                        objective.num_timesteps,
                        config["num_timesteps"],
                    )
                    self.assertEqual(
                        objective.prediction_type,
                        config["prediction_type"],
                    )
                else:
                    self.assertEqual(objective.t_min, config["t_min"])
                    self.assertEqual(objective.t_max, config["t_max"])

    def test_main_saves_the_effective_config_in_json_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            config = dict(train_script.DEFAULT_CONFIG)
            config.update(
                {
                    "objective": "ddpm",
                    "dataset": "mnist",
                    "data_dir": "data",
                    "output_dir": str(output_dir),
                    "batch_size": 2,
                    "num_epochs": 1,
                    "max_steps": 1,
                    "num_workers": 0,
                    "num_timesteps": 2,
                    "save_every": 100,
                    "config_path": "test.yaml",
                }
            )

            with (
                mock.patch.object(
                    train_script,
                    "parse_config",
                    return_value=config,
                ),
                mock.patch.object(
                    train_script,
                    "build_dataset",
                    return_value=(TinyTrainingDataset(), 1, 4, 2),
                ),
                mock.patch.object(
                    train_script,
                    "build_model",
                    return_value=TinyModel(),
                ),
            ):
                train_script.main()

            with (output_dir / "config.json").open(
                "r",
                encoding="utf-8",
            ) as file:
                saved_json_config = json.load(file)

            checkpoint = torch.load(
                output_dir / "checkpoints" / "last.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(saved_json_config, checkpoint["config"])
        self.assertEqual(saved_json_config["max_steps"], 1)
        self.assertEqual(saved_json_config["output_dir"], str(output_dir.resolve()))
        self.assertEqual(saved_json_config["in_channels"], 1)
        self.assertEqual(saved_json_config["image_size"], 4)
        self.assertEqual(saved_json_config["scheduler"], "cosine")
        self.assertEqual(saved_json_config["scheduler_total_steps"], 1)
        self.assertIsNotNone(checkpoint["scheduler"])

    def test_scheduler_horizon_does_not_use_max_steps(self):
        parameter = nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        config = dict(train_script.DEFAULT_CONFIG)
        config.update(
            {
                "scheduler": "cosine",
                "scheduler_total_steps": None,
                "scheduler_min_lr": 0.0,
                "num_epochs": 3,
                "max_steps": 1,
            }
        )

        scheduler = train_script.build_scheduler(
            config=config,
            optimizer=optimizer,
            steps_per_epoch=4,
        )

        self.assertIsInstance(
            scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )
        self.assertEqual(config["scheduler_total_steps"], 12)

    def test_resume_compatibility_allows_new_stop_boundary(self):
        saved = dict(train_script.DEFAULT_CONFIG)
        saved["scheduler_total_steps"] = 100
        saved["max_steps"] = 10

        active = dict(saved)
        active["max_steps"] = 20

        train_script.validate_resume_compatibility(active, saved)

        active["lr"] = 1e-3
        with self.assertRaisesRegex(ValueError, "lr"):
            train_script.validate_resume_compatibility(active, saved)

        active = dict(saved)
        active["cond_drop_prob"] = 0.2
        with self.assertRaisesRegex(ValueError, "cond_drop_prob"):
            train_script.validate_resume_compatibility(active, saved)

    def test_build_dataset_returns_pickleable_dataset_for_worker_processes(self):
        patches = (
            mock.patch.object(train_script.datasets, "MNIST", PickleableDataset),
            mock.patch.object(train_script.datasets, "CIFAR10", PickleableDataset),
        )

        with patches[0], patches[1]:
            for dataset_name in ("mnist", "cifar10"):
                with self.subTest(dataset_name=dataset_name):
                    dataset, _, _, _ = train_script.build_dataset(dataset_name, "data")

                    pickle.dumps(dataset)


if __name__ == "__main__":
    unittest.main()
