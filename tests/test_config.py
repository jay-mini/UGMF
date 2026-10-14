import tempfile
import unittest
from pathlib import Path

from ugm.utils.config import (
    DEFAULT_CONFIG,
    load_yaml_config,
    merge_configs,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_load_yaml_config_returns_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "experiment.yaml"
            config_path.write_text(
                "batch_size: 64\nlr: 0.001\n",
                encoding="utf-8",
            )

            config = load_yaml_config(config_path)

        self.assertEqual(config["batch_size"], 64)
        self.assertEqual(config["lr"], 0.001)

    def test_merge_precedence_is_cli_then_yaml_then_defaults(self):
        config = merge_configs(
            defaults={
                "batch_size": 128,
                "lr": 2e-4,
                "num_epochs": 20,
            },
            yaml_config={
                "batch_size": 64,
                "lr": 1e-4,
            },
            cli_overrides={
                "batch_size": 32,
            },
        )

        self.assertEqual(config["batch_size"], 32)
        self.assertEqual(config["lr"], 1e-4)
        self.assertEqual(config["num_epochs"], 20)

    def test_merge_rejects_unknown_yaml_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown YAML"):
            merge_configs(
                defaults={"batch_size": 128},
                yaml_config={"batch_szie": 64},
                cli_overrides={},
            )

    def test_validate_config_rejects_yaml_boolean_no(self):
        config = dict(DEFAULT_CONFIG)
        config["mixed_precision"] = False

        with self.assertRaisesRegex(ValueError, "mixed_precision"):
            validate_config(config)

    def test_validate_config_rejects_invalid_cfg_values(self):
        config = dict(DEFAULT_CONFIG)
        config["cond_drop_prob"] = 1.1
        with self.assertRaisesRegex(ValueError, "cond_drop_prob"):
            validate_config(config)

        config = dict(DEFAULT_CONFIG)
        config["guidance_scale"] = -0.1
        with self.assertRaisesRegex(ValueError, "guidance_scale"):
            validate_config(config)

    def test_validate_config_accepts_dit_tiny_and_checks_attention_shape(self):
        config = dict(DEFAULT_CONFIG)
        config.update(
            {
                "model_type": "dit_tiny",
                "dit_patch_size": 4,
                "dit_hidden_size": 192,
                "dit_depth": 6,
                "dit_num_heads": 3,
                "dit_mlp_ratio": 4.0,
                "image_size": 28,
            }
        )
        validate_config(config)

        config["dit_num_heads"] = 5
        with self.assertRaisesRegex(ValueError, "dit_num_heads"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
