import json
import tempfile
import unittest
from pathlib import Path

import torch

import scripts.sample as sample_script


class SampleScriptTests(unittest.TestCase):
    def test_config_json_takes_precedence_over_legacy_args_json(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)

            (output_dir / "config.json").write_text(
                json.dumps({"source": "config.json"}),
                encoding="utf-8",
            )
            (output_dir / "args.json").write_text(
                json.dumps({"source": "args.json"}),
                encoding="utf-8",
            )

            config = sample_script.load_config_from_checkpoint_dir(
                checkpoint_dir / "last.pt"
            )

        self.assertEqual(config["source"], "config.json")

    def test_load_initial_noise_preserves_saved_tensor(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            noise_path = Path(temporary_dir) / "initial_noise.pt"
            expected = torch.randn(2, 1, 4, 4)
            torch.save(expected, noise_path)

            actual = sample_script.load_initial_noise(
                path=noise_path,
                shape=tuple(expected.shape),
                device=torch.device("cpu"),
            )

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
