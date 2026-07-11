import pickle
import unittest
from unittest import mock

import scripts.train as train_script


class PickleableDataset:
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform


class TrainScriptTests(unittest.TestCase):
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
