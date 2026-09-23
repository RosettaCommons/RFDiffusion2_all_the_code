import random

import numpy as np
import torch

from rf_diffusion import data_loader
from rf_diffusion import run_inference


_MASK_GEN_SEED = 7_437_314


class _RngConsumingDataset:
    def __init__(self, draws):
        self.draws = draws

    def __getitem__(self, index):
        del index
        torch.rand(self.draws)
        np.random.random(self.draws)
        for _ in range(self.draws):
            random.random()
        return {
            "item_context": "transform RNG isolation test",
            "mask_gen_seed": _MASK_GEN_SEED,
        }

    def __len__(self):
        return 1


def _sample_from_all_rngs(**features):
    return features | {
        "torch_sample": torch.rand(4),
        "numpy_sample": np.random.random(4),
        "python_sample": [random.random() for _ in range(4)],
    }


def test_transform_rng_is_independent_of_dataset_rng_consumption():
    no_draws = data_loader.TransformedDataset(
        _RngConsumingDataset(draws=0), [_sample_from_all_rngs]
    )[0]
    many_draws = data_loader.TransformedDataset(
        _RngConsumingDataset(draws=37), [_sample_from_all_rngs]
    )[0]

    run_inference.seed_all(_MASK_GEN_SEED)
    expected_torch = torch.rand(4)
    expected_numpy = np.random.random(4)
    expected_python = [random.random() for _ in range(4)]

    torch.testing.assert_close(no_draws["torch_sample"], expected_torch, rtol=0, atol=0)
    np.testing.assert_array_equal(no_draws["numpy_sample"], expected_numpy)
    assert no_draws["python_sample"] == expected_python

    torch.testing.assert_close(many_draws["torch_sample"], expected_torch, rtol=0, atol=0)
    np.testing.assert_array_equal(many_draws["numpy_sample"], expected_numpy)
    assert many_draws["python_sample"] == expected_python
