import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from rf_diffusion.inference import model_runners


class _TinyRFScore(torch.nn.Module):
    """Small stand-in that preserves the model/dropout hierarchy."""

    def __init__(self):
        super().__init__()
        self.standard_dropout = torch.nn.Dropout(p=0.5)
        self.broadcast_dropout = model_runners.rf2aa.util_module.Dropout(p_drop=0.5)
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.loaded_state_dict = state_dict


class TestSamplerModelMode(unittest.TestCase):
    def test_load_model_sets_entire_model_to_eval(self):
        sampler = model_runners.Sampler.__new__(model_runners.Sampler)
        sampler.device = torch.device("cpu")
        sampler._conf = OmegaConf.create(
            {
                "inference": {
                    "ckpt_path": "/unused/test-checkpoint.pt",
                    "state_dict_to_load": "final_state_dict",
                }
            }
        )
        checkpoint = {
            "conf": OmegaConf.create({}),
            "final_state_dict": {},
        }
        base_training_conf = OmegaConf.create(
            {
                "diffuser": {},
                "rf": {"model": {}},
            }
        )
        tiny_model = _TinyRFScore()
        noisers_stub = mock.Mock()
        noisers_stub.get.return_value = object()

        with (
            mock.patch.object(model_runners.du, "read_pkl", return_value=checkpoint),
            mock.patch.object(model_runners.OmegaConf, "load", return_value=base_training_conf),
            mock.patch.object(
                model_runners.config_format,
                "translate_obsolete_weight_options",
                side_effect=lambda value: value,
            ),
            mock.patch.object(model_runners.config_format, "alert_obsolete_options"),
            mock.patch.object(model_runners, "noisers", noisers_stub),
            mock.patch.object(model_runners, "RFScore", return_value=tiny_model),
        ):
            sampler.load_model()

        self.assertIs(sampler.model, tiny_model)
        self.assertEqual(tiny_model.loaded_state_dict, {})
        self.assertTrue(all(not module.training for module in tiny_model.modules()))

        inputs = torch.ones((4, 4))
        torch.testing.assert_close(tiny_model.standard_dropout(inputs), inputs)
        torch.testing.assert_close(tiny_model.broadcast_dropout(inputs), inputs)


if __name__ == "__main__":
    unittest.main()
