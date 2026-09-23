import os
import tempfile
import unittest
from functools import partial
from unittest import mock

import aa_model
import hydra
import pytest
import run_inference
import test_utils
import torch
import torch.distributed
import train_multi_deep
from hydra import compose, initialize
from icecream import ic

from rf2aa import tensor_util
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule

REWRITE = False
pytestmark = pytest.mark.cpu_only
class CallException(Exception):
    pass

class TestDistributed(unittest.TestCase):

    def test_distributed_teardown(self):
        """
        This test passes, but destroy_process_group does not work in practice, so we set the MASTER_PORT explicitly.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12317'
        ic('start 1')
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        ic('end 1')
        torch.distributed.destroy_process_group()
        ic('start 2')
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        ic('end 2')
        torch.distributed.destroy_process_group()

class ModelInputs(unittest.TestCase):

    def setUp(self) -> None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().setUp()

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        torch.distributed.destroy_process_group()
        return super().setUp()

    def test_self_cond_tp1(self):
        os.environ['MASTER_PORT'] = '12320'
        run_regression(self,
                        ['prob_self_cond=1.0',
                        '+dataloader.ligands_to_remove=[]', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                        '+dataloader.min_metal_contacts=0', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                       ],
                       'model_input_self_cond_tp1')

    def test_self_cond_t(self):
        os.environ['MASTER_PORT'] = '12321'
        run_regression(self,
                       ['prob_self_cond=1.0',
                        '+dataloader.ligands_to_remove=[]', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                        '+dataloader.min_metal_contacts=0', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                       ],
                       'model_input_self_cond_t',
                       call_number=2)

    def test_atomize_regression(self):
        os.environ['MASTER_PORT'] = '12326'
        run_regression(self,
                       [
                            'dataloader.DIFF_MASK_PROBS=null',
                            '+dataloader.DIFF_MASK_PROBS.get_sm_contacts=1.0',
                            '+dataloader.ligands_to_remove=[]', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                            '+dataloader.min_metal_contacts=0', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                        ],
                        'model_input_atomize',
                        call_number=1)

# Example regression test, with one faked return value from the model.
def run_regression(self, overrides, golden_name, call_number=1, assert_loss=False):
    # This test must be run on a CPU.
    assert torch.cuda.device_count() == 0
    run_inference.make_deterministic()

    with tempfile.TemporaryDirectory() as tmpdir:
        conf = construct_conf(overrides + [f'rundir={tmpdir}'])
        train = train_multi_deep.make_trainer(conf)
        fake_forward = mock.patch.object(LegacyRoseTTAFoldModule, "__call__", autospec=True)
        with fake_forward as mock_forward:
            def side_effect(*args, **kwargs):
                ic(side_effect.call_count)
                side_effect.call_count += 1
                if side_effect.call_count < call_number:
                    ic(kwargs.keys())
                    if 'xyz' in kwargs:
                        xyz = kwargs['xyz']
                    else:
                        xyz = args[4] # [1, L, 36 3]
                    xyz[:,:,14:,:] = 0.0
                    L = xyz.shape[1]
                    px0_xyz = xyz[None,:,:,:3].repeat(40, 1, 1, 1, 1)
                    px0_xyz = torch.normal(0, 1, px0_xyz.shape) + px0_xyz
                    logits = (
                        torch.normal(0, 1, (1, 61, L, L)),
                        torch.normal(0, 1, (1, 61, L, L)),
                        torch.normal(0, 1, (1, 37, L, L)),
                        torch.normal(0, 1, (1, 19, L, L)),
                    )
                    logits_aa = torch.normal(0, 1, (1, 80, L))
                    logits_pae = torch.normal(0, 1, (1, 64, L, L))
                    logits_pde = torch.normal(0, 1, (1, 64, L, L))
                    p_bind = torch.normal(0, 1, (1, 1))
                    alpha_s = torch.normal(0, 1, (40, 1, L, 20, 2))
                    xyz_allatom = torch.normal(0, 1, (1, L, 36, 3))
                    lddt = torch.normal(0, 1, (1, 50, L))
                    ic(xyz_allatom.requires_grad)
                    xyz_allatom.requires_grad = True
                    quat = torch.normal(0, 1, (1, 40, L, 4))
                    quat = quat / quat.norm(dim=-1)[...,None]
                    side_effect.rfo = aa_model.RFO(logits, logits_aa, logits_pae, logits_pde, p_bind, px0_xyz, alpha_s, xyz_allatom, lddt, None, None, None, quat)
                    side_effect.rfo = tensor_util.to_ordered_dict(side_effect.rfo)
                    tensor_util.require_grad(side_effect.rfo)
                    return side_effect.rfo.values()
                else:
                    raise CallException('shouldnt raise')
            side_effect.call_count = 0
            mock_forward.side_effect = side_effect
            train.group_name = golden_name
            try:
                train.run_model_training(torch.cuda.device_count())
            except CallException as e:
                print("CalledException", e)

            args, kwargs = mock_forward.call_args
            cmp = partial(tensor_util.cmp, atol=1e-3, rtol=0)
            # Uncomment to peek at what parts of sequence are masked
            # ic(torch.argmax(argument_map['msa_latent'], dim=-1))
            test_utils.assert_matches_golden(self, golden_name, kwargs, rewrite=REWRITE, custom_comparator=cmp)

def construct_conf(overrides):
    # overrides = overrides + ['inference.cautious=False', 'inference.design_startnum=0']
    hydra.core.global_hydra.GlobalHydra().clear()
    initialize(version_base=None, config_path="config/training", job_name="test_app")
    conf = compose(config_name='debug.yaml', overrides=overrides, return_hydra_config=True)
    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
    # HydraConfig.instance().set_config(conf)
    # conf = compose(config_name='aa_small.yaml', overrides=overrides)
    return conf

class Loss(unittest.TestCase):
    @pytest.mark.noparallel
    def test_loss_grad(self):
        os.environ['MASTER_PORT'] = '12400'
        self.run_regression_loss_grad([
                                        '+dataloader.ligands_to_remove=[]', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                                        '+dataloader.min_metal_contacts=0', # Remove this line when the subsampled dataset is remade, grep for this to find others #$%^
                                        ],
                                        'loss_grad_no_self_cond',
                                        call_number=2,
                                        assert_loss=True)

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        torch.distributed.destroy_process_group()
        return super().tearDown()

    # Example regression test, with one faked return value from the model.
    def run_regression_loss_grad(self, overrides, golden_name, call_number=1, assert_loss=False):
        # This test must be run on a CPU.
        assert torch.cuda.device_count() == 0
        run_inference.make_deterministic()
        with tempfile.TemporaryDirectory() as tmpdir:
            conf = construct_conf(overrides + ['experiment.gamma=0.999', f'rundir={tmpdir}'])
            train = train_multi_deep.make_trainer(conf)
            fake_forward = mock.patch.object(LegacyRoseTTAFoldModule, "__call__", autospec=True)
            a = mock.patch.object(torch.cuda.amp.GradScaler, "scale", autospec=True)
            with fake_forward as mock_forward:
                with a as b:
                    def side_effect(*args, **kwargs):
                        side_effect.call_count += 1
                        if side_effect.call_count < call_number:
                            ic(kwargs.keys())
                            if 'xyz' in kwargs:
                                xyz = kwargs['xyz']
                            else:
                                xyz = args[4] # [1, L, 36 3]
                            L = xyz.shape[1]
                            px0_xyz = xyz[None,:,:,:3].repeat(40, 1, 1, 1, 1)
                            px0_xyz = torch.normal(0, 1, px0_xyz.shape) + px0_xyz
                            logits = (
                                torch.normal(0, 1, (1, 61, L, L)),
                                torch.normal(0, 1, (1, 61, L, L)),
                                torch.normal(0, 1, (1, 37, L, L)),
                                torch.normal(0, 1, (1, 19, L, L)),
                            )
                            logits_aa = torch.normal(0, 1, (1, 80, L))
                            logits_pae = torch.normal(0, 1, (1, 64, L, L))
                            logits_pde = torch.normal(0, 1, (1, 64, L, L))
                            p_bind = torch.normal(0, 1, (1, 1))
                            alpha_s = torch.normal(0, 1, (40, 1, L, 20, 2))
                            xyz_allatom = torch.normal(0, 1, (1, L, 36, 3))
                            lddt = torch.normal(0, 1, (1, 50, L))
                            ic(xyz_allatom.requires_grad)
                            xyz_allatom.requires_grad = True
                            quat = torch.normal(0, 1, (1, 40, L, 4))
                            quat = quat / quat.norm(dim=-1)[...,None]
                            side_effect.rfo = aa_model.RFO(logits, logits_aa, logits_pae, logits_pde, p_bind, px0_xyz, alpha_s,     xyz_allatom, lddt, None, None, None, quat)
                            side_effect.rfo = tensor_util.to_ordered_dict(side_effect.rfo)
                            tensor_util.require_grad(side_effect.rfo)
                            return side_effect.rfo.values()
                        else:
                            raise CallException('shouldnt raise')
                    side_effect.call_count = 0
                    mock_forward.side_effect = side_effect
                    def c(*args, **kwargs):
                        raise CallException("should raise")
                    b.side_effect = c
                    train.group_name = golden_name
                    try:
                        train.run_model_training(torch.cuda.device_count())
                    except CallException as e:
                        print("CalledException", e)
                    # ic(b.call_args)
                    # ic(len(b.call_args))
                    # ic(type(b.call_args))

                    (_, loss,), _ = b.call_args
                    ic(loss)
                    rfo = side_effect.rfo
                    run_inference.seed_all()
                    loss.backward()
                    grads = tensor_util.get_grad(rfo)
                    print(f'grad shapes: {tensor_util.info(grads)}')
                    print(f'grad (min, max): {tensor_util.minmax(grads)}')
                    print(f'loss: {loss}')
                    cmp = partial(tensor_util.cmp, atol=5e-5, rtol=1e-2)
                    # More stringent for running on the same architecture.
                    # cmp = partial(tensor_util.cmp, atol=1e-20, rtol=1e-5)
                    test_utils.assert_matches_golden(self, golden_name, grads, rewrite=REWRITE, custom_comparator=cmp)

if __name__ == '__main__':
        unittest.main()
