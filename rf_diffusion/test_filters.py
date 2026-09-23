import unittest
import torch
import pytest

pytestmark = pytest.mark.gpu
import numpy as np
import hydra
from rf_diffusion.test_inference import construct_conf
from rf_diffusion.inference import model_runners
from rf_diffusion.inference.filters import (
    init_filters,
    FilterFailedException,
    do_filtering
    )
from rf_diffusion import silent_files


def filter_prepare_indep(overrides):
    '''
    Load a config with overrides and return the indep that it generates (along with sampler.sample_init())

    Args:
        overrides (list[str]): Overrides for the config

    Returns:
        indep (indep): The indep from sample_init()
        extra_args (dict): Other variables to pass to filter_evaluate
    '''
    conf = construct_conf(overrides)

    sampler = model_runners.sampler_selector(conf)
    indep, contig_map, atomizer, t_step_input = sampler.sample_init()

    extra_args = {
        'contig_map':contig_map,
        'atomizer':atomizer,
        't_step_input':t_step_input,
        'sampler':sampler
    }

    return indep, extra_args

def filter_evaluate(indep, sampler=None, px0=None, t=1, it=0, **kwargs):
    '''
    Call do_filtering() on your indep

    Args:
        indep (Indep): indep
        sampler (Sampler): the sampler
        px0 (torch.Tensor[float] or None): The px0 you would like to spoof or none to use indep.xyz
        t (int): the t we are spoofing
        it (int): How many steps we've taken so far
        kwargs (dict): Other stuff that do_filtering() needs

    Returns:
        passing (bool): Whether or not all the filters passed
        scores (dict): The scores of the filters
        filters (list[FilterBase]): The filters
    '''
    if px0 is None:
        px0 = indep.xyz.clone()

    filters = init_filters(sampler._conf)

    scores = {}
    passing = True
    try:
        do_filtering(filters, indep, t, it, px0, scores, is_diffused=sampler.is_diffused, **kwargs)
    except FilterFailedException:
        passing = False

    return passing, scores, filters




class TestFilters(unittest.TestCase):
    '''
    Test that filters actually return the values you expect

    A typical filter test will typically look like this:
        1. Call filter_prepare_indep() to setup the config and get an indep
        2. (optional) Modify the indep
        3. Call filter_evaluate()

    That will return the results of the filter which you can then assert
    '''


    def setUp(self) -> None:
        # Some other test is leaving a global hydra initialized, so we clear it here.
        if hydra.core.global_hydra.GlobalHydra().is_initialized():
            hydra.core.global_hydra.GlobalHydra().clear()
        return super().setUp()

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    def test_InterchainClashFilter_filter(self):
        '''
        Tests that the chain_break filter works
        '''
        indep, extra_args = filter_prepare_indep([
                'diffuser.T=1',
                'inference.input_pdb=test_data/two_chain.pdb',
                "contigmap.contigs=['A1-10_B121-130']",
                'filters.names=["InterchainClashFilter"]',
                '+filters.configs.InterchainClashFilter.t=1',
                '+filters.configs.ChainBreak.max_bb_clashes=2',
                '+filters.configs.ChainBreak.clash_dist=3',
                'contigmap.has_termini=[True,True]'
            ])

        mask0, mask1 = [torch.tensor(x) for x in indep.chain_masks()]

        indep.xyz[mask0] -= torch.mean(indep.xyz[mask0,1], axis=0)
        indep.xyz[mask1] -= torch.mean(indep.xyz[mask1,1], axis=0)

        dists = torch.linalg.norm(indep.xyz[mask0,:3].reshape(-1,3)[:,None] - indep.xyz[mask1,:3].reshape(-1,3)[None,:], axis=-1)
        clashes = (dists < 3).sum()

        expected_passing = clashes <= 2


        passing, scores, filters = filter_evaluate(indep, **extra_args)


        assert expected_passing == passing
        assert torch.isclose( scores['t-1_max_clashes'], clashes)


    def test_ChainBreak_filter(self):
        '''
        Tests that the chain_break filter works
        '''
        indep, extra_args = filter_prepare_indep([
                'diffuser.T=1',
                'inference.input_pdb=test_data/1qys.pdb',
                "contigmap.contigs=['A64-65']",
                'filters.names=["ChainBreak"]',
                '+filters.configs.ChainBreak.t="10,5,1"',
                '+filters.configs.ChainBreak.monitor_chains=0',
                '+filters.configs.ChainBreak.C_N_dist_limit=2',
            ])

        indep.xyz[1] += 10

        expected_dist = torch.linalg.norm(indep.xyz[0,2] - indep.xyz[1,0], axis=-1)

        passing, scores, filters = filter_evaluate(indep, **extra_args)

    
        assert not passing
        assert torch.isclose( scores['t-1_max_chain_gap'], expected_dist)

    def test_BBGPSatisfaction_filter(self):
        '''
        Tests that the backbone guidepost filter works
        '''
        indep, extra_args = filter_prepare_indep([
                'diffuser.T=1',
                'inference.input_pdb=test_data/1qys.pdb',
                "contigmap.contigs=['A64-65']",
                'inference.contig_as_guidepost=True',
                'filters.names=["BBGPSatisfaction"]',
                '+filters.configs.BBGPSatisfaction.t="1"',
                '+filters.configs.BBGPSatisfaction.gp_max_error_cut=0.3',
                '+filters.configs.BBGPSatisfaction.gp_rmsd_cut=0.3',
            ])

        assert indep.length() == 4
        assert torch.allclose(indep.is_gp, torch.tensor([False, False, True, True], dtype=bool))
        indep.xyz[0:2] = indep.xyz[2:4]
        indep.xyz[0:2,:,0] += 0.2

        passing, scores, filters = filter_evaluate(indep, **extra_args)

        assert passing
        assert torch.isclose( scores['t-1_gp_max_error'], torch.tensor(0.2))
        assert torch.isclose( scores['t-1_gp_rmsd'], torch.tensor(0.2))

        indep.xyz[0:2,:,0] += 0.2

        passing, scores, filters = filter_evaluate(indep, **extra_args)

        assert not passing
        assert torch.isclose( scores['t-1_gp_max_error'], torch.tensor(0.4))
        assert torch.isclose( scores['t-1_gp_rmsd'], torch.tensor(0.4))

        

    def test_scorefile_formatter(self):
        '''
        Tests that the scorefile formatter behaves like we expect
        '''

        tests = [
            (0, '0'),
            (1, '1'),
            (1.5, '1.500'),
            ('hello', 'hello'),
            (torch.tensor(5), '5'),
            (torch.tensor([0.02]), '0.020'),
            (np.array(-5), '-5'),
            (np.array([[-0.2]]), '-0.200'),
            (0.000012345, '1.23e-05'),
            (True, '1'),
            (False, '0'),
            ([1, 2, 3], None), # This one's just here to make sure it doesn't crash. People shouldn't output stuff like this
        ]

        for value, goal in tests:
            result = silent_files.format_scores(dict(key=value), float_transition_value=0.01)['key']

            if goal is not None:
                assert result == goal, f'{result} did not match {goal}'


if __name__ == '__main__':
        unittest.main()
