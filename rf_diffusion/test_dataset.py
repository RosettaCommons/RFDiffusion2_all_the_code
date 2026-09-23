from functools import partial
import os
import unittest

import assertpy
from icecream import ic
import hydra
import numpy as np
import torch
import copy
import pandas as pd
import pytest

from dev import analyze
from rf_diffusion.data_loader import get_fallback_dataset_and_dataloader
from rf_diffusion import test_utils
from rf2aa import tensor_util
from rf_diffusion.frame_diffusion.data import se3_diffuser
import show
from rf_diffusion import aa_model

from rf_diffusion.chemical import ChemicalData as ChemData

pytestmark = pytest.mark.requires_internal_data

cmd = analyze.cmd


weird_item = """{
    'chosen_dataset': 'sm_compl_covale',
    'mask_gen_seed': 79592293,
    'sel_item': {   'ASSEMBLY': 1,
                    'CHAINID': '6ndp_B',
                    'CLUSTER': 25998,
                    'COVALENT': [   (   ('B', '19', 'CYS', 'SG'),
                                        ('D', '900', 'BLA', 'CBC'))],
                    'DEPOSITION': '2018-12-14',
                    'HASH': '062428',
                    'LEN_EXIST': 602,
                    'LIGAND': [('D', '900', 'BLA')],
                    'LIGATOMS': 43,
                    'LIGATOMS_RESOLVED': 43,
                    'LIGXF': [('D', 3)],
                    'PARTNERS': [   (   'B',
                                        1,
                                        498,
                                        1.8329198360443115,
                                        'polypeptide(L)'),
                                    (   'A',
                                        0,
                                        0,
                                        13.859391212463379,
                                        'polypeptide(L)')],
                    'RESOLUTION': 3.89,
                    'SEQUENCE': 'MHHHHHHSTATNPLDLDVCAREPIHIPGLIQPYGVLLVIDPADGRIVQASTTAADLLGVPMAALLGMPYTQVLTLPEAQPFAVDDQPQHLMHAEVRFPQRATPPASAWVAAWHLYPQQWLVEMEPRDARLLDVTLREAMPLLRSVERDPGIAEAAVRVAKGLRSLIGFDRVMIYRFDEEWNGDIIAEARKPELEAYLGQHYPASDIPAQARALYLRNRVRQIADVGYQPSPIQPTVHPQLGTPVDLSDVSLRSVSPVHLEYLANMGVTATLVASIVVNDALWGLISCHHYSPHFTNHAMRDVTDAVARTLAGRIGALQAVARARLESVLLTVREKLITDFNDAEHMTVELLDDMAPDLMDVVDADGVAIFHGNDISRHGTTPDVAALRRIRDHIESEHHEALREDAVGALHVDAIGEVFPELADLAPLAAGFIFVPLMPQSRSALLWTRREQIQQIKWAGNPQLAKLEDIPNSRLSPRKSFDLWQQTVRGRARRWSPLHLESARSLRVLIELMERKRFQQDFTLLEASLSRLRDGVAIIERGTANAAHRLLFVNTAFADVCGSDVAELIGRELQTLYASDAPRANVELLQDALRNGRAAYVTLPLQVSDGAPVYRQFHLEPLPSPSGVTAHWLLQLRDPE',
                    'SUBSET': 'covale'},
    'task': 'diff',
}"""

bridge_item = """{
    'chosen_dataset': 'sm_compl_covale',
    'mask_gen_seed': 78231548,
    'sel_item': {   'ASSEMBLY': 2,
                    'CHAINID': '5fvb_P',
                    'CLUSTER': 2189,
                    'COVALENT': [   (   ('P', '48', 'CYS', 'SG'),
                                        ('SB', '203', 'PEB', 'CAA')),
                                    (   ('P', '59', 'CYS', 'SG'),
                                        ('SB', '203', 'PEB', 'CAD'))],
                    'DEPOSITION': '2016-02-05',
                    'HASH': '067414',
                    'LEN_EXIST': 183,
                    'LIGAND': [('SB', '203', 'PEB')],
                    'LIGATOMS': 43,
                    'LIGATOMS_RESOLVED': 43,
                    'LIGXF': [('SB', 40)],
                    'PARTNERS': [   (   'P',
                                        9,
                                        446,
                                        1.8050484657287598,
                                        'polypeptide(L)'),
                                    (   [('NA', '201', 'PEB')],
                                        [('NA', 27)],
                                        0,
                                        7.117906093597412,
                                        'nonpoly'),
                                    (   'F',
                                        5,
                                        0,
                                        7.5266852378845215,
                                        'polypeptide(L)'),
                                    (   'A',
                                        0,
                                        0,
                                        7.633793354034424,
                                        'polypeptide(L)'),
                                    (   [('RB', '202', 'PEB')],
                                        [('RB', 39)],
                                        0,
                                        12.504525184631348,
                                        'nonpoly'),
                                    (   'D',
                                        3,
                                        0,
                                        12.61732292175293,
                                        'polypeptide(L)'),
                                    (   [('QB', '201', 'PEB')],
                                        [('QB', 38)],
                                        0,
                                        12.873053550720215,
                                        'nonpoly'),
                                    (   [('Z', '202', 'PEB')],
                                        [('Z', 13)],
                                        0,
                                        15.891844749450684,
                                        'nonpoly'),
                                    (   'R',
                                        11,
                                        0,
                                        20.833248138427734,
                                        'polypeptide(L)')],
                    'RESOLUTION': 1.93,
                    'SEQUENCE': 'MLDAFSRAVVQADASTSVVADMGALKQFIAEGNRRLDAVNAIASNASCMVSDAVAGMICENQGLIQAGGXCYPNRRMAACLRDAEIILRYVTYALLAGDASVLDDRCLNGLKETYAALGVPTTSTVRAVQIMKAQAAAHIKDTPSEARAGGKLRKMGSPVVEDRCASLVAEASSYFDRVISALS',
                    'SUBSET': 'covale'},
    'task': 'diff',
}"""


multi_covale = """{
    'chosen_dataset': 'sm_compl_multi',             
    'mask_gen_seed': 82644955,            
    'sel_item': {   'ASSEMBLY': 1,                         
                    'CHAINID': '1btc_A',           
                    'CLUSTER': 18655,                         
                    'COVALENT': [],                
                    'DEPOSITION': '1993-02-18',
                    'HASH': '101639',                     
                    'LEN_EXIST': 491,               
                    'LIGAND': [   ('B', '1', 'GLC'),          
                                  ('B', '2', 'GLC'),
                                  ('B', '3', 'GLC'),
                                  ('B', '4', 'GLC'),       
                                  ('B', '5', 'GLC'), 
                                  ('B', '6', 'GLC')],         
                    'LIGATOMS': 66,                
                    'LIGATOMS_RESOLVED': 66,
                    'LIGXF': [('B', 1)],                   
                    'PARTNERS': [   (   'A',        
                                        0,  
                                        216,                                                                                                                                                                                                                                              
                                        2.691878318786621,
                                        'polypeptide(L)'),    
                                    (   [('F', '504', 'BME')],                                                                                                                                                                                                                            
                                        [('F', 5)],
                                        0,                                                                                                                                                                                                                                                
                                        5.804349899291992,
                                        'nonpoly'),                                                                                                                                                                                                                                       
                                    (   [('D', '502', 'BME')],
                                        [('D', 3)],                                                                                                                                                                                                                                       
                                        0,                
                                        6.248571395874023,                                                                                                                                                                                                                                
                                        'nonpoly')],
                    'RESOLUTION': 2.0,                                                                                                                                                                                                                                                    
                    'SEQUENCE': 'SNMLLNYVPVYVMLPLGVVNVDNVFEDPDGLKEQLLQLRAAGVDGVMVDVWWGIIELKGPKQYDWRAYRSLFQLVQECGLTLQAIMSFHQCGGNVGDIVNIPIPQWVLDIGESNHDIFYTNRSGTRNKEYLTVGVDNEPIFHGRTAIEIYSDYMKSFRENMSDFLESGLIIDIEVGLGPAGELRYPSYPQSQGWEFPRIGEFQCYDKYLKADFKAAVARAGHPEWELPDDAGKYNDVPESTGFFKSNGTYVTEKGKFFLTWYSNKLLNHGDQILDEANKAFLGCKVKLAIKVSGIHWWYKVENHAAELTAGYYNLNDRDGYRPIARMLSRHHAILNFTCLEMRDSEQPSDAKSGPQELVQQVLSGGWREDIRVAGENALPRYDATAYNQIILNAKPQGVNNNGPPKLSMFGVTYLRLSDDLLQKSNFNIFKKFVLKMHADQDYCANPQKYNHAITPLKPSAPKIPIEVLLEATKPTLPFPWLPETDMKVDG',                                      
                    'SUBSET': 'multi'},
    'task': 'diff',
}"""

nonatomized_covale = """{
    'chosen_dataset': 'sm_compl_covale',
    'index': 2,
    'mask_gen_seed': 72799612,
    'sel_item': {   'ASSEMBLY': 1,
                    'CHAINID': '1nbp_A',
                    'CLUSTER': 22700,
                    'COVALENT': [   (   ('A', '31', 'CYS', 'SG'),
                                        ('C', '201', 'MHC', 'S1'))],
                    'DEPOSITION': '2002-12-03',
                    'HASH': '004725',
                    'LEN_EXIST': 121,
                    'LIGAND': [('C', '201', 'MHC')],
                    'LIGATOMS': 18,
                    'LIGATOMS_RESOLVED': 18,
                    'LIGXF': [('C', 2)],
                    'PARTNERS': [   (   'A',
                                        0,
                                        243,
                                        2.0362677574157715,
                                        'polypeptide(L)')],
                    'RESOLUTION': 2.2,
                    'SEQUENCE': 'APTSSSTKKTQLQLEHLLLDLQMILNGINNCKNPKLTRMLTFKFYMPKKATELKHLQCLEEELKPLEEVLNLAQSKNFHLRPRDLISNINVIVLELKGSETTFMCEYADETATIVEFLNRWITFCQSIISTLT',
                    'SUBSET': 'covale'},
    'task': 'diff',
}"""


multiligand_item = """{
    'chosen_dataset': 'sm_compl_asmb',                                                                                                       
    'index': 0,                                                                                                                              
    'mask_gen_seed': 68054314,                                                                                                               
    'sel_item': {   'ASSEMBLY': 2,                                                                                                           
                'CHAINID': '6fp1_B',                                                                                                     
                'CLUSTER': 6750,                                                                                                         
                'COVALENT': [],                                                                                                          
                'DEPOSITION': '2018-02-08',                                                                                              
                'HASH': '104197',                                                                                                        
                'LEN_EXIST': 445,                                                                                                        
                'LIGAND': [('G', '501', 'FAD')],                                                                                         
                'LIGATOMS': 53,                                                                                                          
                'LIGATOMS_RESOLVED': 53,                                                                                                 
                'LIGXF': [('G', 1)],                                                                                                     
                'PARTNERS': [   (   'B',                                                                                                                                                                                                                                              
                                    0,                                                                                                                                                                                                                                                
                                    605,                                                                                                 
                                    2.6784911155700684,                                                                                  
                                    'polypeptide(L)'),                                                                                   
                                (   [('M', '507', 'E0Q')],                                                                               
                                    [('M', 7)],                                                                                          
                                    16,                                                                                                  
                                    3.4406967163085938,                                                                                  
                                    'nonpoly'),                                                                                          
                                (   [('J', '504', 'PEG')],                                                                               
                                    [('J', 4)],                                                                                          
                                    0,                                                                                                   
                                    12.409589767456055,                                                                                  
                                    'nonpoly'),                                                                                          
                                (   [('K', '505', 'PEG')],                                                                               
                                    [('K', 5)],                                                                                          
                                    0,                                                                                                   
                                    14.890626907348633,                                                                                  
                                    'nonpoly'),                                                                                          
                                (   [('L', '506', 'PEG')],                                                                               
                                    [('L', 6)],                                                                                          
                                    0,
                                    20.84391212463379,
                                    'nonpoly')],
                'RESOLUTION': 1.97,
                'SEQUENCE': 'TATDNARQVTIIGAGLAGTLVARLLARNGWQVNLFERRPDPRIETGARGRSINLALAERGAHALRLAGLEREVLAEAVMMRGRMVHVPGTPPNLQPYGRDDSEVIWSINRDRLNRILLDGAEAAGASIHFNLGLDSVDFARQRLTLSNVSGERLEKRFHLLIGADGCNSAVRQAMASVVDLGEHLETQPHGYKELQITPEASAQFNLEPNALHIWPHGDYMCIALPNLDRSFTVTLFLHHQSPAAQPASPSFAQLVDGHAARRFFQRQFPDLSPMLDSLEQDFEHHPTGKLATLRLTTWHVGGQAVLLGDAAHPMVPFHGQGMNCALEDAVALAEHLQSAADNASALAAFTAQRQPDALAIQAMALENYVEMSSKVASPTYLLERELGQIMAQRQPTRFIPRYSMVTFSRLPYAQAMARGQIQEQLLKFAVANHSDLTSINLDAVEHEVTRCLPPLSHLS',
                'SUBSET': 'asmb'},
    'task': 'diff',
}"""


REWRITE=False
# REWRITE=True
class Dataloader(unittest.TestCase):

    def cmp(self, a, b):
        # For NA compataibility
        a.xyz = a.xyz[:,:ChemData().NHEAVYPROT]  # xyz dimension should be 23, but only check the first 14
        b.xyz = b.xyz[:,:ChemData().NHEAVYPROT]  
        cmp = partial(tensor_util.cmp, atol=1e-4, rtol=1e-5)
        return cmp(a, b)

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        return super().tearDown()

    def indep_for_dataset(self, dataset, mask, epoch=0, overrides=[], **kwargs):

        #show_tip_pa.clear()
        #cmd.set('grid_mode', 1)
        return test_utils.loader_out_for_dataset(dataset,mask,overrides=overrides,epoch=epoch, **kwargs)

    def test_t_1_diffused_centered(self):
        loader_out = self.indep_for_dataset(
            dataset = 'sm_complex',
            mask = 'get_tip_mask',
            config_name = 'fm_tip_center_all_from_scratch_enz_finetune',
            overrides=[
                'dataloader.CROP=60',
                'diffuser.t_distribution=dirac_1',
            ],
        )
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

        diffused_r3 = rfi.xyz[0, is_diffused,1]
        diffused_r3_com = torch.mean(diffused_r3, dim=0)

        # Assert we have some diffused, some not diffused
        assertpy.assert_that(is_diffused.float().mean().item()).is_between(0.001, 0.999)

        # Assert that the diffused atoms are centered
        np.testing.assert_allclose(diffused_r3_com.numpy(), 0, atol=1e-5)


    def test_t_0p4_diffused_centered(self):
        loader_out = self.indep_for_dataset(
            dataset = 'sm_complex',
            mask = 'get_tip_mask',
            config_name = 'fm_tip_center_all_from_scratch_enz_finetune',
            overrides=[
                'dataloader.CROP=60',
                'diffuser.t_distribution=dirac_p4',
            ],
        )
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

        diffused_r3 = rfi.xyz[0, is_diffused,1]
        diffused_r3_com = torch.mean(diffused_r3, dim=0)
        T = 200
        t_flow = 1 - little_t / T

        # Assert we have some diffused, some not diffused
        assertpy.assert_that(is_diffused.float().mean().item()).is_between(0.001, 0.999)

        # Assert that the diffused atoms are interpolated between X0 and the gt.
        # TODO: X0 is not currently recorded, so we can't check this.

        # Assert that the diffused atoms CoM are interpolated between (0,0,0) and the gt.
        diffused_r3_gt = indep.xyz[is_diffused, 1]
        diffused_r3_com_gt = torch.mean(diffused_r3_gt, dim=0)
        diffused_r3_com_interp = diffused_r3_com_gt * t_flow

        ic(
            t_flow,
            diffused_r3_com,
            diffused_r3_com_interp,
        )

        np.testing.assert_allclose(diffused_r3_com.numpy(), diffused_r3_com_interp, 0, atol=1e-5)

    def test_t_1_batch_ot(self):
        loader_out = self.indep_for_dataset(
            dataset = 'sm_complex',
            mask = 'get_tip_mask',
            config_name = 'fm_tip_center_all_from_scratch_enz_finetune',
            overrides=[
                'dataloader.CROP=60',
                'diffuser.t_distribution=dirac_1',
            ],
        )
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

        diffused_r3 = rfi.xyz[0, is_diffused,1]

        try:
            # Assert we have some diffused, some not diffused
            assertpy.assert_that(is_diffused.float().mean().item()).is_between(0.001, 0.999)

            # TODO: Assert that the diffused atoms are interpolated between X0 and the gt.
            # Unfortunately the sampled X0 is not currently recorded by the dataset, so we can't check this.
        except Exception as e:
            # If the assertion fails, save the tensors to disk for inspection.
            diffused_r3_gt = indep.xyz[is_diffused, 1]
            relpath = 'tmp/trash/dataloader_0.pt'
            os.makedirs(os.path.dirname(relpath), exist_ok=True)
            torch.save(dict(diffused_r3_gt=diffused_r3_gt, diffused_r3=diffused_r3), relpath)
            raise e

    def test_uncond_sm(self):
        dataset = 'sm_complex'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_atomized_sm(self):
        dataset = 'sm_complex'
        mask = 'get_closest_tip_atoms'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    
    def test_covale_uncond(self):
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        def cmp(got, want):
            got.idx[-13:] = -1
            want.idx[-13:] = -1
            return self.cmp(got, want)

        if self.show_in_pymol:
            show.one(indep, atomizer)
            cmd.show('licorice', 'resi 491')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)

    def test_simple_mask(self):
        dataset = 'pdb_aa'
        mask = 'get_diffusion_mask_simple'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=60'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)
    
    def test_covale_uncond_weird_covale_bonds(self):
        '''
        The atomized C and the residue C have a distance of 0.03A.
        Currently we are handling this by having the identity cutoff be <0.1A,
        But this begs further investigation, as this makes no sense: atomized coordinates should be exact copies.

        dist = torch.cdist...
        is_weird_distance = (dist > 1e-4) * (dist < 1e-1)
        '''
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false', f'spoof_item="{weird_item}"'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        bonds = indep.metadata['covale_bonds']
        assertpy.assert_that(len(bonds)).is_equal_to(1)

        golden_name = f'indep_{dataset}-{mask}-weird'

        if self.show_in_pymol:
            show.one(indep, None, golden_name)
            show.one(indep, atomizer, golden_name+'_deatomized')

        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)
        
    def test_covale_uncond_bridge(self):
        '''
        Covale which is bonded to 2 residues.
        '''
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false', f'spoof_item="{bridge_item}"'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        bonds = indep.metadata['covale_bonds']
        assertpy.assert_that(len(bonds)).is_equal_to(2)

        golden_name = f'indep_{dataset}-{mask}-bridge'

        if self.show_in_pymol:
            show.one(indep, None, golden_name)
            show.one(indep, atomizer, golden_name+'_deatomized')

        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_covale_uncond_bridge_gpbonds(self):
        '''
        Covale which is bonded to 2 residues.
        '''
        dataset = 'sm_compl_covale'
        mask = 'get_unconditional_diffusion_mask'
        
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[f'spoof_item="{bridge_item}"'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        bonds = indep.metadata['covale_bonds']
        assertpy.assert_that(len(bonds)).is_equal_to(2)
        assertpy.assert_that(indep.same_chain.all()).is_true()

        golden_name = f'indep_{dataset}-{mask}-bridge-gpbonds'

        if self.show_in_pymol:
            show.one(indep, None, golden_name)
            show.one(indep, atomizer, golden_name+'_deatomized')

        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_covale_islands_partial_ligand(self):
        dataset = 'sm_compl_covale'
        mask = 'get_diffusion_mask_islands_partial_ligand'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['guidepost_bonds=false'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        def cmp(got, want):
            got.idx[-13:] = -1
            want.idx[-13:] = -1
            return self.cmp(got, want)
        ic((~is_diffused).nonzero()[:,0])
        ic(np.unique(indep.chains()))

        if self.show_in_pymol:
            show.color_diffused(indep, is_diffused)
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=cmp)
    
    def test_compl_islands(self):
        dataset = 'compl'
        mask = 'get_diffusion_mask_islands'
        loader_out = self.indep_for_dataset(dataset, mask)
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        ic((~is_diffused).nonzero()[:,0])
        ic(indep.chains())
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_PPI_random_motif_no_crop(self):
        dataset = 'compl'
        mask = 'get_PPI_random_motif_no_crop'
        # The PPI crops work with guideposting, it's just that it makes the pymol/golden very hard to inspect
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['transforms.configs.AddConditionalInputs.p_is_guidepost_example=0'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        ic((~is_diffused).nonzero()[:,0])
        ic(indep.chains())
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_PPI_interface_motif_radial_crop(self):
        dataset = 'compl'
        mask = 'get_PPI_interface_motif_radial_crop'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
                                                                      'dataloader.mask.ppi_radial_crop_low=6',
                                                                      'dataloader.mask.ppi_radial_crop_high=6'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        ic((~is_diffused).nonzero()[:,0])
        ic(indep.chains())
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_PPI_no_motif_planar_crop(self):
        dataset = 'compl'
        mask = 'get_PPI_fully_diffused_planar_crop'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
                                                                      'dataloader.mask.ppi_planar_crop_low=6',
                                                                      'dataloader.mask.ppi_planar_crop_high=6'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        
        golden_name = f'indep_{dataset}-{mask}'
        ic((~is_diffused).nonzero()[:,0])
        ic(indep.chains())
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_PPI_random_motif_no_crop_can_be_gp(self):
        '''
        The point of this test is to make sure that can_be_gp is actually working
        '''
        dataset = 'compl'
        mask = 'get_PPI_random_motif_no_crop'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['transforms.configs.AddConditionalInputs.p_is_guidepost_example=1'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}_can_be_gp'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_PPI_hotspots_full_stack(self):
        '''
        This tests the full-stack of stuff for PPI training.
        Very, very much a regression test. This test cycles through like 30 dataloader examples before picking one
        '''
        dataset = 'compl'
        mask = 'get_PPI_random_motif_no_crop'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            '+extra_tXd=["ppi_hotspots_antihotspots"]',
            '+extra_tXd_params.ppi_hotspots_antihotspots={}',
            '++transforms.names=["AddConditionalInputs","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++upstream_training_transforms.names=["PPITrimTailsChain0ComplexTransform","PPIRejectUnfoldedInterfacesTransform","PPIJoeNateDatasetRadialCropTransform","GenerateMasks","PopMask","FindHotspotsTrainingTransform"]',
            '++upstream_training_transforms.configs.PPITrimTailsChain0ComplexTransform={}',
            '++upstream_training_transforms.configs.PPIRejectUnfoldedInterfacesTransform.binder_fbscn_cut=0.30',
            '++upstream_training_transforms.configs.PPIJoeNateDatasetRadialCropTransform={}',
            '++upstream_training_transforms.configs.GenerateMasks={}',
            '++upstream_training_transforms.configs.PopMask={}',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.p_is_hotspot_example=1.0',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.p_is_antihotspot_example=1.0',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.max_hotspot_frac=1.0',
            '++upstream_training_transforms.configs.FindHotspotsTrainingTransform.max_antihotspot_frac=1.0',
            '+dataloader.fast_filters.compl.names=["reject_chain0_longer_than"]',
            '+dataloader.fast_filters.compl.configs.reject_chain0_longer_than.max_length=170',
            '+dataloader.fast_filters.compl.configs.reject_chain0_longer_than.verbose=True',
            '+dataloader.dataset_param_overrides.compl.CROP=600',
            '++dataloader.CROP=190',
            '++dataloader.mask.only_first_chain_ppi_binders=True'
            ])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}_hotspots'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')

            hotspots = '+'.join(str(int(x)) for x in indep.idx[indep.extra_t1d[:,-4].bool()])
            antihotspots = '+'.join(str(int(x)) for x in indep.idx[indep.extra_t1d[:,-2].bool()])
            print('hotspots:', hotspots)
            print('antihotspots:', antihotspots)

        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_ideal_ss(self):
        '''
        This tests the ideal_ss training code
        '''
        dataset = 'compl'
        mask = 'get_PPI_random_motif_no_crop'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            '+extra_tXd=["ideal_ss_cond"]',
            "+extra_tXd_params.ideal_ss_cond.topo_spec_choices=['HH','HHH','HHHH','HHHHH']",
            '++transforms.names=["AddConditionalInputs","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++upstream_training_transforms.names=["GenerateMasks","PopMask","AddIdealSSTrainingTransform"]',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_ideal_ss=1',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_loop_frac=1',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_avg_scn=1',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_topo_spec=1',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_chain_mask=0',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.p_ideal_speckle=0',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.ideal_smooth_window=9',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.ideal_gaussian_std=0.3',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_min_value=1.5',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_max_value=2.5',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_per_res=True',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_smooth_window=9',
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.scn_gaussian_std=0.3',
            "+upstream_training_transforms.configs.AddIdealSSTrainingTransform.topo_spec_choices=['HH','HHH','HHHH','HHHHH']",
            '++upstream_training_transforms.configs.AddIdealSSTrainingTransform.topo_spec_min_helix_length=8',
            '++dataloader.mask.only_first_chain_ppi_binders=True'
            ])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None


        chain_A_size = len(indep.chain_masks()[0])
        ideal_ss_mask = indep.extra_t1d[:,-11]
        avg_scn_mask = indep.extra_t1d[:,-9]
        loop_frac_mask = indep.extra_t1d[:,-7]

        assert ideal_ss_mask[:chain_A_size].all()
        assert not ideal_ss_mask[chain_A_size:].any()
        assert avg_scn_mask[:chain_A_size].all()
        assert not avg_scn_mask[chain_A_size:].any()
        assert loop_frac_mask[:chain_A_size].all()
        assert not loop_frac_mask[chain_A_size:].any()

        golden_name = f'indep_{dataset}-{mask}_ideal_ss'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')


        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)




    def test_dhub_pdb_dataset(self):
        '''
        This tests the datahub dataset loading as well as the ReorderChains dhub transform and the
            machinery that loads dhub values into conditions dict
        '''
        dataset = None
        mask = 'get_diffusion_mask_simple'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            '++datahub.test_dhub_dataset.probability=1.0',
            '++dataloader.DATASET_PROB=[0.0]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            ], config_name='debug_dhub')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None


        assert torch.allclose(conditions_dict['my_arange_condition'].long(), torch.cat([torch.arange(10, 20), torch.arange(10)]))

        golden_name = f'indep_dhub-{mask}_reorder_arange_test'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')


        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)




    def test_dhub_pdb_dataset_both(self):
        '''
        This tests that the DATAHUB_MODE=BOTH actually works. If it doesn't there will be an error about the probabilities not summing to 1
        '''
        dataset = None
        mask = 'get_diffusion_mask_simple'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            ], config_name='debug_dhub')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None


        golden_name = f'indep_dhub-{mask}_both'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')


        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)




    def test_dhub_chirals(self):
        '''
        This tests that datahub correctly loads chirals
        '''


        dataset = 'sm_compl_asmb'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
            'dataloader.CROP=1000',
            f'spoof_item="{multiligand_item}"',
            '+dataloader.DIFF_MASK_PROBS={get_diffusion_mask_simple:1}',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            ])
        indep_old, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

        # I can't make datahub load the peg no matter how hard I try
        peg_mask = torch.zeros(indep_old.length(), dtype=bool)
        for mask in [torch.tensor(x) for x in indep_old.chain_masks()]:
            if mask.sum() == 7:
                peg_mask |= mask
        indep_old, _ = aa_model.slice_indep(indep_old, ~peg_mask)
        indep_old.xyz -= torch.mean(indep_old.xyz[:,1], axis=0)

        pq_name = 'test_dhub_chirals.parquet'
        pq = pd.DataFrame({
            'example_id':['0'],
            'cluster':[0],
            'path':'test_data/6fp1.cif',
            'assembly_id':['2'], #  this line isn't needed if you copy/paste this
            })
        pq.to_parquet(pq_name)


        dataset = None
        mask = 'get_diffusion_mask_simple'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            f'++datahub.test_dhub_dataset.dataset.dataset.data={pq_name}',
            '++datahub.test_dhub_dataset.dataset.dataset.columns_to_load=[example_id,cluster,path,assembly_id]', # you don't need this line if you are copy/pasting
            'dataloader.CROP=1000',
            '+dataloader.DIFF_MASK_PROBS={get_diffusion_mask_simple:1}',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            ], config_name='debug_dhub_simple')
        indep_dhub, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out

        indep_dhub.xyz -= torch.mean(indep_dhub.xyz[:,1], axis=0)

        os.remove(pq_name)

        assert torch.allclose(indep_old.xyz[:,1], indep_dhub.xyz[:,1], atol=0.1)
        assert torch.allclose(indep_old.chirals, indep_dhub.chirals)



    def test_target_hbond_satisfaction(self):
        '''
        This tests the ideal_ss training code
        '''
        dataset = 'compl'
        mask = 'get_PPI_random_motif_no_crop'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            '+extra_tXd=["target_hbond_satisfaction_cond"]',
            '+extra_tXd_params.target_hbond_satisfaction_cond={}',
            '++transforms.names=["AddConditionalInputs","ExpandConditionsDict"]',
            '++transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            '++transforms.configs.ExpandConditionsDict={}',
            '++upstream_training_transforms.names=["GenerateMasks","PopMask","HBondTargetSatisfactionTrainingTransform"]',
            '++upstream_training_transforms.configs.HBondTargetSatisfactionTrainingTransform.p_bb_sc_cats_shown=1',
            '++upstream_training_transforms.configs.HBondTargetSatisfactionTrainingTransform.hb_score_to_count=-0.01',
            '++upstream_training_transforms.configs.HBondTargetSatisfactionTrainingTransform.p_tiny_labeling=1',
            '++dataloader.mask.only_first_chain_ppi_binders=True'
            ])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}_target_hbond_sat'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')


        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_get_diffusion_mask_islands_w_tip_w_seq_islands(self):
        '''
        This test tests two things:
        1. Does the add_tips() function actually work
        2. Does the add_seq_islands() function work with a complex scenario
        '''
        dataset = 'compl'
        mask = 'get_diffusion_mask_islands_w_tip_w_seq_islands'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=['transforms.configs.AddConditionalInputs.p_is_guidepost_example=1',
                                                                      '+dataloader.mask.n_islands_min=5',
                                                                      '+dataloader.mask.n_islands_max=7',
                                                                      '+dataloader.mask.seq_n_islands_min=5',
                                                                      '+dataloader.mask.seq_n_islands_max=7'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        indep.seq = rfi.seq[0] # This line is critical for any test of sequence masking

        golden_name = f'indep_{dataset}-{mask}'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')

        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_multi_covale(self):
        dataset = 'sm_compl_multi'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10_2.pkl',
            f'spoof_item="{multi_covale}"',
            ], config_name='extra_t1d_v2')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        for r in indep.extra_t1d.T:
            print(r)

        golden_name = f'indep_{dataset}-{mask}'
        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            # show.cmd.do(f'util.cbc {name}')
            # show.diffused(indep, is_diffused, 'true')
            # show.one(indep, None)
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    def test_covale_is_guideposted(self):
        dataset = 'sm_compl_multi'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
            f'spoof_item="{nonatomized_covale}"',
            ], config_name='extra_t1d_v2')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        golden_name = f'indep_{dataset}-{mask}-is-guideposted'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do(f'util.cbc {name}')
            show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)
    
    def test_multiligand(self):

        dataset = 'sm_compl_asmb'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
            f'spoof_item="{multiligand_item}"',
            ], config_name='extra_t1d_v2')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None
        ic(
            indep.chains()[indep.is_sm],
            indep.idx[indep.is_sm],
        )
        golden_name = f'indep_{dataset}-{mask}-multiligand'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do(f'util.cbc {name}')
            show.cmd.color('orange', f'{name} and hetatm and elem C')
            # show.cmd.do(f'util.cbc {name}')
            # show.diffused(indep, is_diffused, 'true')
        
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_ss_adj(self):
        '''
        The point of this test is to make sure that ss/adj is actually working during training
        They are stored in extra_t1d[:,-5:] and extra_t2d[:,:,-3:]
        '''
        dataset = 'pdb_aa'
        mask = 'get_unconditional_diffusion_mask'
        loader_out = self.indep_for_dataset(dataset, mask, overrides=[
            'transforms.configs.AddConditionalInputs.p_is_guidepost_example=0',
            'upstream_training_transforms.configs.GenerateSSADJTrainingTransform.p_is_ss_example=1.0',
            'upstream_training_transforms.configs.GenerateSSADJTrainingTransform.p_is_adj_example=1.0',
            'upstream_training_transforms.configs.GenerateSSADJTrainingTransform.ss_max_mask=0.0',
            'upstream_training_transforms.configs.GenerateSSADJTrainingTransform.adj_max_mask=0.0',
            'upstream_training_transforms.configs.GenerateSSADJTrainingTransform.p_any_strand_pairs=1.0',
            'upstream_training_transforms.configs.GenerateSSADJTrainingTransform.adj_strand_pair_max_mask=0.0',
            ], config_name='fm_tip_ss_adj')
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}_ss_adj'

        if self.show_in_pymol:
            name, names = show.one(indep, None)
            show.cmd.do('util.cbc')
            show.diffused(indep, is_diffused, 'true')
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

    def test_atom_order(self):
        '''
        Tests that amino acid atoms from the dataloader are ordered the same way as in rf2aa.chemical.aa2long.
        Checks each amino acid type MIN_AA_COUNT times in each dataset defined in conf_train.dataloader.DATASETS.
        '''
        # Make the training conf
        import rf_diffusion
        pkg_dir = rf_diffusion.__path__[0]
        overrides = [f'dataloader.DATAPKL_AA={pkg_dir}/test_data/aa_dataset_256_subsampled_10.pkl']
        conf_train = test_utils.construct_conf_single(overrides=overrides, config_name='base')

        # Make the training dataloader
        LOAD_PARAM = {
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True
        }
        dataset, dataloader = get_fallback_dataset_and_dataloader(
            conf=conf_train,
            diffuser=se3_diffuser.SE3Diffuser(conf_train.diffuser),
            num_example_per_epoch=conf_train.epoch_size,
            world_size=1,
            rank=0,
            LOAD_PARAM=LOAD_PARAM,
        )

        # Set the minimum number of amino acids to check per dataset
        MIN_AA_COUNT = 5
        aa_count_tmp = {ChemData().num2aa[aa_int]: 0 for aa_int in range(20)}
        aa_count_by_dataset = {}
        for dataset_name, prob in zip(conf_train.dataloader.DATASETS.split(','), conf_train.dataloader.DATASET_PROB):
            if prob > 0:
                aa_count_by_dataset[dataset_name] = aa_count_tmp

        # Sample training examples until minimum counts are statisfied
        for indep_train, rfi_train, chosen_dataset, item, little_t, is_diffused_train, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict in dataloader:
            aa_count = aa_count_by_dataset[chosen_dataset]
            
            # Check if we've already seen enough of each aa in this dataset
            if test_utils.full_count(aa_count, MIN_AA_COUNT):
                continue
            
            # Check indep_train for aa atom permutations
            try:
                item_context = eval(item_context)
                atom_order_results = test_utils.detect_permuted_aa_atoms(indep_train, item_context)               
            except Exception as e:
                print(f'Couldn\'t process {item_context}. Skipping.')
                if False:
                    # For debugging
                    print(f"Caught an exception: {type(e).__name__}: {e}")
                    traceback.print_exc()
                atom_order_results = {}

            if atom_order_results:
                for aa3, record in atom_order_results.items():
                    for is_correct_order, msg in record:
                        if is_correct_order is False:
                            self.fail(msg)

                # Record how many individual amino acids have been checked.
                for aa3 in aa_count:
                    aa_count[aa3] += len(atom_order_results[aa3])

            if all(map(lambda aa_count: test_utils.full_count(aa_count, MIN_AA_COUNT), aa_count_by_dataset.values())):
                # Enough aa in each dataset have been checked
                break

    def test_na_compl(self):
        dataset = 'na_compl'
        mask = 'get_unconditional_diffusion_mask'

        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=364'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

        dataset = 'na_compl'
        mask = 'get_diffusion_mask_simple'

        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=364'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)

    def test_na_compl_island_and_atomization_masks(self):
        dataset = 'na_compl'
        mask = 'get_diffusion_mask_islands'

        loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=364'])
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict= loader_out
        indep.metadata = None

        golden_name = f'indep_{dataset}-{mask}'
        test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)

        unique_chains = np.unique(indep.chains())
        assertpy.assert_that(len(unique_chains)).is_equal_to(2)    

    def test_na_compl_masks(self):
        dataset = 'na_compl'

        masks = ['get_na_motif_scaffold', 'get_na_inverse_motif_scaffold', 'get_prot_unconditional_atomize_na_contacts', 'get_prot_contactmotif_atomize_na_contacts', 'get_prot_tipatom_guidepost_atomize_na_contacts', 'get_prot_tipatom_guidepost_na_contacts', 'get_prot_tipatom_guidepost_anywhere']

        for mask in masks:
            print(f'testing na_compl mask {mask}')
            loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=364'])
            indep = loader_out[0]
            indep.metadata= None

            golden_name = f'indep_{dataset}_{mask}'
            test_utils.assert_matches_golden(self, golden_name, indep, rewrite=REWRITE, custom_comparator=self.cmp)


    # def test_basic_datahub_integration(self):

    #     loader_out = test_utils.loader_out_for_datahub_dataset('datahub_test')
    #     indep = loader_out[0]
    #     indep.metadata=None

    #     golden_name = 'basic_datahub_integration'
    #     test_utils.assert_matches_golden(self, golden_name + '_indep', indep, rewrite=REWRITE, custom_comparator=self.cmp)
    


REWRITE=False
# REWRITE=True
class DataloaderTransformsNucleic(unittest.TestCase):

    def cmp(self, a, b):
        # For NA compataibility
        a.xyz = a.xyz[:,:14]  # xyz dimension should be 23, but only check the first 14
        b.xyz = b.xyz[:,:14]  
        cmp = partial(tensor_util.cmp, atol=1e-4, rtol=1e-5)
        return cmp(a, b)

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        return super().tearDown()

    def indep_for_dataset(self, dataset, mask, epoch=0, overrides=[], **kwargs):

        #show_tip_pa.clear()
        #cmd.set('grid_mode', 1)
        return test_utils.loader_out_for_dataset(dataset,mask,overrides=overrides,epoch=epoch, **kwargs)

    def test_transmute_na(self):
        import rf_diffusion.nucleic_compatibility_utils as nucl_utils

        overrides = ['transforms.names=["TransmuteNA", "GenerateMasks", "Center", "AddConditionalInputs"]',
                     '+transforms.configs.TransmuteNA.p_rna_to_dna=1.0', 
                     '+transforms.configs.TransmuteNA.p_dna_to_rna=1.0',
                     '+transforms.configs.Center={}',
                     '+transforms.configs.GenerateMasks={}',                     
                     ]
        
        loader_out = test_utils.loader_out_for_dataset('na_compl', 'get_unconditional_diffusion_mask', overrides=overrides, epoch=0)
        indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
        loader_out = test_utils.loader_out_for_dataset('na_compl', 'get_unconditional_diffusion_mask', overrides=[], epoch=0)        
        indep_orig = loader_out[0]

        seq_orig = indep_orig.seq[nucl_utils.get_resi_type_mask(indep_orig.seq, 'na')]
        seq = indep.seq[nucl_utils.get_resi_type_mask(indep_orig.seq, 'na')]
        # Ensure transmutation is taking place at the sequence level
        assertpy.assert_that(torch.any(seq != seq_orig)).is_equal_to(True)

        xyz_orig = indep_orig.xyz[nucl_utils.get_resi_type_mask(indep_orig.seq, 'na')]
        xyz = indep.xyz[nucl_utils.get_resi_type_mask(indep_orig.seq, 'na')] 
        # Ensure transmutation is taking place at the xyz level               
        assertpy.assert_that(torch.any(xyz_orig != xyz)).is_equal_to(True)

        # Ensure all backbone positions are the same
        xyz_orig = indep_orig.xyz
        xyz = indep.xyz
        max_diff = torch.max(torch.sqrt(torch.sum((xyz_orig[:,0:4] - xyz[:,0:4]) ** 2, dim=-1))).item()
        assertpy.assert_that(max_diff).is_less_than(1e-5)

        # Check all proteins coordinates are correct
        xyz_orig = indep_orig.xyz[nucl_utils.get_resi_type_mask(indep_orig.seq, 'prot_and_mask')]
        xyz = indep.xyz[nucl_utils.get_resi_type_mask(indep_orig.seq, 'prot_and_mask')]         
        seq_orig = indep_orig.seq[nucl_utils.get_resi_type_mask(indep_orig.seq, 'prot_and_mask')]
        # Compare indep to original using valid coordinates only
        is_valid = torch.zeros_like(xyz_orig[:,:,0], dtype=torch.bool)
        ic(indep_orig.xyz.shape)
        ic(seq_orig.shape)
        for i in range(xyz_orig.shape[0]):
            # Disregard hydrogens
            is_valid[i] = torch.tensor([not (u is None or u.find('H') > -1) 
                                        for u in ChemData().aa2long[seq_orig[i]]][:xyz_orig.shape[1]],
                                    dtype=bool)
        max_diff = torch.max(torch.sqrt(torch.sum((xyz_orig - xyz) ** 2, dim=-1)) * is_valid).item()
        assertpy.assert_that(max_diff).is_less_than(1e-5)            

    def test_transmute_na_raw(self):
        import rf_diffusion.nucleic_compatibility_utils as nucl_utils
        dataset = 'na_compl'
        #mask = 'get_atomized_islands'
        mask = 'get_unconditional_diffusion_mask'

        for epoch in range(10):
            loader_out = self.indep_for_dataset(dataset, mask, overrides=['dataloader.CROP=364'], epoch=epoch)
            indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
            indep.metadata = None

            indep_orig = indep.clone()

            indep = indep.clone()

            is_dna = nucl_utils.get_resi_type_mask(indep.seq, 'dna')
            is_rna = nucl_utils.get_resi_type_mask(indep.seq, 'rna')
            is_other = ~torch.logical_or(is_dna, is_rna)
            has_dna = torch.any(is_dna)
            has_rna = torch.any(is_rna)

            xyz_new = torch.zeros_like(indep.xyz)
            seq_new = copy.deepcopy(indep.seq)

            if has_dna:
                seq_new, xyz_new = nucl_utils.TransmuteNA.transmute_dna_to_rna(indep.seq, indep.xyz, seq_new, xyz_new, is_dna)
            if has_rna:
                seq_new, xyz_new = nucl_utils.TransmuteNA.transmute_rna_to_dna(indep.seq, indep.xyz, seq_new, xyz_new, is_rna)
            # Copy all else
            xyz_new[is_other] = indep.xyz[is_other]

            indep.xyz = xyz_new
            indep.seq = seq_new

            indep = indep.clone()

            is_dna = nucl_utils.get_resi_type_mask(indep.seq, 'dna')
            is_rna = nucl_utils.get_resi_type_mask(indep.seq, 'rna')
            is_other = ~torch.logical_or(is_dna, is_rna)
            has_dna = torch.any(is_dna)
            has_rna = torch.any(is_rna)

            xyz_new = torch.zeros_like(indep.xyz)
            seq_new = copy.deepcopy(indep.seq)

            if has_dna:
                seq_new, xyz_new = nucl_utils.TransmuteNA.transmute_dna_to_rna(indep.seq, indep.xyz, seq_new, xyz_new, is_dna)     
            if has_rna:
                seq_new, xyz_new = nucl_utils.TransmuteNA.transmute_rna_to_dna(indep.seq, indep.xyz, seq_new, xyz_new, is_rna)
            # Copy all else
            xyz_new[is_other] = indep.xyz[is_other]

            indep.xyz = xyz_new
            indep.seq = seq_new

            # Compare indep to original using valid coordinates only
            is_valid = torch.zeros_like(indep_orig.xyz[:,:,0], dtype=torch.bool)
            for i in range(indep_orig.xyz.shape[0]):
                # Disregard hydrogens
                is_valid[i] = torch.tensor([not (u is None or u.find('H') > -1) 
                                            for u in ChemData().aa2long[indep_orig.seq[i]]][:indep_orig.xyz.shape[1]],
                                        dtype=bool)
            max_diff = torch.max(torch.sqrt(torch.sum((indep_orig.xyz - indep.xyz) ** 2, dim=-1)) * is_valid).item()
            assertpy.assert_that(max_diff).is_less_than(5e-1)    

    def test_invalid_mask_exception(self):
        '''
        Ensures that the framework that catches InvalidMaskException actually works.
        If this doesn't throw an exception, it worked.
        '''
        conf = test_utils.construct_conf([
        'dataloader.DATAPKL_AA=aa_dataset_256_subsampled_10.pkl',
        'dataloader.CROP=256',
        'dataloader.DATASETS=compl',
        'dataloader.DATASET_PROB=[1.0]',
        'dataloader.DIFF_MASK_PROBS=null',
        'dataloader.DIFF_MASK_PROBS={get_diffusion_mask_simple:1,get_invalid_mask:1e100}',
        'debug=True',
        'spoof_item=null',])
        test_utils.loader_out_from_conf(conf)


REWRITE=False
# REWRITE=True
class TestCenterDiffused(unittest.TestCase):

    def tearDown(self) -> None:
        hydra.core.global_hydra.GlobalHydra().clear()
        return super().tearDown()

    def test_center_diffused(self):
        overrides = ['upstream_training_transforms.names=["GenerateMasks", "PopMask"]',
                     '+upstream_training_transforms.configs.GenerateMasks={}',            
                     '+upstream_training_transforms.configs.PopMask={}',              
                     'transforms.names=["AddConditionalInputs", "CenterPostTransform"]',
                     '+transforms.configs.CenterPostTransform.jitter=0.0',                     
                    ]

        for mask in ['get_diffusion_mask_islands_partial_ligand', 'get_unconditional_diffusion_mask']:
            for dataset in ['pdb_aa', 'sm_complex']:
                loader_out = test_utils.loader_out_for_dataset(dataset, mask, overrides=overrides, epoch=0)
                indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
                com = torch.mean(indep.xyz[is_diffused,1,:], dim=0)
                zero = torch.tensor([0,0,0], dtype=indep.xyz.dtype)
                assertpy.assert_that(torch.norm(com - zero).item()).is_less_than(1e-4)

    def test_not_diffused(self):
        overrides = ['upstream_training_transforms.names=["GenerateMasks", "PopMask"]',
                     '+upstream_training_transforms.configs.GenerateMasks={}',            
                     '+upstream_training_transforms.configs.PopMask={}',    
                     'transforms.names=["AddConditionalInputs", "CenterPostTransform"]',
                     '+transforms.configs.CenterPostTransform.jitter=0.0',  
                     '+transforms.configs.CenterPostTransform.center_type="is_not_diffused"',
                     ]
        
        for mask in ['get_diffusion_mask_islands_partial_ligand', 'get_unconditional_diffusion_mask']:
            for dataset in ['pdb_aa', 'sm_complex']:
                loader_out = test_utils.loader_out_for_dataset(dataset, mask, overrides=overrides, epoch=0)
                indep, rfi, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, diffuser_out, item_context, conditions_dict = loader_out
                com = torch.mean(indep.xyz[~is_diffused,1,:], dim=0)
                zero = torch.tensor([0,0,0], dtype=indep.xyz.dtype)
                assertpy.assert_that(torch.norm(com - zero).item()).is_less_than(1e-4)

if __name__ == '__main__':
        unittest.main()
