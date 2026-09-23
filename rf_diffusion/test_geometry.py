#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#apptainer exec --nv exec/rf_diffusion_aa.sif pytest test_geometry.py --disable-warnings -s -m "not nondeterministic"

import unittest
from icecream import ic
import tempfile

import numpy as np
import torch
import glob
import time
import pandas as pd


import rf_diffusion
from rf_diffusion.inference.utils import parse_pdb
from rf_diffusion.write_file import writepdb
from rf_diffusion.parsers import parse_pdb_lines_target
from rf_diffusion.dev import analyze

from benchmark.util import geometry_metrics_utils
from benchmark.per_sequence_metrics import geometry
from benchmark.util.geometry_metrics_run import compute_geometry_df

PKG_DIR = rf_diffusion.__path__[0]
ic.configureOutput(includeContext=True)
WRITE=False

class TestGeometryMetricsUtils(unittest.TestCase):
    parsed_pdb = parse_pdb(f'{PKG_DIR}/test_data/1qys.pdb')
    # motif_idxs = np.array([37, 48, 20])
    # motif_idxs = np.array([20, 37, 48, 38, 36])
    motif_idxs = np.arange(parsed_pdb['seq'].shape[0])
    write=WRITE

    def test_angle_to_xyz_conversion(self):
        xyz = torch.tensor(self.parsed_pdb['xyz'][self.motif_idxs])[...,:14,:3]
        seq = torch.tensor(self.parsed_pdb['seq'][self.motif_idxs])
        mask = torch.tensor(self.parsed_pdb['mask'][self.motif_idxs])[...,:14]

        angles = geometry_metrics_utils.xyz_to_angles(xyz, seq)
        xyz_reconstructed = geometry_metrics_utils.angles_to_xyz(angles, seq)
        xyz_reconstructed = [geometry_metrics_utils.align_rotamer_to_tip(
            xyz[i], mask[i], xyz_reconstructed[i][None], [0,1,2,3]
        ) for i in range(len(xyz_reconstructed))]
        xyz_reconstructed = torch.concatenate(xyz_reconstructed, dim=0)
        if self.write:
            writepdb(f'{PKG_DIR}/test_data/test3_out.pdb',
                atoms=torch.tensor(xyz_reconstructed),
                seq=torch.tensor(seq),
                idx_pdb = torch.tensor([l for _, l in enumerate(self.motif_idxs)]),
            )

        rmsd = float(torch.sqrt(torch.mean((xyz - xyz_reconstructed)**2, dim=(-1,-2))).mean())
        print(f'xyz -> angles -> xyz reconstruction RMSD: {rmsd}')
        self.assertLess(rmsd, 0.05)

    def test_chi_to_xyz_conversion(self):
        xyz = torch.tensor(self.parsed_pdb['xyz'][self.motif_idxs])[...,:14,:3]
        seq = torch.tensor(self.parsed_pdb['seq'][self.motif_idxs])
        mask = torch.tensor(self.parsed_pdb['mask'][self.motif_idxs])[...,:14]

        angles = geometry_metrics_utils.xyz_to_angles(xyz, seq)
        chis = angles[:, 3:7].clone()
        del angles
        xyz_reconstructed = geometry_metrics_utils.chis_to_xyz(chis, seq)
        xyz_reconstructed = [geometry_metrics_utils.align_rotamer_to_tip(
            xyz[i], mask[i], xyz_reconstructed[i][None]
        ) for i in range(len(xyz_reconstructed))]
        xyz_reconstructed = torch.concatenate(xyz_reconstructed, dim=0)
        
        if self.write:
            writepdb(f'{PKG_DIR}/test_data/test4_out.pdb', 
                atoms=torch.tensor(xyz_reconstructed),
                seq=torch.tensor(seq),
                idx_pdb=torch.tensor([l for _, l in enumerate(self.motif_idxs)]),
            )
        
        rmsd = float(torch.sqrt(torch.mean((xyz - xyz_reconstructed)**2, dim=(-1,-2))).mean())
        print("xyz -> chis -> xyz reconstruction RMSD:", rmsd)
        self.assertLess(rmsd, 0.25)

    def test_find_ideal_irots(self):
        for name, motif_idxs, ground_truth_pass in [
            ('chorismate_enzyme_design_pass', None, True),
            ('chorismate_enzyme_design_fail', None, False),
            ('1qys', [20, 37, 48, 38, 36], True),
        ]:
            pdb = f'{PKG_DIR}/test_data/{name}.pdb'
            parsed = parse_pdb_lines_target(open(pdb, 'r').readlines(), parse_hetatom=True)
            if motif_idxs is None:
                row = analyze.make_row_from_traj(pdb)
                trb = analyze.get_trb(row)
                motif_idxs = trb['con_hal_pdb_idx']
                motif_idxs = [parsed['pdb_idx'].index(idx) for idx in motif_idxs]

            t0 = time.time()
            parsed_ideal, o = geometry_metrics_utils.find_ideal_irots(parsed=parsed, motif_idxs=motif_idxs, return_stack=self.write)
            xyz_stack = o.pop('xyz_stack') if self.write else None
            idealization_rmsd = np.array([v['allatom_rmsd'] for v in o.values()]).mean()
            fraction_changed = np.array([v['has_changed'] for v in o.values()]).mean()

            print('Idealized motif in {:.2f}s'.format(time.time() - t0), 'mean idealization rmsd:', idealization_rmsd, 'fraction changed:', fraction_changed)

            if ground_truth_pass:
                self.assertLess(idealization_rmsd, 3)
                self.assertLess(fraction_changed, 0.9)
            else:
                self.assertGreater(idealization_rmsd, 0.2)
                self.assertGreater(fraction_changed, 0.5)

        if self.write:
            xyz = torch.tensor(parsed_ideal['xyz'])
            seq = torch.tensor(parsed_ideal['seq'])
            writepdb(f'{PKG_DIR}/test_data/test5_out.pdb', 
                atoms=xyz,
                seq=seq,
                idx_pdb=torch.tensor([int(l) for _, (_, l) in enumerate(parsed_ideal['pdb_idx'])]),
            )

            xyz_stack = torch.concatenate(xyz_stack, dim=0)
            seq = torch.tensor([0]*len(xyz_stack))
            writepdb(f'{PKG_DIR}/test_data/test5_1_out.pdb', 
                atoms=(xyz_stack),
                seq=(seq),
                idx_pdb=torch.tensor([l for l in range(1, len(xyz_stack)+1)]),
            )
            print("wrote", f'{PKG_DIR}/test_data/test5_1_out.pdb')


class TestGeometry(unittest.TestCase):
    write=WRITE

    def test_geometry_result(self):
        
        for name, motif_idxs, ground_truth_pass in [
            ('chorismate_enzyme_design_pass', None, True),
            ('chorismate_enzyme_design_fail', None, False),
            # ('1qys', [37, 20, 48], True),
        ]:
            pdb = f'{PKG_DIR}/test_data/{name}.pdb'
            print('Testing on:', name, ground_truth_pass, motif_idxs)
            t0 = time.time()
            o = geometry(pdb, idx=motif_idxs)
            print(o)
            print('time to compute geometry:', time.time() - t0)
            self.assertEqual(not o['geometry.fails'], ground_truth_pass)

    def test_run_geometry_and_recompile(self):
        tmpdir = tempfile.mkdtemp()
        pdbs = glob.glob(f'{PKG_DIR}/test_data/chorismate_enzyme_design_*.pdb')
        df = compute_geometry_df(pdbs)
        df.to_csv(f'{tmpdir}/geometry_metrics.csv', index=False)
        df_reloaded = pd.read_csv(f'{tmpdir}/geometry_metrics.csv')
        df_reloaded = df_reloaded.apply(geometry_metrics_utils.compile_geometry_from_rows, axis=1)
        self.assertEqual(
            df['geometry.fails'].values.tolist(), df_reloaded['geometry.fails'].values.tolist()
        )
        for a, b in zip(df['geometry.score'].values, df_reloaded['geometry.score'].values):
            self.assertAlmostEqual(a, b, places=7)

if __name__ == '__main__':
    unittest.main()
