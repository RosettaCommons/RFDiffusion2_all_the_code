#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

# Currently this test must be run like ./test_chai.py, as it uses a different apptainer than
# the default rf_diffusion one that runs all other tests.  In addition, this test must
# be run on a GPU.

import os
import shutil
import unittest
from datetime import datetime
import subprocess
import glob

import pytest

pytestmark = [pytest.mark.requires_internal_data, pytest.mark.gpu, pytest.mark.slow]


def chai1_predict(
        input_path, 
        num_trunk_recycles = 3,
        num_diffn_timesteps = 200,
        tmp_dir_name=None,
):
    
    tmp_dir_name = tmp_dir_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    pdb_basename = os.path.basename(input_path).split('.')[0]
    chai1_test_dir = f'test_outputs/{tmp_dir_name}/chai_test_{pdb_basename}'
    if os.path.exists(chai1_test_dir):
        shutil.rmtree(chai1_test_dir)

    chai1_test_input_dir = os.path.join(chai1_test_dir, 'input')
    os.makedirs(chai1_test_input_dir)
    chai1_test_output_dir = os.path.join(chai1_test_dir, 'output')
    os.makedirs(chai1_test_output_dir)

    chai1_input_pdb_path = os.path.join(chai1_test_input_dir, os.path.basename(input_path))
    shutil.copy(input_path, chai1_input_pdb_path)

    chai_image = os.environ.get('RFDIFFUSION2_CHAI_IMAGE')
    chai_predict = os.environ.get('RFDIFFUSION2_CHAI_PREDICT')
    if not chai_image or not chai_predict:
        raise RuntimeError('Set RFDIFFUSION2_CHAI_IMAGE and RFDIFFUSION2_CHAI_PREDICT for Chai tests')
    cmd = f'''
apptainer run --nv {chai_image} {chai_predict} \
--output_dir {os.path.abspath(chai1_test_output_dir)} \
--num_trunk_recycles {num_trunk_recycles} --num_diffn_timesteps={num_diffn_timesteps}
    '''.strip()

    if os.path.splitext(input_path)[1] == '.fasta':
        cmd += f' --fasta_folder {os.path.abspath(chai1_test_input_dir)}'
    elif os.path.splitext(input_path)[1] == '.pdb':
        cmd += f' --pdb_folder {os.path.abspath(chai1_test_input_dir)}'
    else:
        raise Exception(f'{os.path.splitext(input_path)[1]=} not ".fasta" or ".pdb"')

    print(f'{cmd=}')

    # subprocess.run(cmd, shell=True)
    subprocess.run(cmd, shell=True, check=True)

    output_pdbs = sorted(glob.glob(os.path.join(chai1_test_output_dir, '*.pdb')))
    output_pdbs = [os.path.abspath(p) for p in output_pdbs]
    return output_pdbs

class TestPrediction(unittest.TestCase):

    @pytest.mark.nondeterministic
    def test_ligand_ccd_prediction(self):
        '''
        Tests that ligands are written as HETATM records in the output pdb.
        '''

        input_pdb = os.environ.get('RFDIFFUSION2_CHAI_TEST_PDB')
        if not input_pdb:
            pytest.skip('Set RFDIFFUSION2_CHAI_TEST_PDB to run the optional Chai integration test')
        output_pdbs = chai1_predict(input_pdb, tmp_dir_name = 'true_pdb')
        # Uncomment the below to run a shorter prediction for debugging.
        # output_pdbs = chai1_predict(input_pdb, tmp_dir_name = 'short_prediction', num_trunk_recycles=1, num_diffn_timesteps=5)
        print("output pdbs:\n" + "\n".join(output_pdbs))

        # Assert that there are hetatms
        with open(output_pdbs[0], 'r') as fh:
            pdb_str = fh.read()
        
        assert "HETATM" in pdb_str


if __name__ == '__main__':
        unittest.main()
