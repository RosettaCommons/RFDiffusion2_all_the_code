import os
import shutil
import assertpy
import subprocess
import unittest

import pytest

import pandas as pd
from icecream import ic
# from deepdiff import DeepDiff

import error
import benchmark.sweep_hyperparam
pytestmark = pytest.mark.benchmark
ic.configureOutput(includeContext=True)

class TestBenchmark(unittest.TestCase):

    def test_arg_combos(self):
        for arg_str, want in [
            ('''
        a=1|2|3
        ''',
        [
            {'a':'1'},
            {'a':'2'},
            {'a':'3'}
        ]
        ),
        ('''
        (a=1 b=2)|(a=3 b=4)
        ''',
        [
            {'a':'1', 'b':'2'},
            {'a':'3', 'b':'4'},
        ]),
        ('''
        c=5
        (a=1 b=2)|(a=3 b=4)
        ''',
        [
            {'c':'5','a':'1', 'b':'2'},
            {'c':'5','a':'3', 'b':'4'},
        ]),

        ('''
        a=1
        (b=2)|(a=3 b=4)
        ''',
        [
            {'a':'1', 'b':'2'},
            {'a':'3', 'b':'4'},
        ]),
        ('''
        a=1
        (a=2 b=3)|(b=4|5)
        ''',
        [
            {'a':'2', 'b':'3'},
            {'a':'1', 'b':'4'},
            {'a':'1', 'b':'5'},
        ]),
        ('''
        (a=1)|(b=4|5)
        ''',
        [
            {'a':'1'},
            {'b':'4'},
            {'b':'5'},
        ]),
        ('''
        (a=1)|((b=4)|(b=5))
        ''',
        [
            {'a':'1'},
            {'b':'4'},
            {'b':'5'},
        ]),
        ('''
        a=1|
            2|
            3
        ''',
        [
            {'a':'1'},
            {'a':'2'},
            {'a':'3'},
        ]),
        ('''
        a=1
        ()|(a=2)
        ''',
        [
            {'a':'1'},
            {'a':'2'},
        ]),
        ('''
        ()|(a=2)
        a=1
        ''',
        [
            {'a':'1'},
            {'a':'1'},
        ]),
        ('''
        POST(()|(a=2))
        a=1
        ''',
        [
            {'a':'1'},
            {'a':'2'},
        ]),
        ('''
        a=1
        (b=3)|(a=2)
        ''',
        [
            {'a':'1', 'b':'3'},
            {'a':'2'},
        ]),
        ('''
        a=1
        (b=3)|(a=2|3)
        ''',
        [
            {'a':'1', 'b':'3'},
            {'a':'2'},
            {'a':'3'},
        ]),
        # TODO: Implement check such that this testcase returns an error, as it is somewhat nonsensical.
        # ('''
        # arg1=A|B
        # (arg1=C arg3=D)|(arg1=c arg3=d)
        # ''',
        # ['error']
        # ),
        ('''
        a=1
        b=2
        *benchmark/test_benchmarks.txt
        ''',
        [
            {'a':'1', 'b':'2', 'c':'3'},
            {'a':'1', 'b':'2', 'c':'4'},
        ]),
        ('''
        a=(ddd eee fff)
        b=2
        ''',
        [
            {'a':'ddd eee fff', 'b':'2'}
        ]),
        ]:
            with error.context(f'{arg_str=} {want=}'):
                got = benchmark.sweep_hyperparam.parse_arg_str(arg_str)
                ic(got, want)
                self.assertEqual(got, want)

    def test_subprocess_retcode(self):
        job = '''python raise_exception.py'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}') 
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_not_equal_to(0)
    
    def test_pipeline_completes(self):
        '''
        Tests that the pipeline runs end-to-end and produces the appropriate metrics
        for a toy input.
        '''

        expected_number_of_sequences = 2
        outdir = os.path.abspath('test_outputs/pipeline_0')
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        # Speed up the process by making the expected AF2 outputs, so that AF2 doesn't have to run.
        shutil.copytree('test_data/pipeline_0', outdir)

        job = f'''./benchmark/pipeline.py --config-name=pipeline_test in_proc=1 outdir={outdir}'''
        print(f'running job: {job}')
        proc = subprocess.run(job, shell=True)
        print(f'{proc=}')
        print(f'{proc.returncode=}')
        assertpy.assert_that(proc.returncode).is_equal_to(0)

        expected_metrics_csv_path = os.path.join(outdir, 'compiled_metrics.csv')
        assert os.path.exists(expected_metrics_csv_path)

        df = pd.read_csv(expected_metrics_csv_path)
        assertpy.assert_that(df.shape[0]).is_equal_to(expected_number_of_sequences)

        af2_success_metric = 'backbone_aligned_allatom_rmsd_af2_unideal_sym_resolved'
        assert df[af2_success_metric].notna().all(), f'expected non nans: {df[af2_success_metric]=}'


if __name__ == '__main__':
        unittest.main()
