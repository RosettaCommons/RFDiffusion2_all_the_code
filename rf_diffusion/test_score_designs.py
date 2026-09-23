import shlex
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from rf_diffusion.benchmark import score_designs
from rf_diffusion.benchmark.util.af2_cache import cached_af2_prediction_is_reusable

pytestmark = pytest.mark.benchmark


def test_af2_binary_uses_resolved_runtime_paths():
    runtime = score_designs.Af2Runtime(
        image=Path('/images with spaces/mlfold.sif'),
        alphafold_dir=Path('/software with spaces/alphafold'),
        parameters=Path('/databases with spaces/model.pt'),
        databases_dir=Path('/databases with spaces'),
        projects_dir=Path('/projects with spaces'),
    )

    args = shlex.split(score_designs.get_af2_binary(runtime))

    assert args == [
        'apptainer',
        'run',
        '--nv',
        '--bind',
        '/net:/net',
        '--bind',
        '/projects with spaces:/projects',
        '--bind',
        '/databases with spaces:/databases',
        '/images with spaces/mlfold.sif',
    ]


def test_missing_af2_runtime_path_has_actionable_error(tmp_path):
    missing = tmp_path / 'missing.sif'

    with pytest.raises(FileNotFoundError, match='Unable to locate test image') as exc:
        score_designs._first_existing_path('test image', [missing], Path.is_file)

    assert str(missing) in str(exc.value)


def _pdb_ca_line(serial, residue_name, residue_number, altloc=' '):
    return (
        f'ATOM  {serial:5d}  CA {altloc}{residue_name:>3s} A{residue_number:4d}    '
        '   0.000   0.000   0.000  1.00  0.00           C  \n'
    )


def _valid_npz_payload(length=2):
    return {
        'plddt': np.full(length, 85.0),
        'pae': np.zeros((length, length), dtype=np.float32),
        'ptm': np.array(0.75),
    }


def test_cached_af2_prediction_requires_matching_sequence_and_valid_npz(tmp_path):
    input_pdb = tmp_path / 'input.pdb'
    matching_pdb = tmp_path / 'matching.pdb'
    stale_pdb = tmp_path / 'stale.pdb'
    cached_npz = tmp_path / 'prediction.npz'
    input_pdb.write_text(_pdb_ca_line(1, 'TYR', 1) + _pdb_ca_line(2, 'HIS', 2))
    matching_pdb.write_text(_pdb_ca_line(1, 'TYR', 10) + _pdb_ca_line(2, 'HIS', 11))
    stale_pdb.write_text(_pdb_ca_line(1, 'HIS', 1) + _pdb_ca_line(2, 'TYR', 2))
    np.savez(cached_npz, **_valid_npz_payload())

    assert cached_af2_prediction_is_reusable(input_pdb, matching_pdb, cached_npz)
    assert not cached_af2_prediction_is_reusable(input_pdb, stale_pdb, cached_npz)


def test_missing_or_corrupt_cached_af2_artifact_is_not_reused(tmp_path):
    input_pdb = tmp_path / 'input.pdb'
    matching_pdb = tmp_path / 'matching.pdb'
    invalid_pdb = tmp_path / 'invalid.pdb'
    cached_npz = tmp_path / 'prediction.npz'
    input_pdb.write_text(_pdb_ca_line(1, 'TYR', 1))
    matching_pdb.write_text(_pdb_ca_line(1, 'TYR', 1))
    invalid_pdb.write_text('END\n')

    assert not cached_af2_prediction_is_reusable(input_pdb, invalid_pdb, cached_npz)
    assert not cached_af2_prediction_is_reusable(
        input_pdb, tmp_path / 'missing.pdb', cached_npz
    )
    assert not cached_af2_prediction_is_reusable(input_pdb, matching_pdb, cached_npz)

    cached_npz.write_bytes(b'not an npz archive')
    assert not cached_af2_prediction_is_reusable(input_pdb, matching_pdb, cached_npz)


@pytest.mark.parametrize('missing_field', ['plddt', 'pae', 'ptm'])
def test_cached_af2_npz_requires_every_metric(tmp_path, missing_field):
    input_pdb = tmp_path / 'input.pdb'
    cached_pdb = tmp_path / 'cached.pdb'
    cached_npz = tmp_path / 'prediction.npz'
    pdb = _pdb_ca_line(1, 'TYR', 1) + _pdb_ca_line(2, 'HIS', 2)
    input_pdb.write_text(pdb)
    cached_pdb.write_text(pdb)
    payload = _valid_npz_payload()
    del payload[missing_field]
    np.savez(cached_npz, **payload)

    assert not cached_af2_prediction_is_reusable(input_pdb, cached_pdb, cached_npz)


@pytest.mark.parametrize(
    ('field', 'invalid_value'),
    [
        ('plddt', np.array([85.0])),
        ('pae', np.zeros((2, 1))),
        ('ptm', np.array([0.75])),
        ('plddt', np.array([85.0, np.nan])),
        ('pae', np.array([[0.0, np.inf], [0.0, 0.0]])),
        ('ptm', np.array(np.nan)),
        ('ptm', np.array(None, dtype=object)),
        ('plddt', np.array([-1.0, 85.0])),
        ('pae', np.array([[0.0, -1.0], [0.0, 0.0]])),
        ('ptm', np.array(1.1)),
    ],
)
def test_cached_af2_npz_rejects_invalid_metric_content(
    tmp_path, field, invalid_value
):
    input_pdb = tmp_path / 'input.pdb'
    cached_pdb = tmp_path / 'cached.pdb'
    cached_npz = tmp_path / 'prediction.npz'
    pdb = _pdb_ca_line(1, 'TYR', 1) + _pdb_ca_line(2, 'HIS', 2)
    input_pdb.write_text(pdb)
    cached_pdb.write_text(pdb)
    payload = _valid_npz_payload()
    payload[field] = invalid_value
    np.savez(cached_npz, **payload)

    assert not cached_af2_prediction_is_reusable(input_pdb, cached_pdb, cached_npz)


def test_af2_metrics_job_is_not_pruned_by_a_valid_prediction_cache(tmp_path):
    input_pdb = tmp_path / 'design.pdb'
    cached_dir = tmp_path / 'af2'
    cached_pdb = cached_dir / 'design.pdb'
    cached_npz = cached_dir / 'design.npz'
    input_list = tmp_path / 'inputs.list'
    outcsv = tmp_path / 'af2_metrics.csv.0'
    cached_dir.mkdir()
    input_pdb.write_text(_pdb_ca_line(1, 'TYR', 1) + _pdb_ca_line(2, 'HIS', 2))
    cached_pdb.write_text(_pdb_ca_line(1, 'TYR', 1) + _pdb_ca_line(2, 'HIS', 2))
    np.savez(cached_npz, **_valid_npz_payload())
    input_list.write_text(f'{input_pdb}\n')
    outcsv.write_text('name\ndesign\n')

    conf = OmegaConf.create(
        {
            'filenames': str(input_list),
            'datadir': str(tmp_path),
            'trb_dir': str(tmp_path),
            'chunk': 1,
            'tmp_pre': 'score.list',
            'run': 'af2',
            'slurm': {
                'submit': True,
                'J': 'cache-regression',
                'p': 'gpu',
                'gres': 'gpu:1',
                'in_proc': False,
                'keep_logs': False,
            },
        }
    )

    with (
        mock.patch.object(score_designs, 'get_af2_binary', return_value='af2-runtime'),
        mock.patch.object(
            score_designs.slurm_tools, 'array_submit', return_value=(-1, None)
        ) as array_submit,
    ):
        score_designs.main.__wrapped__(conf)

    array_submit.assert_called_once()
    assert 'already_ran' not in array_submit.call_args.kwargs
    submitted_job_list = Path(array_submit.call_args.args[0])
    assert 'af2_metrics.py' in submitted_job_list.read_text()
