import shlex
from pathlib import Path

import pytest

from rf_diffusion.benchmark import mpnn_designs_v2


def test_resolve_mpnn_runtime_prefers_repository_checkpoints(tmp_path):
    rf2_root = tmp_path / 'lib' / 'RFdiffusion2'
    script = rf2_root / 'fused_mpnn' / 'run.py'
    paths_module = rf2_root / 'paths.py'
    weights = rf2_root / 'rf_diffusion' / 'third_party_model_weights' / 'ligand_mpnn'
    ligand_checkpoint = weights / 's25_r010_t300_p.pt'
    packer_checkpoint = weights / 's_300756.pt'
    for path in (script, paths_module, ligand_checkpoint, packer_checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    runtime = mpnn_designs_v2._resolve_mpnn_runtime(tmp_path)

    assert runtime.script == script
    assert runtime.pythonpath == rf2_root
    assert runtime.ligand_checkpoint == ligand_checkpoint
    assert runtime.packer_checkpoint == packer_checkpoint


def test_mpnn_command_sets_pythonpath_and_explicit_checkpoints():
    runtime = mpnn_designs_v2.MpnnRuntime(
        script=Path('/repo with spaces/fused_mpnn/run.py'),
        pythonpath=Path('/repo with spaces'),
        ligand_checkpoint=Path('/weights/ligand.pt'),
        packer_checkpoint=Path('/weights/packer.pt'),
    )

    args = shlex.split(mpnn_designs_v2.get_binary(in_proc=True, runtime=runtime))

    assert args[:5] == [
        'apptainer',
        'exec',
        '--nv',
        '--env',
        'PYTHONPATH=/repo with spaces',
    ]
    assert args[5:7] == ['--bind', '/net:/net']
    assert '/net/software/:/net/software/' not in args
    assert args[-5:] == [
        'python',
        '-u',
        '/repo with spaces/fused_mpnn/run.py',
        '--checkpoint_ligand_mpnn=/weights/ligand.pt',
        '--checkpoint_path_sc=/weights/packer.pt',
    ]


def test_missing_mpnn_runtime_file_has_actionable_error(tmp_path):
    missing = tmp_path / 'missing.pt'

    with pytest.raises(FileNotFoundError, match='Unable to locate test checkpoint') as exc:
        mpnn_designs_v2._first_existing_file('test checkpoint', [missing])

    assert str(missing) in str(exc.value)
