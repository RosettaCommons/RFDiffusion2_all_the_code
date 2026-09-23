"""Small release check that can run from a public checkout."""
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def test_portable_runtime_and_cuda():
    assert torch.cuda.is_available(), 'A CUDA-capable GPU is required for the portable smoke test'
    x = torch.ones(4, device='cuda')
    assert float((x * 2).sum()) == 8.0


def test_portable_inference_invocation(tmp_path):
    """Run the zero-design inference path without internal data or checkpoints."""
    import hydra
    from hydra import compose, initialize
    from rf_diffusion import run_inference

    input_pdb = Path(__file__).parent / 'test_data' / '1qys.pdb'
    output = tmp_path / 'smoke'
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path='config/inference', job_name='portable_smoke'):
        conf = compose(config_name='aa_small.yaml', overrides=[
            'inference.num_designs=0',
            f'inference.input_pdb={input_pdb}',
            f'inference.output_prefix={output}',
        ])
    run_inference.main(conf)
    assert input_pdb.is_file()
