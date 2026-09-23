from pathlib import Path
from zipfile import BadZipFile

import numpy as np


_REQUIRED_NPZ_FIELDS = frozenset({'plddt', 'pae', 'ptm'})


def pdb_residue_names(pdb_path) -> tuple[str, ...]:
    """Return the ordered protein residue names represented by PDB C-alpha atoms."""
    residue_names = []
    seen = set()
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
                continue
            if line[16:17] not in {' ', 'A'}:
                continue
            residue_id = (line[21:22], line[22:26], line[26:27])
            if residue_id in seen:
                continue
            seen.add(residue_id)
            residue_names.append(line[17:20].strip())
    if not residue_names:
        raise ValueError(f'No protein C-alpha atoms found in {pdb_path}')
    return tuple(residue_names)


def cached_af2_sequence_matches(input_pdb, cached_pdb) -> bool:
    """Whether a cached AF2 structure was generated for the input sequence."""
    try:
        return pdb_residue_names(input_pdb) == pdb_residue_names(cached_pdb)
    except (OSError, ValueError):
        return False


def _is_finite_numeric_array(value, shape) -> bool:
    value = np.asarray(value)
    return (
        value.shape == shape
        and np.issubdtype(value.dtype, np.number)
        and bool(np.isfinite(value).all())
    )


def cached_af2_prediction_is_reusable(input_pdb, cached_pdb, cached_npz) -> bool:
    """Whether cached AF2 prediction artifacts are complete and internally valid."""
    try:
        input_residues = pdb_residue_names(input_pdb)
        if input_residues != pdb_residue_names(cached_pdb):
            return False

        residue_count = len(input_residues)
        with np.load(cached_npz, allow_pickle=False) as archive:
            if not _REQUIRED_NPZ_FIELDS.issubset(archive.files):
                return False
            plddt = archive['plddt']
            pae = archive['pae']
            ptm = archive['ptm']

        if not _is_finite_numeric_array(plddt, (residue_count,)):
            return False
        if not _is_finite_numeric_array(pae, (residue_count, residue_count)):
            return False
        if not _is_finite_numeric_array(ptm, ()):
            return False
        return (
            bool(((0 <= plddt) & (plddt <= 100)).all())
            and bool((pae >= 0).all())
            and bool(0 <= ptm <= 1)
        )
    except (AttributeError, BadZipFile, EOFError, KeyError, OSError, TypeError, ValueError):
        return False
