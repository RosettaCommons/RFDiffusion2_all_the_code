#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
"""Build the CD3epsilon and STAT5 phosphotyrosine peptide inputs."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyrosetta
from pyrosetta import pose_from_sequence


@dataclass(frozen=True)
class Target:
    annotated_sequence: str
    sequence: str
    ptr_residue: int
    filename: str
    description: str


TARGETS = {
    "cd3epsilon": Target(
        annotated_sequence="PVPNPDY[TYR:phosphorylated]EPIRKG",
        sequence="PVPNPDYEPIRKG",
        ptr_residue=7,
        filename="cd3epsilon_ptr.pdb",
        description="CD3epsilon PVPNPD-pY-EPIRKG peptide",
    ),
    "stat5": Target(
        annotated_sequence="TPVLAKAVDGY[TYR:phosphorylated]VKPQIKQVVP",
        sequence="TPVLAKAVDGYVKPQIKQVVP",
        ptr_residue=11,
        filename="stat5_pY694.pdb",
        description="STAT5 TPVLAKAVDG-pY-VKPQIKQVVP peptide (pY694)",
    ),
}

THREE_TO_ONE = {
    "ALA": "A",
    "ASP": "D",
    "GLU": "E",
    "GLY": "G",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "THR": "T",
    "VAL": "V",
    "TYR": "Y",
    "PTR": "Y",
}


def _is_hydrogen(line: str) -> bool:
    padded = line.ljust(80)
    element = padded[76:78].strip()
    if element:
        return element.upper() == "H"
    atom_name = padded[12:16].strip().lstrip("0123456789")
    return atom_name.startswith("H")


def _conect_edges(lines: list[str]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        serials = [int(value) for value in line.split()[1:]]
        source = serials[0]
        for destination in serials[1:]:
            if source != destination:
                edges.add(tuple(sorted((source, destination))))
    return edges


def _normalize_pdb(raw_pdb: Path, output_pdb: Path, target: Target) -> None:
    lines = raw_pdb.read_text().splitlines()
    atom_lines = [
        line
        for line in lines
        if line.startswith(("ATOM  ", "HETATM")) and not _is_hydrogen(line)
    ]

    old_to_new: dict[int, int] = {}
    atoms: dict[int, tuple[str, int, str]] = {}
    residues: list[tuple[int, str]] = []
    ptr_serials: set[int] = set()
    normalized_atoms: list[str] = []

    for new_serial, line in enumerate(atom_lines, start=1):
        padded = line.ljust(80)
        old_serial = int(padded[6:11])
        atom_name = padded[12:16].strip()
        residue_name = padded[17:20].strip()
        residue_number = int(padded[22:26])

        old_to_new[old_serial] = new_serial
        atoms[old_serial] = (atom_name, residue_number, residue_name)
        if not residues or residues[-1][0] != residue_number:
            residues.append((residue_number, residue_name))
        if residue_name == "PTR":
            ptr_serials.add(old_serial)

        normalized_atoms.append(
            f"{padded[:6]}{new_serial:5d}{padded[11:21]}B{padded[22:80]}".rstrip()
        )

    observed_sequence = "".join(THREE_TO_ONE[name] for _, name in residues)
    if observed_sequence != target.sequence:
        raise ValueError(
            f"Expected sequence {target.sequence}, generated {observed_sequence}"
        )
    if [number for number, name in residues if name == "PTR"] != [target.ptr_residue]:
        raise ValueError(f"Expected one PTR at residue {target.ptr_residue}")

    ptr_atom_names = {atoms[serial][0] for serial in ptr_serials}
    required_ptr_atoms = {"N", "CA", "C", "O", "P", "O1P", "O2P", "O3P"}
    if not required_ptr_atoms.issubset(ptr_atom_names):
        missing = ", ".join(sorted(required_ptr_atoms - ptr_atom_names))
        raise ValueError(f"Generated PTR is missing atoms: {missing}")

    ptr_edges = {
        edge
        for edge in _conect_edges(lines)
        if edge[0] in old_to_new
        and edge[1] in old_to_new
        and (edge[0] in ptr_serials or edge[1] in ptr_serials)
    }
    cross_edges = {
        edge
        for edge in ptr_edges
        if (edge[0] in ptr_serials) != (edge[1] in ptr_serials)
    }

    actual_cross_bonds = {
        frozenset((atoms[left][:2], atoms[right][:2])) for left, right in cross_edges
    }
    expected_cross_bonds = {
        frozenset(
            (
                ("C", target.ptr_residue - 1),
                ("N", target.ptr_residue),
            )
        ),
        frozenset(
            (
                ("C", target.ptr_residue),
                ("N", target.ptr_residue + 1),
            )
        ),
    }
    if actual_cross_bonds != expected_cross_bonds:
        raise ValueError(
            "Expected exactly the two native peptide-PTR bonds; "
            f"generated {sorted(actual_cross_bonds, key=repr)}"
        )

    conect_lines = [
        f"CONECT{old_to_new[left]:5d}{old_to_new[right]:5d}"
        for left, right in sorted(
            ptr_edges, key=lambda edge: (old_to_new[edge[0]], old_to_new[edge[1]])
        )
    ]
    output_lines = [
        f"REMARK  99 {target.description}; generated with PyRosetta.",
        *normalized_atoms,
        "TER",
        *conect_lines,
        "END",
    ]
    output_pdb.write_text("\n".join(output_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", *TARGETS),
        default="all",
        help="Peptide to prepare (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs",
        help="Destination directory (default: rf_diffusion/examples/inputs)",
    )
    args = parser.parse_args()

    selected = (
        TARGETS if args.target == "all" else {args.target: TARGETS[args.target]}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pyrosetta.init("-mute all")
    with tempfile.TemporaryDirectory(prefix="prepare_ptr_inputs_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        for target in selected.values():
            pose = pose_from_sequence(target.annotated_sequence)
            raw_pdb = temporary_path / target.filename
            pose.dump_pdb(str(raw_pdb))

            output_pdb = args.output_dir / target.filename
            _normalize_pdb(raw_pdb, output_pdb, target)
            print(
                f"Wrote {output_pdb} "
                f"({len(target.sequence)} residues, PTR B{target.ptr_residue}, "
                "2 peptide-PTR bonds)"
            )


if __name__ == "__main__":
    main()
