#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

import os
import sys
from collections import defaultdict
import copy
from icecream import ic

from functools import partial
import warnings
import mdtraj as md

# Hack for autobenching
PKG_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SE3_DIR = os.path.join(PKG_DIR, 'lib/se3_flow_matching')
sys.path.append(SE3_DIR)

import pandas as pd
import fire
from tqdm import tqdm
import torch
import numpy as np
import scipy
from pathlib import Path
import glob

from ipd.dev import safe_eval
from rf_diffusion import aa_model
from rf_diffusion.dev import analyze
from rf_diffusion.inference import utils
import rf_diffusion.dev.analyze
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.dev import benchmark as bm
from rf_diffusion import loss
import rf_diffusion.atomization_primitives
from rf2aa.util import rigid_from_3_points

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio import PDB
from rf_diffusion.dev import biotite_tools as bt
import biotite

from rf_diffusion.dev.show_bench import get_last_px0
from rf_diffusion.benchmark.util import geometry_metrics_utils
from rf_diffusion.benchmark.util.geometry_metrics_utils import (
    geometry_inner, 
    compile_geometry_dict, 
    junction_bond_len_inner, 
    rmsd
)

from rf_diffusion.parsers import parse_pdb_lines_target
from rf_diffusion.dev import show_bench

script_dir = os.path.dirname(os.path.realpath(__file__))


pdb_parser = PDBParser(PERMISSIVE=0, QUIET=1)
default_probe_radius=1.4

def main(pdb_names_file, outcsv=None, metric='default'):
    ic(__name__)
    thismodule = sys.modules[__name__]
    metric_f = getattr(thismodule, metric)

    with open(pdb_names_file, 'r') as fh:
        pdbs = [pdb.strip() for pdb in fh.readlines()]
    
    df = get_metrics(pdbs, metric_f)

    print(f'Outputting computed metrics dataframe with shape {df.shape} to {outcsv}')
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv)

def get_metrics(pdbs, metric):
    records = []
    for pdb in tqdm(pdbs):
        print(f'Calculating metrics for: {pdb}')
        record = metric(pdb)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df

def get_aligner(
        f, # from [L, 3]
        t, # to [L, 3]
        ):
    T = analyze.register_full_atom(f[:, None, :], t[:, None, :])
    def T_flat(f):
        f_aligned = T(f[:, None, :])
        return f_aligned[:, 0, :]

    return T_flat

def aligned_rmsd(
        f, # from [L, 3]
        t, # to [L, 3]
        ):
    T = get_aligner(f, t)
    f_aligned = T(f)
    return rmsd(f_aligned, t)   

def catalytic_constraints_mpnn_packed(pdb):
    out = catalytic_constraints_inner(pdb, mpnn_packed=True)
    out = {f'catalytic_constraints.mpnn_packed.{k}': v for k,v in out.items()}
    return out

def catalytic_constraints(pdb):
    out = catalytic_constraints_inner(pdb, mpnn_packed=False)
    out = {f'catalytic_constraints.raw.{k}': v for k,v in out.items()}
    return out

def make_ligand_pids_unique(point_ids):
    new_pids = []
    ligand_pid_counts = defaultdict(int)
    for pid in point_ids:
        new_pid = pid
        if pid.startswith('L'):
            new_pid = f'{pid}_{ligand_pid_counts[pid]}'
            ligand_pid_counts[pid] += 1
        new_pids.append(new_pid)
    return new_pids


def catalytic_constraints_inner(pdb, mpnn_packed: bool):
    out = {}
    row = analyze.make_row_from_traj(pdb[:-4])
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']
    af2_pdb = analyze.get_af2(row)
    if mpnn_packed:
        des_pdb = analyze.get_design_pdb(row)
    else:
        des_pdb = analyze.get_diffusion_pdb(row)
    ref_pdb = analyze.get_input_pdb(row)

    indeps = {}
    indeps_a = {}
    atomizers = {}
    point_ids = {}
    ligand = row['inference.ligand']
    for name, pdb in [
        ('af2', af2_pdb),
        ('ref', ref_pdb),
        ('des', des_pdb),
    ]:
        if not os.path.exists(pdb):
            warnings.warn(f'{name} pdb: {pdb} for design {des_pdb} does not exist')
            return {}
        # f['name'] = utils.process_target(pdb, parse_hetatom=True, center=False)
        indeps[name] = aa_model.make_indep(pdb, ligand=None if name == 'af2' else ligand)
        is_atomized = ~indeps[name].is_sm
        atomization_state = aa_model.get_atomization_state(indeps[name])
        atomizers[name] = aa_model.AtomizeResidues(atomization_state, is_atomized)
        indeps_a[name] = atomizers[name].atomize(indeps[name])
        point_ids[name] = aa_model.get_point_ids(indeps_a[name], atomizers[name])
        point_ids[name] = make_ligand_pids_unique(point_ids[name])
    
    trb = analyze.get_trb(row)
    heavy_motif_atoms = {}
    for ref_idx0, (ref_chain, ref_idx_pdb) in zip(trb['con_ref_idx0'], trb['con_ref_pdb_idx']):
        aa = indeps['ref'].seq[ref_idx0]
        heavy_atom_names = aa_model.get_atom_names(aa)
        heavy_motif_atoms[f'{ref_chain}{ref_idx_pdb}'] = heavy_atom_names

    contig_atoms = row['contigmap.contig_atoms']
    if contig_atoms is not None:
        contig_atoms = safe_eval(contig_atoms)
        contig_atoms = {k:v.split(',') for k,v in contig_atoms.items()}
    else:
        contig_atoms = heavy_motif_atoms
    
    def get_pids(name, *getters):
        pids = []
        for g in getters:
            pids.extend(g(point_ids[name]))
        return np.array(pids)
    
    def get_ii(name, *getters):
        pids = get_pids(name, *getters)
        i_by_pid = {pid: i for i, pid in enumerate(point_ids[name])}
        i_by_pid_v = np.vectorize(i_by_pid.__getitem__, otypes=[int])
        ii = i_by_pid_v(pids)
        return ii
    
    def xyz_by_id(name, ii):
        return indeps_a[name].xyz[ii, 1]
    
    def xyz(name, *getters):
        ii = get_ii(name, *getters)
        return xyz_by_id(name, ii)
    
    def zip_safe(*args):
        assert len(set(map(len, args))) == 1
        return zip(*args)

    def get_motif(_, ref: bool, contig_atoms=contig_atoms):
        pids = []
        idx0 = trb[f'con_{"ref" if ref else "hal"}_idx0']
        for (chain, ref_pdb_i), hal_i in zip_safe(
            trb['con_ref_pdb_idx'],
            idx0,
        ):
            atom_names = contig_atoms[f'{chain}{ref_pdb_i}']
            for a in atom_names:
                pids.append(f'A{hal_i}-{a}')
        return pids
    
    get_ref_motif = partial(get_motif, ref=True)
    get_des_motif = partial(get_motif, ref=False)
    
    def get_ligand(pids):
        return [pid for pid in pids if pid.startswith('L')]
    
    # 1. All-atom RMSD of scaffolded region, diffusion backbone to input motif < 1.0 Å
    # NOTE: Aligned on all heavy motif atoms, RMSD calculated on heavy motif atoms + ligand
    motif_des = xyz('des', get_des_motif)
    motif_ref = xyz('ref', get_ref_motif)
    T_motif_des_to_ref = get_aligner(motif_des, motif_ref)
    motif_and_ligand_des = xyz('des', get_des_motif, get_ligand)
    motif_and_ligand_ref = xyz('ref', get_ref_motif, get_ligand)
    ref_aligned_motif_and_ligand_des = T_motif_des_to_ref(motif_and_ligand_des)
    out['des_ref_motif_aligned_motif_ligand_rmsd'] = rmsd(ref_aligned_motif_and_ligand_des, motif_and_ligand_ref).item()
    out['criterion_1_metric'] = out['des_ref_motif_aligned_motif_ligand_rmsd']
    out['criterion_1_cutoff'] = 1.
    out['criterion_1'] = out['des_ref_motif_aligned_motif_ligand_rmsd'] < 1

    # 2. All-atom RMSD of scaffolded region, AF2 prediction to input motif < 1.0 Å
    motif_af2 = xyz('af2', get_des_motif)
    T = get_aligner(motif_des, motif_af2)
    af2_aligned_motif_des = T(motif_des)
    out['af2_des_motif_aligned_motif_rmsd'] = rmsd(af2_aligned_motif_des, motif_af2).item()
    out['criterion_2_metric'] = out['af2_des_motif_aligned_motif_rmsd']
    out['criterion_2_cutoff'] = 1.
    out['criterion_2'] = out['af2_des_motif_aligned_motif_rmsd'] < 1

    # 3. Cα RMSD of design model to AF2 prediction < 1.5 A
    def get_ca(pids):
        return [pid for pid in pids if pid.startswith('A') and pid.endswith('-CA')]
    def get_backbone(pids):
        return [pid for pid in pids if pid.startswith('A') and
                (pid.endswith('-CA') or pid.endswith('-C') or pid.endswith('-N') or pid.endswith('-O'))]
    
    ca_af2 = xyz('af2', get_ca)
    ca_des = xyz('des', get_ca)
    T = get_aligner(ca_des, ca_af2)
    af2_aligned_ca_des = T(ca_des)
    out['af2_des_ca_aligned_ca_rmsd'] = rmsd(af2_aligned_ca_des, ca_af2).item()
    out['criterion_3_metric'] = out['af2_des_ca_aligned_ca_rmsd']
    out['criterion_3_cutoff'] = 1.5
    out['criterion_3'] = out['af2_des_ca_aligned_ca_rmsd'] < 1.5

    # 4. No backbone-ligand clashes* in diffusion output
    # In diffusion model: Scaffolded region will be aligned (all atom) to input motif. If any backbone (N,CA,C)
    #    atom is < 2.0 Å from a ligand atom, the diffusion model is clashing.
    ligand_ref = xyz('ref', get_ligand)
    T = get_aligner(motif_ref, motif_des)
    des_aligned_ref_ligand = T(ligand_ref)
    backbone_des = xyz('des', get_backbone)

    ligand_bb_dist = scipy.spatial.distance.cdist(des_aligned_ref_ligand, backbone_des)
    out['des_ref_motif_aligned_ligand_bb_dist'] = ligand_bb_dist.min()
    out['criterion_4_metric'] = out['des_ref_motif_aligned_ligand_bb_dist']
    out['criterion_4_cutoff'] = 2.
    out['criterion_4'] = (ligand_bb_dist > 2).all()

    # 5. No backbone-ligand clashes* in AF2 prediction
    # In AF2 prediction: Design model will be aligned (Cα) to AF2 prediction. If any backbone atom is < 2.0 Å 
    #    from a ligand atom, the AF2 prediction is clashing.
    T = get_aligner(motif_des, motif_af2)
    ligand_des = xyz('des', get_ligand)
    af2_aligned_ligand_des = T(ligand_des)
    ligand_bb_dist = scipy.spatial.distance.cdist(af2_aligned_ligand_des, backbone_des)
    out['des_af2_motif_aligned_ligand_bb_dist'] = ligand_bb_dist.min()
    out['criterion_5_metric'] = out['des_af2_motif_aligned_ligand_bb_dist']
    out['criterion_5_cutoff'] = 2.
    out['criterion_5'] = (ligand_bb_dist > 2).all()

    # 6. AF2 mean plDDT > 0.7
    af2_npz = af2_pdb[:-4] + '.npz'
    af2_metrics = np.load(af2_npz)
    out['criterion_6_metric'] = np.mean(af2_metrics['plddt'])
    out['criterion_6_cutoff'] = 0.7
    out['criterion_6'] = np.mean(af2_metrics['plddt']) > 0.7

    # Extras
    # get_ref_motif_all_heavy = partial(get_motif, ref=True, contig_atoms=heavy_motif_atoms)
    get_des_motif_all_heavy = partial(get_motif, ref=False, contig_atoms=heavy_motif_atoms)
    motif_des_all_heavy = xyz('des', get_des_motif_all_heavy)
    motif_af2_all_heavy = xyz('af2', get_des_motif_all_heavy)
    T = get_aligner(motif_des_all_heavy, motif_af2_all_heavy)
    af2_aligned_motif_des_all_heavy = T(motif_des_all_heavy)
    out['af2_des_motif_aligned_motif_rmsd_all_heavy'] = rmsd(af2_aligned_motif_des_all_heavy, motif_af2_all_heavy).item()

    # # Testing
    # motif_des = xyz('des', get_des_motif)
    # motif_ref = xyz('ref', get_ref_motif)
    # T = get_aligner(motif_des, motif_ref)
    # motif_and_ligand_des = xyz('des', get_des_motif)
    # motif_and_ligand_ref = xyz('ref', get_ref_motif)
    # aligned_motif_and_ligand_des = T(motif_and_ligand_des)
    # out['des_ref_motif_aligned_motif_ligand_rmsd'] = rmsd(aligned_motif_and_ligand_des, motif_and_ligand_ref)
    # ic(pdb, des_pdb)
    # ic(out)
    # raise Exception('storp')
    return out

def default(pdb):
    record = {}

    row = analyze.make_row_from_traj(pdb[:-4])
    # ic(row['mpnn_index'])
    record['name'] = row['name']
    record['mpnn_index'] = row['mpnn_index']

    design_pdb = analyze.get_design_pdb(row)
    design_info = utils.process_target(design_pdb, center=False, parse_hetatom=True)
    des = design_info['xyz_27']
    mask = design_info['mask_27']
    des[~mask] = torch.nan
    dgram = torch.sqrt(torch.sum((des[None, None,:,:,:] - des[:,:,None,None, :]) ** 2, dim=-1))
    dgram = torch.nan_to_num(dgram, 999)

    # Ignore backbone-backbone distance, as ligandmpnn is not responsible for this.
    bb2bb = torch.full(dgram.shape, False)
    bb2bb[:, :4, :, :4] = True
    # ic(bb2bb.dtype)
    dgram[bb2bb] = 999

    dgram = dgram.min(dim=3)[0]
    dgram = dgram.min(dim=1)[0]
    # ic(dgram)
    # ic(dgram.shape)

    dgram.fill_diagonal_(999)
    min_dist = dgram.min()
    record['res_to_res_min_dist'] = min_dist.item()
    is_dist = torch.ones_like(dgram).bool()
    is_dist = torch.triu(is_dist, diagonal=1)
    dists = dgram[is_dist]
    clash_dist = 2 # roughly 2 VDW radii
    
    n_pair_clash = torch.sum(dists < clash_dist).item()
    record['n_pair_clash'] = n_pair_clash
    res_clash = (dgram < clash_dist).any(dim=-1)
    record['n_res_clash'] = res_clash.sum().item()
    record['fraction_res_clash'] = res_clash.float().mean().item()
    record['res_clash'] = res_clash.tolist()

    diffusion_pdb = analyze.get_diffusion_pdb(row)
    diffusion_info = utils.process_target(diffusion_pdb, center=False, parse_hetatom=True)
    diff = diffusion_info['xyz_27']
    diff[~mask] = torch.nan

    trb = analyze.get_trb(row)
    des_motif = des[trb['con_hal_idx0']]
    diff_motif = diff[trb['con_hal_idx0']]
    flat_des = des_motif[~des_motif.isnan().any(dim=-1)]
    flat_diff = diff_motif[~diff_motif.isnan().any(dim=-1)]

    mpnn_motif_dist = ((flat_des - flat_diff) ** 2).sum(dim=-1) ** 0.5
    record['motif_ideality_diff'] = mpnn_motif_dist.mean().item()
    return record

def guidepost(pdb):
    row = analyze.make_row_from_traj(pdb[:-4])
    o = {}
    trb = analyze.get_trb(row)
    config = trb['config']

    if config['inference'].get('contig_as_guidepost', False):

        bb_i = np.array(trb['con_hal_idx0'])
        gp_i = np.array(list(trb['motif'].keys()))

        deatomized_xyz, is_het = get_last_px0(row)
        het_idx = is_het.nonzero()[0]
        gp_i = gp_i[~np.isin(gp_i, het_idx)]
        gp_motif = deatomized_xyz[gp_i]
        bb_motif = deatomized_xyz[bb_i]
        ca_dist = np.linalg.norm(gp_motif[:, 1] - bb_motif[:, 1], axis=-1)
        bb_dist = np.linalg.norm(gp_motif[:, :3].reshape((-1, 3)) - bb_motif[:, :3].reshape((-1, 3)), axis=-1)
    else:
        ca_dist = np.zeros((1,), dtype=float)
        bb_dist = np.zeros((1,), dtype=float)
    o['ca_dist.max'] = np.max(ca_dist)
    o['ca_dist.min'] = np.min(ca_dist)
    o['ca_dist.mean'] = np.mean(ca_dist)
    o['ca_dist.rmsd'] = np.mean(ca_dist**2)**0.5
    o['bb_dist.max'] = np.max(bb_dist)
    o['bb_dist.min'] = np.min(bb_dist)
    o['bb_dist.mean'] = np.mean(bb_dist)
    o['bb_dist.rmsd'] = np.mean(bb_dist**2)**0.5
    o = {f'guidepost.{k}':v for k,v in o.items()}
    o['name'] = row['name']
    return o

def guidepost_af2_rmsd(pdb):
    row = analyze.make_row_from_traj(pdb[:-4])
    o = {}
    trb = analyze.get_trb(row)
    config = trb['config']

    af2_bb_rmsd = -1.0
    if config['inference']['contig_as_guidepost']:

        bb_i = np.array(trb['con_hal_idx0'])
        gp_i = np.array(list(trb['motif'].keys()))

        af2_pdb = analyze.get_af2(row)
        af2_indep = aa_model.make_indep(af2_pdb, ligand=None)
        af2_xyz = af2_indep.xyz

        deatomized_xyz, is_het = get_last_px0(row)
        het_idx = is_het.nonzero()[0]
        gp_i = gp_i[~np.isin(gp_i, het_idx)]
        gp_motif = deatomized_xyz[gp_i]
        af2_motif = af2_xyz[bb_i]
        af2_bb_flat = af2_motif[:, :3].reshape((-1, 3))
        gp_bb_flat = gp_motif[:, :3].reshape((-1, 3))
        gp_bb_flat = torch.tensor(gp_bb_flat, dtype=torch.float32)

        T = get_aligner(af2_bb_flat, gp_bb_flat)
        af2_bb_flat_aligned = T(af2_bb_flat)
        af2_bb_rmsd = rmsd(af2_bb_flat_aligned, gp_bb_flat).item()
    
    o['guidepost_af2_rmsd'] = af2_bb_rmsd
    o['name'] = row['name']
    o['mpnn_index'] = row['mpnn_index']
    return o

def junction_bond_len(xyz, is_motif, idx):
    '''
        Args:
            xyz: [L, 14, 3] protein only xyz
            is_motif: [L] boolean motif mask
            idx: [L] pdb index
    '''
    sig_len=0.02
    ideal_NC=1.329
    blen_CN  = loss.length(xyz[:-1,2], xyz[1:,0])
    CN_loss = torch.clamp( torch.abs(blen_CN - ideal_NC) - sig_len, min=0.0 )

    pairsum = is_motif[:-1].double() + is_motif[1:].double()
    pairsum[idx[:-1] - idx[1:] != -1] = -1

    junction = pairsum == 1
    intra_motif = pairsum == 2
    intra_diff = pairsum == 0

    return {
        'junction_CN_loss': CN_loss[junction].mean().item(),
        'intra_motif_CN_loss': CN_loss[intra_motif].mean().item(),
        'intra_diff_CN_loss': CN_loss[intra_diff].mean().item()
    }


def junction_cn(pdb):
    row = analyze.make_row_from_traj(pdb[:-4])
    trb = analyze.get_trb(row)
    des_pdb = analyze.get_diffusion_pdb(row)
    indep = aa_model.make_indep(des_pdb, row['inference.ligand'])
    is_motif = torch.zeros(indep.length()).bool()
    is_motif[trb['con_hal_idx0']] = True
    # ic(is_motif)
    o = junction_bond_len_inner(
        indep.xyz[~indep.is_sm],
        is_motif[~indep.is_sm],
        indep.idx[~indep.is_sm])
    o['name'] = row['name']
    return o

def backbone(pdb):
    record = {}
    row = rf_diffusion.dev.analyze.make_row_from_traj(pdb[:-4])
    record['name'] = row['name']

    if not analyze.is_rfd(row):
        traj_metrics = bm.get_inference_metrics_base(bm.get_trb_path(row), regenerate_cache=True)
        traj_t0_metrics = traj_metrics[traj_metrics.t==traj_metrics.t.min()]
        assert len(traj_t0_metrics) == 1
        traj_t0_metrics = traj_t0_metrics.iloc[0].to_dict()
        record.update(traj_t0_metrics)

    # Ligand distance
    if row['inference.ligand']:
        for af2, c_alpha in [
            (False, True),
            (False, False),
            # (True, True),
            # (True, True)
        ]:
            dgram = rf_diffusion.dev.analyze.get_dist_to_ligand(row, af2=af2, c_alpha=c_alpha) # [P, L]
            maybe_af2 = 'af2' if af2 else 'des'
            maybe_c_alpha = 'c-alpha' if c_alpha else 'ncac'
            record[f'ligand_dist_{maybe_af2}_{maybe_c_alpha}'] = dgram.min(-1)[0].tolist() # [P]
            record[f'ligand_dist_{maybe_af2}_{maybe_c_alpha}_min'] = dgram.min().item()

    # Secondary structure and radius of gyration
    record.update(calc_mdtraj_metrics(pdb))
    # Broken due to residue indexing.
    # record['rigid_loss'] = rigid_loss(row)
    return record

def calc_mdtraj_metrics(pdb_path):
    try:
        traj = md.load(pdb_path)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == 'C')
        pdb_helix_percent = np.mean(pdb_ss == 'H')
        pdb_strand_percent = np.mean(pdb_ss == 'E')
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
        pdb_rg = md.compute_rg(traj)[0]
    except IndexError as e:
        print('Error in calc_mdtraj_metrics: {}'.format(e))
        pdb_ss_percent = 0.0
        pdb_coil_percent = 0.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
    return {
        'non_coil_percent': pdb_ss_percent,
        'coil_percent': pdb_coil_percent,
        'helix_percent': pdb_helix_percent,
        'strand_percent': pdb_strand_percent,
        'radius_of_gyration': pdb_rg,
    }

def sidechain_symmetry_resolved(pdb):
    o = sidechain(pdb, resolve_symmetry=True)
    return {k if k in ['name', 'mpnn_index'] else f'{k}_sym_resolved': v for k,v in o.items()}

def sidechain(pdb, resolve_symmetry=False):
    out = {}
    row = analyze.make_row_from_traj(pdb[:-4])
    trb = analyze.get_trb(row)
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']
    ref_pdb = analyze.get_input_pdb(row)
    unidealized_pdb = analyze.get_unidealized_pdb(row, return_design_if_backbone_only=True)
    des_pdb = analyze.get_design_pdb(row)
    packed_pdb = analyze.get_mpnn_pdb(row)
    af2_pdb = analyze.get_af2(row)

    indeps = {}
    indeps_a = {}
    atomizers = {}
    point_ids = {}
    ligand = row['inference.ligand']

    name_pdb_pairs = [
        ('ref', ref_pdb),
        ('unideal', unidealized_pdb),
        ('des', des_pdb),
        ('packed', packed_pdb),
    ]

    has_chai = analyze.has_chai1(row)
    n_chai = 5
    chai_names = []
    if has_chai:
        chai_df = analyze.get_chai1_df(row)
        assert chai_df.shape[0] == n_chai
        for _, chai_row in chai_df.iterrows():
            chai_model_index = chai_row['model_idx']
            chai_name = f'chai_{chai_model_index}'
            name_pdb_pairs.append((chai_name, chai_row['pdb_path']))
            chai_names.append(chai_name)

            chai_model_suffix = f'_chaimodel_{chai_model_index}'
            for k, v in chai_row.to_dict().items():

                # Ignore iterable values
                if isinstance(v, list):
                    continue

                out[f'{k}{chai_model_suffix}'] = copy.deepcopy(v)
    
    has_af2 = os.path.exists(af2_pdb)
    if has_af2:
        name_pdb_pairs.append(('af2', af2_pdb))

    for name, pdb in name_pdb_pairs:
        if not os.path.exists(pdb):
            warnings.warn(f'{name} pdb: {pdb} for design {des_pdb} does not exist')
            return {}
        # f['name'] = utils.process_target(pdb, parse_hetatom=True, center=False)
        print(f'parsing {name=} {pdb=}')
        indeps[name] = aa_model.make_indep(pdb, ligand=None if name == 'af2' else ligand)
        if name == 'af2' and resolve_symmetry:
            indep_target = indeps['packed']
            indep_target, _ = aa_model.slice_indep(indep_target, ~indep_target.is_sm)
            indeps['af2'].xyz = resolve_symmetry_indeps(indeps['af2'], indep_target, debug_motif=trb['con_hal_idx0'])
        
        is_atomized = ~indeps[name].is_sm
        atomization_state = aa_model.get_atomization_state(indeps[name])
        atomizers[name] = aa_model.AtomizeResidues(atomization_state, is_atomized)
        indeps_a[name] = atomizers[name].atomize(indeps[name])
        point_ids[name] = aa_model.get_point_ids(indeps_a[name], atomizers[name])
        point_ids[name] = make_ligand_pids_unique(point_ids[name])

            
    
    heavy_motif_atoms = {}
    for ref_idx0, (ref_chain, ref_idx_pdb) in zip(trb['con_ref_idx0'], trb['con_ref_pdb_idx']):
        aa = indeps['ref'].seq[ref_idx0]
        heavy_atom_names = aa_model.get_atom_names(aa)
        heavy_motif_atoms[f'{ref_chain}{ref_idx_pdb}'] = heavy_atom_names

    # Parse contig atoms
    contig_atoms = None
    if 'metrics_meta.contig_atoms' in row:
        contig_atoms = row['metrics_meta.contig_atoms']
    elif 'contigmap.contig_atoms' in row:
        contig_atoms = row['contigmap.contig_atoms']
    if contig_atoms is not None:
        contig_atoms = eval(contig_atoms)
        contig_atoms = {k:v.split(',') for k,v in contig_atoms.items()}
    else:
        contig_atoms = heavy_motif_atoms
    
    def get_pids(name, *getters):
        pids = []
        for g in getters:
            pids.extend(g(point_ids[name]))
        return np.array(pids)
    
    def get_ii(name, *getters):
        pids = get_pids(name, *getters)
        i_by_pid = {pid: i for i, pid in enumerate(point_ids[name])}
        i_by_pid_v = np.vectorize(i_by_pid.__getitem__, otypes=[int])
        ii = i_by_pid_v(pids)
        return ii
    
    def xyz_by_id(name, ii):
        return indeps_a[name].xyz[ii, 1]
    
    def xyz(name, *getters):
        ii = get_ii(name, *getters)
        return xyz_by_id(name, ii)
    
    def zip_safe(*args):
        assert len(set(map(len, args))) == 1
        return zip(*args)

    def get_motif(_, ref: bool, contig_atoms=contig_atoms):
        pids = []
        idx0 = trb[f'con_{"ref" if ref else "hal"}_idx0']
        for (chain, ref_pdb_i), hal_i in zip_safe(
            trb['con_ref_pdb_idx'],
            idx0,
        ):
            atom_names = contig_atoms[f'{chain}{ref_pdb_i}']
            for a in atom_names:
                pids.append(f'A{hal_i}-{a}')
        return pids
    
    heavy_motif_atoms_by_subset = {'all': heavy_motif_atoms}
    empty = {k: [] for k in heavy_motif_atoms.keys()}
    for ref_residue_id, motif_atoms in heavy_motif_atoms.items():
        residue_subset = copy.deepcopy(empty)
        residue_subset[ref_residue_id] = motif_atoms
        heavy_motif_atoms_by_subset[f'residue_{ref_residue_id}'] = residue_subset

    atom_by_residue_by_subset = {}
    atom_by_residue_by_subset['backbone'] = {k: ['N', 'CA', 'C', 'O'] for k,v in heavy_motif_atoms.items()}
    atom_by_residue_by_subset['allatom'] = {k: v for k,v in heavy_motif_atoms.items()}
    atom_by_residue_by_subset['contigatom'] = contig_atoms

    # Dictionary of alignment atoms subset to rmsd atoms subset. e.g.:
    # {
    #     'motif_bb': 'motif_bb',
    #     'motif_bb': 'motif_atoms'
    # }
    rmsd_by_align = (
        ('backbone', 'backbone'),
        ('backbone', 'allatom'),
        ('backbone', 'contigatom'),
        ('allatom', 'allatom'),
        ('allatom', 'contigatom'),
        ('contigatom', 'contigatom'),
    )

    for align_to, rmsd_to in rmsd_by_align:
        ic(align_to, rmsd_to)
        pairs_to_compare = [
                ('des', 'unideal'),
                ('packed', 'unideal'),
            ]
        if has_chai:
            for chai_name in chai_names:
                pairs_to_compare.append((chai_name, 'unideal'))
                pairs_to_compare.append((chai_name, 'packed'))
                pairs_to_compare.append((chai_name, 'ref'))
        if has_af2:
            pairs_to_compare.append(('af2', 'unideal'))
            pairs_to_compare.append(('af2', 'packed'))
            pairs_to_compare.append(('af2', 'ref'))

        for source, target in pairs_to_compare:
            def get_motif_coords(tag):
                is_ref = tag=='ref'
                align_atom_by_residue = atom_by_residue_by_subset[align_to]
                rmsd_atom_by_residue = atom_by_residue_by_subset[rmsd_to]
                get_alignment_atoms = partial(get_motif, ref=is_ref, contig_atoms=align_atom_by_residue)
                get_rmsd_atoms = partial(get_motif, ref=is_ref, contig_atoms=rmsd_atom_by_residue)
                return xyz(tag, get_alignment_atoms), xyz(tag, get_rmsd_atoms)

            motif_source_backbone, motif_source_allatom = get_motif_coords(source)
            motif_target_backbone, motif_target_allatom = get_motif_coords(target)
            T = get_aligner(motif_source_backbone, motif_target_backbone)
            motif_source_allatom_backbone_aligned = T(motif_source_allatom)

            chai_model_suffix = ''
            if 'chai' in source:
                source, chai_model_index = source.split('_')
                chai_model_suffix = f'_chaimodel_{chai_model_index}'

            out[f'{align_to}_aligned_{rmsd_to}_rmsd_{source}_{target}{chai_model_suffix}'] = rmsd(motif_source_allatom_backbone_aligned, motif_target_allatom).item()

    # Attach common names for backwards compatibility
    if has_chai:
        for chai_model_idx in range(n_chai):
            out[f'backbone_aligned_allatom_rmsd_chai_unideal_all_chaimodel_{chai_model_idx}'] = out[f'backbone_aligned_allatom_rmsd_chai_unideal_chaimodel_{chai_model_idx}']
            out[f'constellation_backbone_aligned_allatom_rmsd_chai_unideal_all_chaimodel_{chai_model_idx}'] = out[f'allatom_aligned_allatom_rmsd_chai_unideal_chaimodel_{chai_model_idx}']
    return out


def make_alternate_xyz_indexes():
    n_tokens = len(ChemData().aa2long)
    n_atoms = len(ChemData().aa2long[0])
    out = torch.zeros((n_tokens, n_atoms), dtype=int)
    for i, atom_names in enumerate(ChemData().aa2long[:ChemData().NPROTAAS]):
        for j, atom_name in enumerate(atom_names):
            out[i, j] = ChemData().aa2longalt[i].index(atom_name)
    return out

alternate_xyz_indexes = make_alternate_xyz_indexes()

def get_alternate_xyz(xyz, seq):
    return xyz[torch.arange(xyz.size(0)).unsqueeze(1), alternate_xyz_indexes[seq]]

def get_coords_in_backbone_frame(xyz):
    # A bug in MPNN has resulted in occasional frames that are linear, this correction prevents resulting
    # numerical instabilities.
    xyz = xyz[:]
    xyz[..., 0, 0] += 1e-6

    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], is_na=False)
    xyz = xyz - Ts[...,None,:]
    for i,R in enumerate(Rs):
        try:
            torch.inverse(R)
        except Exception as e:
            ic(
                i,
                R,
                xyz[i, :3]
            )
            raise e
    xyz = torch.einsum('...ij,...kj->...ki', torch.inverse(Rs), xyz)
    return xyz

def resolve_symmetry_indeps(source_indep, target_indep, debug_motif=None):
    atm_mask = ChemData().allatom_mask[target_indep.seq]
    # Heavy atoms only
    atm_mask[:, 14:] = False
    source_indep.xyz[~atm_mask] = float('nan')
    target_indep.xyz[~atm_mask] = float('nan')


    coords_source = source_indep.xyz
    coords_source_alt = get_alternate_xyz(source_indep.xyz, source_indep.seq)
    coords_target = target_indep.xyz

    rel_coords_source = get_coords_in_backbone_frame(coords_source)
    rel_coords_source_alt = get_coords_in_backbone_frame(coords_source_alt)
    rel_coords_target = get_coords_in_backbone_frame(coords_target)

    # same_coords = rel_coords_source == rel_coords_source_alt
    # test_seq_i = 1
    sym_agrees = [[ChemData().aa2long[target_indep.seq[test_seq_i]][i] == ChemData().aa2longalt[target_indep.seq[test_seq_i]][i] for i in
                  range(14)] for test_seq_i in range(target_indep.length())]
    sym_agrees = torch.tensor(sym_agrees)
    has_alt = ~sym_agrees.all(dim=-1)

    distances_true_to_pred = (rel_coords_target - rel_coords_source)**2
    distances_alt_to_pred = (rel_coords_target - rel_coords_source_alt)**2
    distances_true_to_pred[~atm_mask] = 0.
    distances_alt_to_pred[~atm_mask] = 0.

    assert not distances_true_to_pred[0].isnan().any()
    assert not distances_true_to_pred.isnan().any()

    distance_scores_true_to_pred = torch.sum(distances_true_to_pred, dim=(1, 2))
    distance_scores_alt_to_pred = torch.sum(distances_alt_to_pred, dim=(1, 2))

    is_better_alt = distance_scores_alt_to_pred < distance_scores_true_to_pred
    is_better_alt_crds = is_better_alt[:, None, None].repeat(1, ChemData().NTOTAL, 3)

    symmetry_resolved_true_crds = torch.where(is_better_alt_crds, coords_source_alt, coords_source)
    alt_matchs_ref = torch.isclose(coords_source_alt, coords_source, equal_nan=True).all(dim=-1).all(dim=-1)
    assert (has_alt == ~alt_matchs_ref).all()
    return symmetry_resolved_true_crds

def ligand_sasa(pdb):
    out = {}
    row = analyze.make_row_from_traj(pdb[:-4])
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']
    out['ligand_sasa'] = get_ligand_sasa(row)
    return out

def retroaldolase_sasa(pdb):
    out = {}
    row = analyze.make_row_from_traj(pdb[:-4])
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']
    out['ligand_sasa'] = get_ligand_sasa(row)
    out['lysine_sasa'] = get_motif_residue_sasa(row, motif_res=(('A', 1083)))
    return out

def get_ligand_sasa(row):
    mpnn_pdb = analyze.get_mpnn_pdb(row)
    with open(mpnn_pdb) as pdb_file:
        struct = pdb_parser.get_structure('none', pdb_file)
        sr = ShrakeRupley(probe_radius=default_probe_radius)
        sr.compute(struct, level="R")
    resList = PDB.Selection.unfold_entities(struct, target_level='R')
    het_res  = [r for r in PDB.Selection.unfold_entities(resList, target_level='R') if not PDB.is_aa(r)]
    return sum([r.sasa for r in het_res])

def get_motif_residue_sasa(row, motif_res=(('A', 1083))):
    trb = analyze.get_trb(row)

    motif_i = trb['con_ref_pdb_idx'].index(motif_res)
    chain, pdb_i = trb['con_hal_pdb_idx'][motif_i]

    mpnn_pdb = analyze.get_mpnn_pdb(row)
    with open(mpnn_pdb) as pdb_file:
        struct = pdb_parser.get_structure('none', pdb_file)
        sr = ShrakeRupley(probe_radius=default_probe_radius)
        sr.compute(struct, level="R")
    resList = PDB.Selection.unfold_entities(struct, target_level='R')
    res_by_chain_i = {(r.full_id[2], r.full_id[3][1]):r for r in resList}
    res = res_by_chain_i[(chain, pdb_i)]
    return res.sasa


def atoms_within_distance(atom_array, xyz_target, dist_cutoff):
    '''
    Parameters:
        xyz: [N, 3] array of coordinates
        xyz_target: [M, 3] array of coordinates
    '''

    dist = biotite.structure.geometry.distance(
        atom_array.coord[:, None,:],
        xyz_target[None,:,:]
    )
    min_dist = np.min(dist, axis=1)
    return atom_array[min_dist < dist_cutoff]


def get_chai1_metrics(pdb, chai1_pdbs):
    '''
    Given a pdb and the list of pdbs predicted by chai1, return a dictionary of metrics.

    These metrics include:
        Internal ligand RMSD:
        Pocket-aligned ligand RMSD:
        Motif-backbone-aligned sidechain RMSD:
    '''
    pass

def get_mistmatched_annotations(atom_array_a, atom_array_b):
    mismatched = []
    for annotation in atom_array_a.get_annotation_categories():
        if not np.all(atom_array_a.get_annotation(annotation) == atom_array_b.get_annotation(annotation)):
            mismatched.append(annotation)

    return mismatched

def check_atom_names_set_matches(atom_array_a, atom_array_b):
    a_names = set(atom_array_a.atom_name)
    b_names = set(atom_array_b.atom_name)

    a_not_b = a_names.difference(b_names)
    b_not_a = b_names.difference(a_names)

    return a_not_b, b_not_a
    
def print_check_atom_names_set_matches(atom_array_a, atom_array_b):
    a_not_b, b_not_a = check_atom_names_set_matches(atom_array_a, atom_array_b)
    print(f'Atom names in a but not b: {a_not_b}')
    print(f'Atom names in b but not a: {b_not_a}')


def get_ligands(atom_array):
    ligand_by_resname = {}
    ligand_resnames = set(atom_array.res_name[atom_array.hetero])
    for resname in ligand_resnames:
        ligand_by_resname[resname] = atom_array[atom_array.res_name == resname]
    return ligand_by_resname

def atom_array_from_pdb(pdb):
    atom_array = bt.atom_array_from_pdb(pdb)
    atom_array = bt.without_hydrogens(atom_array)
    return atom_array
    
def ligands_match(pdb, chai1_pdbs):
    out = {}

    def atom_array_from_pdb(pdb):
        atom_array = bt.atom_array_from_pdb(pdb)
        atom_array = bt.without_hydrogens(atom_array)
        return atom_array

    atoms_true = atom_array_from_pdb(pdb)
    atoms_pred_stack = [atom_array_from_pdb(p) for p in chai1_pdbs[0:1]]
    # Strip b factors:
    for i, atoms_pred in enumerate(atoms_pred_stack):
        atoms_pred.set_annotation('b_factor', np.ones(len(atoms_pred)))

    atoms_pred_stack = biotite.structure.stack(atoms_pred_stack)
    atoms_pred = atoms_pred_stack[0]

    # Matching sanity checks
    ligands_true = get_ligands(atoms_true)
    atoms_pred.hetero = np.isin(atoms_pred.res_name, list(ligands_true.keys()))
    ligands_pred = get_ligands(atoms_pred)

    out['ligand_match.has_same_ligands'] = set(ligands_true.keys()) == set(ligands_pred.keys())
    ligands_true_lengths = {k: len(v) for k,v in ligands_true.items()}
    ligands_pred_lengths = {k: len(v) for k,v in ligands_pred.items()}

    length_matches = {}
    for k in ligands_true.keys():
        length_matches[k] = ligands_true_lengths[k] == ligands_pred_lengths.get(k, 0)

    order_match = False
    length_match = atoms_pred.hetero.sum() == atoms_true.hetero.sum()
    if length_match:
        order_match = np.all(atoms_pred.res_name[atoms_pred.hetero] == atoms_true.res_name[atoms_true.hetero])
    
    element_match = False
    atoms_pred_hetero = atoms_pred[atoms_pred.hetero]
    atoms_true_hetero = atoms_true[atoms_true.hetero]
    if order_match:
        is_element_match = atoms_pred_hetero.element == atoms_true_hetero.element
        element_match = np.all(is_element_match)
    
    bond_graph_match = False
    if element_match:
        bonds_true = biotite.structure.connect_via_distances(atoms_true_hetero)
        bonds_pred = biotite.structure.connect_via_distances(atoms_pred_hetero)
        bond_graph_match = np.mean(bonds_true.adjacency_matrix() == bonds_pred.adjacency_matrix())
    
    atom_names_match = False
    if length_match:
        is_atom_name_match = atoms_pred_hetero.atom_name == atoms_true_hetero.atom_name
        atom_names_match = np.all(is_atom_name_match)
    
    atom_names_set_match = False
    true_hetero_names = set(atoms_true_hetero.atom_name)
    pred_hetero_names = set(atoms_pred_hetero.atom_name)
    atom_names_set_match = true_hetero_names == pred_hetero_names

    atom_names_set_difference = format_set_difference(true_hetero_names, pred_hetero_names)

    res_name_atom_name_true = set((res_name, atom_name) for res_name, atom_name in zip(atoms_true_hetero.res_name, atoms_true_hetero.atom_name))
    res_name_atom_name_pred = set((res_name, atom_name) for res_name, atom_name in zip(atoms_pred_hetero.res_name, atoms_pred_hetero.atom_name))

    unique_atoms_true = res_name_atom_name_true.difference(res_name_atom_name_pred)
    unique_atoms_pred = res_name_atom_name_pred.difference(res_name_atom_name_true)
    unique_atoms_true_str = ','.join([f'{res_name}_{atom_name}' for res_name, atom_name in unique_atoms_true])
    unique_atoms_pred_str = ','.join([f'{res_name}_{atom_name}' for res_name, atom_name in unique_atoms_pred])


    # Get atoms where atom names match but elements do not
    atom_name_match_element_mismatch_str = ""
    if length_match:
        is_atom_name_match = atoms_pred_hetero.atom_name == atoms_true_hetero.atom_name
        is_element_match = atoms_pred_hetero.element == atoms_true_hetero.element
        is_atom_name_match_element_mismatch = is_atom_name_match & ~is_element_match
        
        matched_names = atoms_true_hetero.atom_name[is_atom_name_match_element_mismatch]
        true_elements = atoms_true_hetero.element[is_atom_name_match_element_mismatch]
        pred_elements = atoms_pred_hetero.element[is_atom_name_match_element_mismatch]

        atom_name_element_mismatch = [f'{name}_{true}_{pred}' for name, true, pred in zip(matched_names, true_elements, pred_elements)]
        n_hetero = len(atoms_true_hetero)
        n_name_match_element_mismatch = is_atom_name_match_element_mismatch.sum()
        atom_name_match_element_mismatch_str = f'{n_name_match_element_mismatch}/{n_hetero}: ' + ','.join(atom_name_element_mismatch)

    
    out['ligand_match.f_match'] = np.array(list(length_matches.values())).mean()
    out['ligand_match.n_match'] = sum(length_matches.values())
    out['ligand_match.n_ligands'] = len(length_matches)
    out['ligand_match.true_lengths'] = str(sorted(ligands_true_lengths.items()))
    out['ligand_match.pred_lengths'] = str(sorted(ligands_pred_lengths.items()))
    out['ligand_match.order_match'] = order_match
    out['ligand_match.element_match'] = element_match
    out['ligand_match.bond_graph_match'] = bond_graph_match
    out['ligand_match.atom_names_match'] = atom_names_match
    out['ligand_match.atom_names_set_match'] = atom_names_set_match
    out['ligand_match.atom_names_set_difference'] = atom_names_set_difference
    out['ligand_match.atom_name_element_mismatch_str'] = atom_name_match_element_mismatch_str
    out['ligand_match.unique_atoms_true_str'] = unique_atoms_true_str
    out['ligand_match.unique_atoms_pred_str'] = unique_atoms_pred_str
    return out

def format_set_difference(set_a, set_b):
    a_not_b = set_a.difference(set_b)
    b_not_a = set_b.difference(set_a)
    n_same = len(set_a.intersection(set_b))
    n_both = len(set_a.union(set_b))
    return f'matching/total: {n_same}/{n_both}, a_not_b: {a_not_b}, b_not_a: {b_not_a}'

def without_sidechains(atom_array):
    is_backbone = biotite.structure.filter_peptide_backbone(atom_array)
    is_returned = is_backbone | atom_array.hetero
    if isinstance(atom_array, biotite.structure.AtomArrayStack):
        return atom_array[..., is_returned]
    return atom_array[is_returned]

def assert_arrays_equal(a, b):
    if (a == b).all():
        return
    i_mismatch = np.where(a != b)[0]
    fraction_matched = np.mean(a == b)
    raise Exception(f'{fraction_matched=}, mismatched (i, got, want): {list(zip(i_mismatch, a[i_mismatch], b[i_mismatch]))}')

def get_atom_id(a):
    return (a.chain_id, a.res_id, a.atom_name)

def set_res_name_occurance_old(atom_array):
    new_id_by_res_name_res_id = {}
    res_name_last_i = defaultdict(int)
    if isinstance(atom_array, biotite.structure.AtomArrayStack):
        aa = atom_array[0]
    else:
        aa = atom_array
    
    for i, atom in enumerate(aa):

        k = (atom.res_name, atom.res_id)
        if k not in new_id_by_res_name_res_id:
            new_id_by_res_name_res_id[k] = f'{atom.res_name}_{res_name_last_i[atom.res_name]}'
            res_name_last_i[atom.res_name] += 1
    atom_array.set_annotation('res_name_occurance', np.array([new_id_by_res_name_res_id[(res_name, res_id)] for res_name, res_id in zip(atom_array.res_name, atom_array.res_id)]))


def set_res_name_occurance(atom_array):
    if isinstance(atom_array, biotite.structure.AtomArrayStack):
        aa = atom_array[0]
    else:
        aa = atom_array
    
    def get_residue_id(atom):
        return (atom.chain_id, atom.res_name, atom.res_id)
    
    def get_res_name(atom):
        return atom.res_name

    subgroup_i_within_group = enumerate_distinct_subgroups(aa, get_res_name, get_residue_id)
    groups = [get_res_name(atom) for atom in aa]
    atom_array.set_annotation('res_name_occurance', np.array([f'{group}_{subgroup_i}' for group, subgroup_i in zip(groups, subgroup_i_within_group)]))

def enumerate_distinct_subgroups(iterable, get_group, get_subgroup):

    new_id_by_res_name_res_id = {}
    res_name_last_i = defaultdict(int)
    
    for element in iterable:
        group = get_group(element)
        subgroup = get_subgroup(element)
        if subgroup not in new_id_by_res_name_res_id:
            new_id_by_res_name_res_id[subgroup] = res_name_last_i[group]
            res_name_last_i[group] += 1
    
    return np.array([new_id_by_res_name_res_id[get_subgroup(element)] for element in iterable])

def find_indices(a, b):
    '''
    Find the index array i such that a[i] == b.
    '''

    if set(a) != set(b):
        raise Exception(format_set_difference(set(a), set(b)))

    # Create a dictionary to map values in a to their indices
    index_map = {value: idx for idx, value in enumerate(a)}
    
    # Use list comprehension to get indices from b using the index map
    indices = np.array([index_map[element] for element in b])

    # assert_arrays_equal
    assert (a[indices] == b).all()
    
    return indices

def assert_unique(a):
    '''
    Parameters:
        a: iterable
    Raises:
        Exception if a contains duplicate elements
    '''
    if len(set(a)) != len(a):
        raise Exception(f'Duplicate elements found in {a}')

def correspond_chai_predictions(atoms_true, atoms_pred_stack):
    '''
    Corresponds the input with the output of chai.

    The output `atoms_true` and `atoms_pred_stack` will have the
    same number of atoms, and atoms_true[i] will correspond to
    atoms_pred_stack[:, i].

    Params:
        atoms_true [AtomArray]: Input to chai
        atoms_pred_stack [AtomArrayStack]: chai predictions
    Returns:
        atoms_true [AtomArray]: Normalized chai input atoms
        atoms_pred_stack [AtomArrayStack]: Normalized chai predictions
        true_extra[AtomArray]: Atoms in the input that were not in the chai prediction
        pred_extra[AtomArray]: Atoms in the prediction that were not in the input
    '''

    # Some initial standardization
    atoms_true = atoms_true.copy()
    atoms_pred_stack = atoms_pred_stack.copy()

    set_res_name_occurance(atoms_true)
    set_res_name_occurance(atoms_pred_stack)

    def get_ids(atom_array):
        return np.array([f'{res_name_occurance}_{atom_name}' for res_name_occurance, atom_name in zip(atom_array.res_name_occurance, atom_array.atom_name)])

    def set_ids(atom_array):
        ids = get_ids(atom_array)
        atom_array.set_annotation('res_name_occurance_atom_name', ids)
        assert_unique(ids)
        return ids
    
    true_ids = set_ids(atoms_true)
    pred_ids = set_ids(atoms_pred_stack)

    true_extra_ids = set(true_ids).difference(set(pred_ids))
    true_extra = atoms_true[np.isin(true_ids, list(true_extra_ids))]

    is_expected_true_extra = [
        lambda atom: (not atom.hetero) and atom.atom_name == 'OXT',
    ]

    def expected_heavy_atom_names(res_name):
        residue = biotite.structure.info.residue(res_name)
        return residue.atom_name[residue.element != 'H']
    
    def true_missing_heavy_atom_names(res_name_occurance):
        residue = atoms_true[res_name_occurance == atoms_true.res_name_occurance]
        true_heavy_atom_names = residue.atom_name
        res_name, occurance = res_name_occurance.split('_')
        return set(expected_heavy_atom_names(res_name)).difference(set(true_heavy_atom_names))

    def corresponding_atom_missing_in_true(atom):
        return (
            (not atom.hetero) and
            atom.atom_name in true_missing_heavy_atom_names(atom.res_name_occurance)
        )
    is_expected_pred_extra = [
        corresponding_atom_missing_in_true
    ]

    unexpected_true_extra = []
    for atom in true_extra:
        for is_expected in is_expected_true_extra:
            for is_expected in is_expected_true_extra:
                if is_expected(atom):
                    break
            else:
                unexpected_true_extra.append(atom)
    
    pred_extra_ids = set(pred_ids).difference(set(true_ids))
    pred_extra = atoms_pred_stack[0, np.isin(pred_ids, list(pred_extra_ids))]
    unexpected_pred_extra = []
    for atom in pred_extra:
        for is_expected in is_expected_pred_extra:
            if is_expected(atom):
                break
        else:
            unexpected_pred_extra.append(atom)
    
    errors = []
    if len(unexpected_true_extra) > 0:
        errors.append(f'Unexpected true extra atoms: {unexpected_true_extra}')
    if len(unexpected_pred_extra) > 0:
        errors.append(f'Unexpected pred extra atoms: {unexpected_pred_extra}')
    
    if len(errors):
        raise Exception('\n'.join(errors))

    shared_ids = set(true_ids).intersection(set(pred_ids))

    atoms_true = atoms_true[np.isin(true_ids, list(shared_ids))]
    atoms_pred_stack = atoms_pred_stack[:, np.isin(pred_ids, list(shared_ids))]
    
    i = find_indices(atoms_pred_stack.res_name_occurance_atom_name, atoms_true.res_name_occurance_atom_name)
    atoms_pred_stack = atoms_pred_stack[:, i]

    assert_arrays_equal(atoms_true.res_name, atoms_pred_stack.res_name)
    assert_arrays_equal(atoms_true.atom_name, atoms_pred_stack.atom_name)

    return atoms_true, atoms_pred_stack, true_extra, pred_extra


def set_is_hetero(atoms_true, atoms_pred_stack):
    '''
    Copy the hetero annotation from atoms_true to atoms_pred_stack by comparing res_name.
    '''
    # Pocket-aligned ligand RMSD
    ligands_true = get_ligands(atoms_true)
    atoms_pred_stack.hetero = np.isin(atoms_pred_stack.res_name, list(ligands_true.keys()))

def get_pocket_aligned_ligand_rmsds(pdb, chai1_pdbs, aligned_pdb_dir=None):

    out = {}

    def atom_array_from_pdb(pdb):
        atom_array = bt.atom_array_from_pdb(pdb)
        atom_array = bt.without_hydrogens(atom_array)
        return atom_array

    atoms_true = atom_array_from_pdb(pdb)
    atoms_pred_stack = [atom_array_from_pdb(p) for p in chai1_pdbs]
    # Strip b factors:
    for i, atoms_pred in enumerate(atoms_pred_stack):
        atoms_pred.set_annotation('b_factor', np.ones(len(atoms_pred)))

    atoms_pred_stack = biotite.structure.stack(atoms_pred_stack)
    set_is_hetero(atoms_true, atoms_pred_stack)

    # Save raw arrays
    atoms_true.set_annotation('original_idx', np.arange(len(atoms_true)))
    atoms_pred_stack.set_annotation('original_idx', np.arange(len(atoms_pred_stack[0])))

    atoms_pred_stack_complete = atoms_pred_stack.copy()

    atoms_true, atoms_pred_stack, _, _ = correspond_chai_predictions(atoms_true, atoms_pred_stack)

    assert_arrays_equal(atoms_true.hetero, atoms_pred_stack.hetero)
    assert_arrays_equal(atoms_true.element, atoms_pred_stack.element)
    assert_arrays_equal(atoms_true.res_name, atoms_pred_stack.res_name)

    # Get the pocket
    atoms_true.set_annotation('corresponded_idx', np.arange(len(atoms_true)))
    atoms_true_het = atoms_true[atoms_true.hetero]

    atoms_true_pocket_candidates = atoms_true[biotite.structure.filter_peptide_backbone(atoms_true) & ~atoms_true.hetero]
    assert atoms_true_pocket_candidates.hetero.sum() == 0
    atoms_true_pocket = atoms_within_distance(atoms_true_pocket_candidates, atoms_true_het.coord, 10)

    pocket_i = atoms_true_pocket.corresponded_idx
    atoms_pred_stack_pocket = atoms_pred_stack[:, pocket_i]
    
    assert (atoms_pred_stack_pocket.res_name == atoms_true_pocket.res_name).all(), f'{list(zip(atoms_pred_stack_pocket.res_name, atoms_true_pocket.res_name))=}'
    fitted, transformations = biotite.structure.superimpose(atoms_true_pocket, atoms_pred_stack_pocket)

    if aligned_pdb_dir is not None:
        atoms_pred_stack_pocket_aligned_complete = transformations.apply(atoms_pred_stack_complete)
        aligned_chai_pdbs = []
        os.makedirs(aligned_pdb_dir, exist_ok=True)
        for i, (chai1_pdb, atoms_pred_aligned) in enumerate(zip(chai1_pdbs, atoms_pred_stack_pocket_aligned_complete)):
            aligned_pdb = os.path.join(aligned_pdb_dir, os.path.basename(chai1_pdb))
            bt.pdb_from_atom_array(atoms_pred_aligned, aligned_pdb)
            aligned_chai_pdbs.append(aligned_pdb)
            out[f'pocket_aligned_chai1_pdb_{i}'] = aligned_chai_pdbs[i]

    atoms_pred_stack_pocket_aligned = transformations.apply(atoms_pred_stack)
    atoms_pred_stack_pocket_aligned_het = atoms_pred_stack_pocket_aligned[:, atoms_pred_stack_pocket_aligned.hetero]

    pocket_rmsd = biotite.structure.rmsd(atoms_true_pocket, atoms_pred_stack_pocket_aligned[:, pocket_i])

    pocket_aligned_ligand_rmsds = biotite.structure.rmsd(atoms_true_het, atoms_pred_stack_pocket_aligned_het)
    for i in range(len(pocket_aligned_ligand_rmsds)):
        out[f'pocket_aligned_model_{i}_total_ligand_rmsd'] = pocket_aligned_ligand_rmsds[i]
        out[f'pocket_aligned_model_{i}_pocket_rmsd'] = pocket_rmsd[i]
    
    # TODO: Handle duplicate ligands
    ligand_resnames = set(atoms_true[atoms_true.hetero].res_name)
    for ligand_resname in ligand_resnames:
        res_ids = set(atoms_true[atoms_true.res_name == ligand_resname].res_id)
        assert len(res_ids) == 1, f'{res_ids=} for {ligand_resname=}.  Currently code assumes single ligand per ligand resname.'
    get_ligand_key = lambda atom: f'{atom.res_name}'

    ligand_rmsd = []
    ligand_keys = np.array([get_ligand_key(atom) for atom in atoms_true])
    ligand_key_set = set(ligand_keys[atoms_true.hetero])
    for ligand_key in ligand_key_set:
        is_ligand = ligand_key == ligand_keys
        atoms_true_ligand = atoms_true[is_ligand]
        atoms_pred_stack_pocket_aligned_ligand = atoms_pred_stack_pocket_aligned[:, is_ligand]

        assert len(set(atoms_true_ligand.res_id)) == 1, f'len({set(atoms_true_ligand.res_id)}) != 1'
        assert len(set(atoms_pred_stack_pocket_aligned_ligand.res_id)) == 1, f'len({set(atoms_pred_stack_pocket_aligned_ligand.res_id)}) != 1'
        ligand_rmsd.append((
                atoms_pred_stack_pocket_aligned_ligand.copy(),
                ligand_key,
                biotite.structure.rmsd(atoms_true_ligand, atoms_pred_stack_pocket_aligned_ligand)
        ))

    ligand_rmsd.sort(key=lambda x: len(x[0][0]), reverse=True)
    for i in range(len(atoms_pred_stack)):
        for ligand_i, (atoms_pred_stack_pocket_aligned_ligand, ligand_key, rmsds) in enumerate(ligand_rmsd):
            # print(f'{ligand_i=} {len(atoms_pred_stack_pocket_aligned_ligand[i])=}')
            out[f'pocket_aligned_model_{i}_ligand_{ligand_i}_rmsd'] = rmsds[i]
            out[f'pocket_aligned_model_{i}_ligand_{ligand_i}_len'] = len(atoms_pred_stack_pocket_aligned_ligand[i])


    i_min = np.argmin(pocket_aligned_ligand_rmsds)
    out['pocket_aligned_total_ligand_rmsd_argmin'] = i_min
    out['pocket_aligned_total_ligand_rmsd_min'] = pocket_aligned_ligand_rmsds[i_min]

    true_pocket_i = atoms_true_pocket.original_idx
    pred_pocket_i = atoms_pred_stack_pocket.original_idx
    out['true_pocket_atom_idx'] = '+'.join(map(str, true_pocket_i))
    out['pred_pocket_atom_idx'] = '+'.join(map(str, pred_pocket_i))

    return out

def has_valid_ccd(row):
    '''
    This stores the labels of M-CSA entries with valid CCD codes (i.e. CCD codes correctly recognized by chai)
    '''
    if 'M0' not in row['name']:
        return True
    good_labels = 'M0040_13pk,M0050_1dbt,M0078_1al6,M0092_1dli,M0093_1dqa,M0096_1chm,M0097_1ctt,M0129_1os7,M0151_1q0n,M0157_1qh5,M0179_1q3s,M0188_1xel,M0315_1ey3,M0349_1e3v,M0365_1pfk,M0375_4ts9,M0500_1e3i,M0552_1fgh,M0555_1f8r,M0584_1ldm,M0636_1uaq,M0663_1rk2,M0664_2dhn,M0710_1ra0,M0711_2esd,M0732_1xs1,M0738_1o98,M0739_1knp,M0904_1qgx,M0907_1rbl'.split(',')
    for label in good_labels:
        if label in row['name']:
            return True
    return False
    

def chai1_pocket_aligned_ligand(pdb):
    row_new = analyze.make_row_from_traj(pdb[:-4], use_trb=False)
    row = row_new

    out = {}
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']
    if not analyze.has_chai1(row):
        return out
    

    valid_ccd = has_valid_ccd(row)
    print(f'{valid_ccd=} {os.path.basename(pdb)=}')
    if not valid_ccd:
        return out
    
    chai_paths_df = analyze.get_chai1_df(row)
    chai1_pdbs = sorted(chai_paths_df['pdb_path'])
    design_pdb = analyze.get_mpnn_pdb(row)
    pocket_metrics = get_pocket_aligned_ligand_rmsds(design_pdb, chai1_pdbs, aligned_pdb_dir=None)
    out.update(pocket_metrics)
    return out

####################
# Geometry Metrics
####################
def geometry(pdb, pdb_idx=None, idx=None, t_step=0):
    """
    Calculates geometrically intuitive metrics for guideposted and non-guideposted designs.
    Can also be used for trajectory files by passing t_step > 0.
    Can be used on native or non-trajectory pdbs by passing the pdb index.
    
    Args:
        pdb (str): path to pdb file (w or w/o trbs)
        pdb_idx (list): pdb indices of the motif residues
        idx (list): list of indices (idx0) of the motif residues
        t_step (int): time step of the trajectory to analyze (default 0), requires inference.write_trajectory=True if > 0

    NB: this function is made more complex because of handling different arguments (trajectory, native / raw pdb or designs w trbs) 
        see geometry_inner for the the computation 
    """

    o = {}

    if pdb_idx is None and idx is None: # Assume pdb is a trajectory pdb

        row = analyze.make_row_from_traj(pdb)
        trb = analyze.get_trb(row)
        config = trb['config']

        o['name'] = row['name']

        motif_idxs = trb['con_hal_pdb_idx']
        parsed = parse_pdb_lines_target(open(pdb, 'r').readlines(), parse_hetatom=True)
        motif_idxs = [parsed['pdb_idx'].index(idx) for idx in motif_idxs]

        # Get Ca-Ca dislocations by using trajectory files if present
        if t_step > 0:
            o, parsed = traj_geometry_precompute(row, trb, config, parsed, motif_idxs, o, t_step)
        elif config['inference'].get('write_trajectory') and t_step <= 0:
            o = o | geometry_metrics_utils.dislocated_ca(pdb)
        else: # otherwise you can't compute them:
            o['gp.ca_dists'] = [0.0] * len(motif_idxs)
    else:
        parsed = parse_pdb_lines_target(open(pdb,'r').readlines(), parse_hetatom=True)
        motif_idxs = [parsed['pdb_idx'].index(idx) for idx in pdb_idx] if idx is None else idx
        o['name'] = Path(pdb).stem
    o = o | geometry_inner(parsed, motif_idxs)
    o = compile_geometry_dict(o)
    return o

def traj_geometry_precompute(row, trb, config, parsed, motif_idxs, o={}, t_step=0):
    """
    Handles geometry pre-calculation for trajectory files
    Loads trajecty and handles guidepost dislocation metric
    
    Args:
        row (dict): design row
        trb (dict): design trb
        config (dict): inference config
        parsed (dict): parsed pdb
        motif_idxs (list): motif indices
        o (dict): output dictionary to append to
        t_step (int): time step of the trajectory to analyze (default 0), requires inference.write_trajectory=True

    Returns:
        o (dict): updated output dictionary
        parsed (dict): updated parsed pdb dictionary

    """

    assert config['inference'].get('write_trajectory'), "write_trajectory must be enabled to compute geometry metrics"
    assert config['inference']['contig_as_guidepost'] 

    ts = trb['t']
    o['t'] = ts[t_step]
    o['1_minus_t'] = 1 - ts[t_step]

    px0_traj_path = analyze.get_traj_path(row, 'X0')
    if not os.path.exists(px0_traj_path): px0_traj_path = analyze.get_traj_path(row, 'x0')
    parsed_t = show_bench.parse_traj(px0_traj_path, t_step=t_step)
    assert not isinstance(parsed_t, list), f"parsed_t is a list: {parsed_t}"
    gp_idxs = np.arange(parsed_t['xyz'].shape[0])[-len(motif_idxs):]

    motif_mask = torch.zeros(parsed_t['xyz'].shape[0]).bool()
    motif_mask[motif_idxs] = True
    gp_mask = torch.zeros(parsed_t['xyz'].shape[0]).bool()
    gp_mask[gp_idxs] = True

    o = o | geometry_metrics_utils.dislocated_ca_inner(parsed_t['xyz'], bb_i=motif_idxs, gp_i=gp_idxs)
    
    # Also overwrite motif coordinates in the parsed structure as the guidepost coordinates 
    # (for metrics: junctions, rotamer probability, angle deviation, tyr contortion etc.)
    xyz = parsed_t['xyz'][~gp_mask]
    xyz[motif_idxs] = parsed_t['xyz'][gp_idxs]
    parsed['xyz'] = xyz  # sequence will be the same (since motif_idxs are the same)
    
    return o, parsed

####################
#
####################


def zip_safe(*args):
    assert len(set(map(len, args))) == 1
    return zip(*args)

def atom_by_id(atoms, get_id=get_atom_id):
    return OrderedDict((get_id(a), a) for a in atoms)

from collections import OrderedDict
def get_atom_by_reference_atom_id(
    contig_atoms, # dict of lists
    con_ref_pdb_idx,
    con_hal_pdb_idx,
    ref = False,
):

    current_by_ref = OrderedDict()

    current_pdb_idx = con_ref_pdb_idx if ref else con_hal_pdb_idx
    for (ref_chain, ref_idx_pdb), (chain, idx_pdb) in zip_safe(con_ref_pdb_idx, current_pdb_idx):
        for atom_name in contig_atoms[f'{ref_chain}{ref_idx_pdb}']:
            current_by_ref[(ref_chain, ref_idx_pdb, atom_name)] = ((chain, idx_pdb, atom_name))

    return current_by_ref

def make_atom_array_safe(atom_list):
    '''
    Constructor for biotite.structure.AtomArray that handles empty lists.
    '''
    if len(atom_list) == 0:
        return biotite.structure.AtomArray(0)
    return biotite.structure.array(atom_list)

def get_motif_atoms(
    atoms,
    contig_atoms, # dict of lists
    con_ref_pdb_idx,
    con_hal_pdb_idx,
    ref = False,
):
    current_by_ref = get_atom_by_reference_atom_id(contig_atoms, con_ref_pdb_idx, con_hal_pdb_idx, ref=ref)
    atoms_by_id = atom_by_id(atoms, get_atom_id)
    atoms = make_atom_array_safe([atoms_by_id[v] for v in current_by_ref.values()])
    return atoms    

def biotite_aligned_rmsd(atoms_true, atoms_pred):
    '''
    Calculate the RMSD between two sets of atoms, using biotite's superimpose function.
    '''
    fitted, transformation = biotite.structure.superimpose(atoms_true, atoms_pred)
    return biotite.structure.rmsd(atoms_true, fitted)


def get_all_heavy_motif_atoms(seq, con_ref_idx0, con_ref_pdb_idx):
    heavy_motif_atoms = {}
    for ref_idx0, (ref_chain, ref_idx_pdb) in zip(con_ref_idx0, con_ref_pdb_idx):
        aa = seq[ref_idx0]
        heavy_atom_names = aa_model.get_atom_names(aa)
        heavy_motif_atoms[f'{ref_chain}{ref_idx_pdb}'] = heavy_atom_names
    return heavy_motif_atoms

def get_implied_contig_backbone_atoms(contig_atoms, con_ref_pdb_idx):
    '''
    Adds N Ca C for con_ref_pdb_idx entries without a corresponding entry in contig_atoms to contig_atoms.
    '''
    implied_contig_atoms = {}
    for (chain, idx_pdb) in con_ref_pdb_idx:
        key = f'{chain}{idx_pdb}'
        if key not in contig_atoms:
            implied_contig_atoms[key] = ['N', 'CA', 'C']
        else:
            implied_contig_atoms[key] = contig_atoms[key]
    return implied_contig_atoms

def get_implied_contig_atoms(contig_atoms, con_ref_pdb_idx, con_ref_idx0, seq):
    '''
    Adds all heavy atoms for con_ref_pdb_idx entries without a corresponding entry in contig_atoms to contig_atoms.
    '''
    implied_contig_atoms = {}
    heavy_motif_atoms = get_all_heavy_motif_atoms(seq, con_ref_idx0, con_ref_pdb_idx)
    for (chain, idx_pdb) in con_ref_pdb_idx:
        key = f'{chain}{idx_pdb}'
        if key not in contig_atoms:
            implied_contig_atoms[key] = heavy_motif_atoms[key]
        else:
            implied_contig_atoms[key] = contig_atoms[key]
    return implied_contig_atoms

def parse_contig_atoms(contig_atoms):
    if contig_atoms is None:
        return {}
    assert isinstance(contig_atoms, str), f'expected string, but {type(contig_atoms)=} != str: {contig_atoms=}'
    contig_atoms = eval(contig_atoms)
    return {k:v.split(',') for k,v in contig_atoms.items()}

def rmsd_to_input(
        pdb,
):
    row = analyze.make_row_from_traj(pdb[:-4])
    out = {'name': row['name']}
    output_pdb = analyze.get_unidealized_pdb(row, return_design_if_backbone_only=True)
    input_pdb = analyze.get_input_pdb(row)

    print(f'{output_pdb=}')

    atoms_input = bt.atom_array_from_pdb(input_pdb)
    atoms_output = bt.atom_array_from_pdb(output_pdb)

    trb = analyze.get_trb(row)
    indep = aa_model.make_indep(input_pdb, ligand=None)

    all_contig_atoms = get_implied_contig_atoms(parse_contig_atoms(row.get('contigmap.contig_atoms')), trb['con_ref_pdb_idx'], trb['con_ref_idx0'], indep.seq)
    all_contig_atoms_minus_backbone_oxygen = {}
    for k,v in all_contig_atoms.items():
        all_contig_atoms_minus_backbone_oxygen[k] = [a for a in v if a != 'O']

    contig_heavy = get_implied_contig_atoms(parse_contig_atoms(None), trb['con_ref_pdb_idx'], trb['con_ref_idx0'], indep.seq)
    contig_heavy_minus_backbone_oxygen = {}
    for k,v in contig_heavy.items():
        contig_heavy_minus_backbone_oxygen[k] = [a for a in v if a != 'O']
    contig_NCaCO = {}
    for k,v in contig_heavy.items():
        contig_NCaCO[k] = ['N', 'CA', 'C', 'O']
    contig_NCaC = {}
    for k,v in contig_heavy.items():
        contig_NCaC[k] = ['N', 'CA', 'C']

    meta_contig_atoms = {}
    if 'metrics_meta.contig_atoms' in row:
        meta_contig_atoms = row['metrics_meta.contig_atoms']
    elif 'contigmap.contig_atoms' in row:
        meta_contig_atoms = row['contigmap.contig_atoms']
    all_contig_atoms_metrics_meta = get_implied_contig_atoms(parse_contig_atoms(meta_contig_atoms), trb['con_ref_pdb_idx'], trb['con_ref_idx0'], indep.seq)
    
    for cohort, contig_atoms_cohort in {
        # 'NCaC': 
        'all': all_contig_atoms,
        'all_minus_backbone_oxygen': all_contig_atoms_minus_backbone_oxygen,
        'heavy': contig_heavy,
        'heavy_minus_backbone_oxygen': contig_heavy_minus_backbone_oxygen,
        'NCaCO': contig_NCaCO,
        'NCaC': contig_NCaC,
        'metrics_meta': all_contig_atoms_metrics_meta,
    }.items():
        atoms_input_cohort = get_motif_atoms(atoms_input, contig_atoms_cohort, trb['con_ref_pdb_idx'], trb['con_hal_pdb_idx'], ref=True)
        atoms_output_cohort = get_motif_atoms(atoms_output, contig_atoms_cohort, trb['con_ref_pdb_idx'], trb['con_hal_pdb_idx'], ref=False)
        assert_arrays_equal(atoms_input_cohort.res_name, atoms_output_cohort.res_name)
        assert_arrays_equal(atoms_input_cohort.atom_name, atoms_output_cohort.atom_name)
        out[f'{cohort}_rmsd_to_input'] = biotite_aligned_rmsd(atoms_input_cohort, atoms_output_cohort)

    out['motif_rmsd_to_input'] = out['all_rmsd_to_input']
    return out

def af2_initial_guess_metrics(pdb):

    row_new = analyze.make_row_from_traj(pdb[:-4], use_trb=False)
    row = row_new

    out = {}
    out['name'] = row['name']
    out['mpnn_index'] = row['mpnn_index']

    tag = os.path.basename(pdb)[:-4]
    af2_dir = os.path.join(os.path.dirname(pdb),"af2_initial_guess/out")

    df = None
    for file in glob.glob(af2_dir + '/*_out.sc'):
        if tag in open(file).read():
            df = pd.read_csv(file, sep='\s+')
            df = df[df['description'].str.contains(tag)]
            df = df[['binder_rmsd', 'interface_rmsd', 'pae_interaction', 'plddt_binder']]
    assert df is not None

    out.update(df.iloc[0].to_dict())

    return out


# For debugging, can be run like:
# exec/rf_diffusion_aa.sif -m fire benchmark/per_sequence_metrics.py single --metric sidechain --pdb=/net/scratch/ahern/se3_diffusion/benchmarks/2024-12-16_08-05-59_enzyme_bench_n41_fixedligand/ligmpnn/packed/run_M0711_2esd_cond32_97-atomized-bb-True_4_1.pdb --log=backbone_aligned_allatom_rmsd_chai_unideal_all_chaimodel_0,constellation_backbone_aligned_allatom_rmsd_chai_unideal_all_chaimodel_0
def single(metric, pdb, log=None, **kwargs):
    metric_f = globals()[metric]
    df = get_metrics([pdb], metric_f)
    ic(df.to_dict())
    if log is not None:
        ic(df[list(log)].to_dict())

def multi(metric, pdbs, **kwargs):
    metric_f = globals()[metric]
    df = get_metrics(pdbs, metric_f)
    return df

if __name__ == '__main__':
    fire.Fire(main)
