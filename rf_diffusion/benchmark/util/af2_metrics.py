# 
# Makes AlphaFold2 predictions and computes various RMSDs between AF2 predictions,
# design models, and reference structures.
#
# Usage:
#
#     ./af2_metrics.py FOLDER
#
# This outputs AF2 models to FOLDER/af2/ and metrics to FOLDER/af2_metrics.csv.
# The script automatically uses a template PDB found in the .trb file
# corresponding to each design. If you would like to specify the template, you
# can do:
#
#   ./af2_metrics.py --template TEMPLATE_PDB FOLDER
#
# Updated 2022-8-4

import os
import sys
import argparse
import glob
import time
import re
import numpy as np
import pandas as pd
from collections import OrderedDict

sys.path.insert(0,'/software/mlfold/') # common path on digs
# Shim for deprecated numpy type aliases used by alphafold.
np.int = np.int32
np.object = object
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

from ipd.dev import safe_eval
from alphafold.model.tf import shape_placeholders                                                        
import tensorflow.compat.v1 as tf

class FakeHydra: # spoofing hydra for rf2aa.chemical
    compose=None
    initialize=None
sys.modules["hydra"] = FakeHydra
from rf_diffusion.chemical import ChemicalData as ChemData
from rf_diffusion.benchmark.util.af2_cache import cached_af2_prediction_is_reusable

os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2.0'

def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('input_data', help='Folder of designs, or a text file with a list of paths to designs')
    p.add_argument('-t','--template', help='Template (natural binder) structure (.pdb)')
    p.add_argument('--template_dir', help='Template (natural binder) directory')
    p.add_argument('--subset_res',
        help='Manually specified residue positions to compute rmsd over, e.g. "A163-181,B10"')
    p.add_argument('--outdir', type=str, help='Folder to output predicted structures')
    p.add_argument('--outcsv', type=str, help='Name of output csv file with metrics')
    p.add_argument('--trb_dir', type=str, help='Folder containing .trb files (if not same as pdb folder)')
    p.add_argument('--pdb_suffix', default='', help='PDB files have this suffix relative to trb files')
    p.add_argument('--model_num', default=4, type=int, choices=[1,2,3,4,5], help='AlphaFold model to use')
    p.add_argument('--use_ptm', default=False, action="store_true", help='Use ptm model variant')
    p.add_argument('--num_recycle', default=3, type=int, help='Number of recycles for AlphaFold prediction')
    args = p.parse_args()
    return args


num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    ]

aa2num= {x:i for i,x in enumerate(num2aa)}

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_N = {a:n for n,a in enumerate(alpha_1)}

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
]

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out

def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size,
                   num_res, num_templates=0):
  """Guess at the MSA and sequence dimensions to make fixed size."""
  NUM_RES = shape_placeholders.NUM_RES
  NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ                                                             
  NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
  NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

  pad_size_map = {
      NUM_RES: num_res,
      NUM_MSA_SEQ: msa_cluster_size,
      NUM_EXTRA_SEQ: extra_msa_size,
      NUM_TEMPLATES: num_templates,
  }

  for k, v in protein.items():
    # Don't transfer this to the accelerator.
    if k == 'extra_cluster_assignment':
      continue
    shape = list(v.shape)
    schema = shape_schema[k]
    assert len(shape) == len(schema), (
        f'Rank mismatch between shape and shape schema for {k}: '
        f'{shape} vs {schema}')
    pad_size = [
        pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
    ]
    padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]
    if padding:
      protein[k] = tf.pad(
          v, padding, name=f'pad_to_fixed_{k}')
      protein[k].set_shape(pad_size)
  return {k:np.asarray(v) for k,v in protein.items()}

def idx2contigstr(idx):
    istart = 0
    contigs = []
    for iend in np.where(np.diff(idx)!=1)[0]:
            contigs += [f'{idx[istart]}-{idx[iend]}']
            istart = iend+1
    contigs += [f'{idx[istart]}-{idx[-1]}']
    return contigs

def calc_rmsd(xyz1, xyz2, eps=1e-6):

    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd

def extract_contig_str(trb):
    if trb['settings']['contigs'] is not None:
        return trb['settings']['contigs']
    elif 'mask' in trb['settings']:
        return ','.join([x for x in trb['settings']['mask'].split(',') if x[0].isalpha()])

def contigs2idx(contigs):
    idx = []
    for con in contigs:
        idx.extend(np.arange(con[0],con[1]+1))
    return idx

def parse_range(_range):
    if '-' in _range:
      s, e = _range.split('-')
    else:
      s, e = _range, _range

    return int(s), int(e)

def parse_contig(contig):
    '''
    Return the chain, start and end residue in a contig or gap str.

    Ex:
    'A4-8' --> 'A', 4, 8
    'A5'   --> 'A', 5, 5
    '4-8'  --> None, 4, 8
    'A'    --> 'A', None, None
    '''

    # is contig
    if contig[0].isalpha():
      ch = contig[0]
      if len(contig) > 1:
        s, e = parse_range(contig[1:])
      else:
        s, e = None, None
    # is gap
    else:
      ch = None
      s, e = parse_range(contig)

    return ch, s, e

def expand(mask_str):
    '''
    Ex: '2,A3-5,3' --> [None, None, (A,3), (A,4), (A,5), None, None, None]
    '''
    expanded = []
    for l in mask_str.split(','):
      ch, s, e = parse_contig(l)

      # contig
      if ch:
        expanded += [(ch, res) for res in range(s, e+1)]
      # gap
      else:
        expanded += [None for _ in range(s)]

    return expanded

def lazy_get_model_runner():
    pass


def main():

    args = get_args()

    # Enables lazy-loading of model_runner, if all PDBs already scored.
    model_runner = None
    model2crop_feats = None
    def setup_model():
        # setup AF2 model
        model_name = f'model_{args.model_num}'
        if args.use_ptm:
            model_name += '_ptm'
        print(f'Using {model_name}')
        model_config = config.model_config(model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.model.num_recycle = args.num_recycle
        model_config.data.common.num_recycle = args.num_recycle

        model_config.data.common.max_extra_msa = 1
        model_config.data.eval.max_msa_clusters = 1

        model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/software/mlfold/alphafold-data")
        model_runner = model.RunModel(model_config, model_params)

        eval_cfg = model_config.data.eval
        model2crop_feats = {k:[None]+v for k,v in dict(eval_cfg.feat).items()}

        return model_runner, model2crop_feats

    # reference pdb
    if args.template is not None:
        pdb_ref = parse_pdb(args.template)
        xyz_ref = pdb_ref['xyz'][:,:3]

    # manually specified motif
    if args.subset_res is not None:
        subset_res = expand(args.subset_res)
        L_int = len(subset_res)

    # files to process
    if os.path.isdir(args.input_data):
        filenames = sorted(glob.glob(os.path.join(args.input_data,'*.pdb')))
    else:
        with open(args.input_data) as f:
            filenames = [l.strip() for l in f.readlines()]

    # max length for padding
    seqs = []
    names = []
    for fn in filenames:
        try:
            parsed = parse_pdb(fn)
        except Exception as e:
            print(f'error parsing design {fn}')
            raise e
        seq = ''.join([aa_N_1[a] for a in parsed['seq']])
        seqs.append(seq)
        names.append(os.path.basename(fn).replace('.pdb',''))
    Lmax = max([len(s) for s in seqs])

    # output paths
    if args.outcsv is None:
        args.outcsv = os.path.join(os.path.dirname(filenames[0]),'af2_metrics.csv')
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(filenames[0]),'af2/')
    os.makedirs(args.outdir, exist_ok=True)

    # print table of metrics
    print(f'{"name":>12}{"time (s)":>12}{"af2_plddt":>12}{"rmsd_af2_des":>18}{"contig_rmsd_af2":>18}',end='')
    if args.subset_res is not None:
        print(f'{"subset_rmsd_af2":>20}')
    else:
        print()

    records = []
    for fn, name, seq in zip(filenames, names, seqs):
        t0 = time.time()
        
        row = OrderedDict()
        row['name'] = name
        L = len(seq)

        cached_pdb = os.path.join(args.outdir, name+'.pdb')
        cached_npz = os.path.join(args.outdir, name+'.npz')
        cache_files_exist = os.path.exists(cached_pdb) or os.path.exists(cached_npz)
        reuse_cached_output = cached_af2_prediction_is_reusable(
            fn, cached_pdb, cached_npz
        )
        if cache_files_exist and not reuse_cached_output:
            print(
                f'Cached AF2 artifacts are stale, incomplete, or invalid for {fn}. '
                f'Regenerating {name}.'
            )

        if reuse_cached_output:

            print(f'Output already exists for {name}. Skipping AF2 prediction and calculating '\
                   'RMSD from existing pdb.')
            pdb_af2 = parse_pdb(cached_pdb)
            xyz_pred = pdb_af2['xyz']

            with np.load(cached_npz, allow_pickle=False) as npz:
                row['af2_plddt'] = np.mean(npz['plddt'][:L])
                row['af2_pae_mean'] = np.mean(npz['pae'])
                row['af2_ptm'] = npz['ptm']

        else:
            
            if model_runner is None:
                model_runner, model2crop_feats = setup_model()

            # run AF2
            feature_dict = {
                **pipeline.make_sequence_features(sequence=seq,description="none",num_res=len(seq),idx_res=np.arange(len(seq))),
                **pipeline.make_msa_features(msas=[[seq]],deletion_matrices=[[[0]*len(seq)]]),
            }
            inputs = model_runner.process_features(feature_dict, random_seed=0)
            inputs_padded = make_fixed_size(inputs, model2crop_feats, msa_cluster_size=0, extra_msa_size=0, num_res=Lmax, num_templates=0)

            outputs = model_runner.predict(inputs_padded)
            row['af2_plddt'] = np.mean(outputs['plddt'][:L])
            pae = outputs.get('predicted_aligned_error')
            if pae is not None: pae = pae[:L,:L]
            row['af2_pae_mean'] = np.mean(pae)
            row['af2_ptm'] = outputs.get('ptm')
            
            # unrelaxed_protein = protein.from_prediction(inputs_padded,outputs)
            unrelaxed_protein = protein.Protein(
                aatype=inputs_padded['aatype'][0][:L],
                atom_positions=outputs['structure_module']['final_atom_positions'][:L],
                atom_mask=outputs['structure_module']['final_atom_mask'][:L],
                residue_index=inputs_padded['residue_index'][0][:L]+1,
                b_factors=np.zeros_like(outputs['structure_module']['final_atom_mask'])
            )
            pdb_lines = protein.to_pdb(unrelaxed_protein)

            # save AF2 pdb
            with open(os.path.join(args.outdir,name+'.pdb'), 'w') as f:
                f.write(pdb_lines) 

            # save AF2 residue-wise plddt, pAE, pTM
            np.savez(os.path.join(args.outdir, name+'.npz'),
                 plddt=outputs['plddt'][:L],
                 pae=pae,
                 ptm=outputs.get('ptm')
            )
        pdb_af2 = parse_pdb(os.path.join(args.outdir, name+'.pdb'))
        xyz_pred = pdb_af2['xyz']

        # load designed pdb
        pdb_des = parse_pdb(fn)
        xyz_des = pdb_des['xyz']

        # load trb (has motif residue numbers)
        if args.trb_dir is not None: 
            trb_dir = args.trb_dir
        else:
            trb_dir = os.path.dirname(fn)+'/'
        trbname = os.path.join(trb_dir, name+args.pdb_suffix+'.trb')
        # strip the mpnn suffixes: SEQ_INDEX, PACK_INDEX
        for i in range(2):
            if not os.path.exists(trbname):
                trbname = re.sub('_\d+\.trb$', '.trb', trbname)
        if not os.path.exists(trbname):
            trbname = re.sub('_packed', '', trbname)

        assert os.path.exists(trbname), f'{trbname} does not exist'
        if os.path.exists(trbname): 
            trb = np.load(trbname,allow_pickle=True)

        def get_input_pdb(trb):
            if 'inference.input_pdb' in trb:
                return trb['inference.input_pdb']
            if 'config' in trb:
                return trb['config']['inference']['input_pdb']
            raise Exception('input_pdb not found')
        
        def get_contig_atoms(trb):
            if 'contigmap.contig_atoms' in trb:
                return trb['contigmap.contig_atoms']
            if 'config' in trb:
                return trb['config']['contigmap']['contig_atoms']
            return None

        # load reference structure, if needed
        if args.template is None and args.template_dir is None and os.path.exists(trbname):
            refpdb_fn = get_input_pdb(trb)
            pdb_ref = parse_pdb(refpdb_fn)
            xyz_ref = pdb_ref['xyz']
        if args.template_dir is not None and os.path.exists(trbname):
            pdb_ref = parse_pdb(args.template_dir+trb['settings']['pdb'].split('/')[-1])
            xyz_ref = pdb_ref['xyz']

        # calculate 0-indexed motif residue positions (ignore the ones from the trb)
        if os.path.exists(trbname):
            idxmap = dict(zip(pdb_ref['pdb_idx'],range(len(pdb_ref['pdb_idx']))))
            trb['con_ref_idx0'] = np.array([idxmap[i] for i in trb['con_ref_pdb_idx']])
            idxmap = dict(zip(pdb_des['pdb_idx'],range(len(pdb_des['pdb_idx']))))
            con_hal_pdb_idx = trb['con_hal_pdb_idx']
            def untensor(x):
                if hasattr(x, 'item'):
                    return x.item()
                return x
            con_hal_pdb_idx = [(chain, untensor(idx)) for chain, idx in con_hal_pdb_idx]
            trb['con_hal_idx0'] = np.array([idxmap[i] for i in con_hal_pdb_idx])

        # calculate rmsds
        row['rmsd_af2_des'] = calc_rmsd(xyz_pred[:,:3].reshape(L*3,3), xyz_des[:,:3].reshape(L*3,3))
 
        # load contig position
        if os.path.exists(trbname):
            idx_motif = [i for i,idx in zip(trb['con_hal_idx0'],trb['con_ref_pdb_idx']) 
                         if idx[0]!='R']

            idx_motif_ref = [i for i,idx in zip(trb['con_ref_idx0'],trb['con_ref_pdb_idx']) 
                             if idx[0]!='R']

            atom_exists = pdb_ref['mask'][idx_motif_ref]
            bb_mask = np.zeros_like(atom_exists).astype(bool)
            bb_mask[:,:3] = True
            ca_mask = np.zeros_like(atom_exists).astype(bool)
            ca_mask[:,1] = True

            contig_atoms = get_contig_atoms(trb)
            if contig_atoms is not None:
                contig_atoms = safe_eval(contig_atoms)
                contig_atoms = {k:v.split(',') for k,v in contig_atoms.items()}
                # For debugging: print all contig related keys.
                # for k in trb.keys():
                #     if 'con' in k and k != 'config':
                #         print(f"{k}:{trb[k]}")
                def get_atom_idx(aa, atom_names):
                    i_by_name = {name if name is None else name.strip():i for i, name in enumerate(ChemData().aa2long[aa])}
                    ii = []
                    for a in atom_names:
                        assert a in i_by_name, f'{a=}, {i_by_name=}, {ChemData().num2aa[aa]}'
                        ii.append(i_by_name[a])
                    return ii
                
                motif_atom_idx = []
                for i, (ref_chain, ref_idx_pdb) in zip(trb['con_ref_idx0'], trb['con_ref_pdb_idx']):
                    contig_atoms_key = f'{ref_chain}{ref_idx_pdb}'
                    atom_names = contig_atoms[contig_atoms_key]
                    aa = pdb_ref['seq'][i]
                    motif_atom_idx.append(get_atom_idx(
                        aa, atom_names,
                    ))
                is_motif_atom = np.zeros_like(atom_exists).astype(bool)
                for i, motif_i in enumerate(motif_atom_idx):
                    is_motif_atom[i, motif_i] = True
            else:
                is_motif_atom = atom_exists
            
            for suffix, has_atom in [
                    ('', bb_mask),
                    ('_c_alpha', ca_mask),
                    ('_full_atom', atom_exists),
                    ('_motif_atom', is_motif_atom),
                        ]:
                xyz_ref_motif = xyz_ref[idx_motif_ref][has_atom].reshape(-1,3)
                xyz_pred_motif = xyz_pred[idx_motif][has_atom].reshape(-1,3)
                xyz_des_motif = xyz_des[idx_motif][has_atom].reshape(-1,3)
                row['contig_rmsd_af2_des' + suffix] = calc_rmsd(xyz_pred_motif, xyz_des_motif)
                row['contig_rmsd_af2' + suffix] = calc_rmsd(xyz_pred_motif, xyz_ref_motif)
                row['contig_rmsd' + suffix] = calc_rmsd(xyz_des_motif, xyz_ref_motif)

            xyz_ref = xyz_ref[:,:3]
            xyz_des = xyz_des[:,:3]
            xyz_pred = xyz_pred[:,:3]

            if args.subset_res is not None: 
                idxmap = dict(zip(trb['con_ref_pdb_idx'],trb['con_ref_idx0']))
                idxmap2 = dict(zip(trb['con_ref_pdb_idx'],trb['con_hal_idx0']))
                idx_int_ref = [idxmap[i] for i in subset_res if i in trb['con_ref_pdb_idx']]
                idx_int_hal = [idxmap2[i] for i in subset_res if i in trb['con_ref_pdb_idx']]
                L_int = len(idx_int_ref)

                row['subset_rmsd_af2'] = calc_rmsd(xyz_pred[idx_int_hal].reshape(L_int*3,3), xyz_ref[idx_int_ref].reshape(L_int*3,3))
                row['subset_rmsd_af2_des'] = calc_rmsd(xyz_pred[idx_int_hal].reshape(L_int*3,3), xyz_des[idx_int_hal].reshape(L_int*3,3))

        records.append(row)

        t = time.time() - t0
        print(f'{name:>12}{t:>12.2f}{row["af2_plddt"]:>12.2f}{row["rmsd_af2_des"]:>18.2f}',end='')
        if os.path.exists(trbname):
            print(f'{row["contig_rmsd_af2"]:>18.2f}',end='')
        if args.subset_res is not None:
            print(f'{row["subset_rmsd_af2"]:>20.2f}',end='')
        print()

    df = pd.DataFrame.from_records(records)

    print(f'Outputting computed metrics to {args.outcsv}')
    df.to_csv(args.outcsv)

if __name__ == "__main__":
    main()
