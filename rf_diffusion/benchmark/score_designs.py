#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

import sys
import os
import glob
import copy
from dataclasses import dataclass
from pathlib import Path
import shlex
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
from rf_diffusion.benchmark.util import slurm_tools


@dataclass(frozen=True)
class Af2Runtime:
    image: Path
    alphafold_dir: Path
    parameters: Path
    databases_dir: Path
    projects_dir: Path


def _first_existing_path(label, candidates, predicate) -> Path:
    # Preserve paths as mounted into the nested container. Resolving a
    # host-side symlink can produce a target outside that container's mounts.
    checked = [Path(path).expanduser().absolute() for path in candidates]
    for path in checked:
        if predicate(path):
            return path
    locations = '\n'.join(f'  - {path}' for path in checked)
    raise FileNotFoundError(f'Unable to locate {label}. Looked in:\n{locations}')


def _resolve_af2_runtime() -> Af2Runtime:
    databases_dir = _first_existing_path(
        'database mount',
        ['/net/databases', '/databases'],
        Path.is_dir,
    )
    return Af2Runtime(
        image=_first_existing_path(
            'mlfold Apptainer image',
            ['/net/software/containers/mlfold.sif', '/software/containers/mlfold.sif'],
            Path.is_file,
        ),
        alphafold_dir=_first_existing_path(
            'AlphaFold installation',
            ['/net/software/mlfold/alphafold', '/software/mlfold/alphafold'],
            Path.is_dir,
        ),
        parameters=_first_existing_path(
            'AlphaFold model parameters',
            [
                databases_dir / 'alphafold' / 'params' / 'params_model_4_ptm.npz',
            ],
            Path.is_file,
        ),
        databases_dir=databases_dir,
        projects_dir=_first_existing_path(
            'project mount',
            ['/projects'],
            Path.is_dir,
        ),
    )


def get_af2_binary(runtime=None) -> str:
    runtime = runtime or _resolve_af2_runtime()
    return shlex.join([
        'apptainer',
        'run',
        '--nv',
        '--bind',
        '/net:/net',
        '--bind',
        f'{runtime.projects_dir}:/projects',
        '--bind',
        f'{runtime.databases_dir}:/databases',
        str(runtime.image),
    ])


@hydra.main(version_base=None, config_path='configs/', config_name='score_designs')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    datadir:        Folder of designs to score.
    trb_dir:        Folder containing .trb files (if not same as datadir).
    filenames:      A path to a list of PDBs to score, rather than scoring everything in datadir.
    chunk:          How many designs to score in each job.
    tmp_pre:        Name prefix of temporary files with lists of designs to score.
    pipeline:       Pipeline mode - submit the next script to slurm with a dependency on jobs from this script. <True, False>
    run:            Comma-separated (no whitespace) list of scoring scripts to run (e.g. "af2,pyrosetta"). <"af2", pyrosetta", "chemnet", "rosettalig">

    slurm:
        J:          Job name
        p:          Partition
        gres:       Gres specification
        submit:     False = Do not submit slurm array job, only generate job list. <True, False>
        in_proc:    Run slurm array job on the current node? <True, False>
        keep_logs:  Keep the slurm logs? <True, False>
    '''
    if conf.filenames:
        filenames = [l.strip() for l in open(conf.filenames).readlines()]
    else:
        filenames = sorted(glob.glob(conf.datadir+'/*.pdb'))
    if len(filenames)==0: sys.exit('No pdbs to score. Exiting.')

    conf.run = conf.run.split(',')

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    job_ids = []

    if 'protein_metrics' in conf.run:
        af2_binary = get_af2_binary()
        # General metrics
        job_fn = conf.datadir + '/jobs.score.protein_metrics.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.protein_metrics.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'{af2_binary} {script_dir}/util/af2_metrics.py --use_ptm '\
                  f'--outcsv {conf.datadir}/protein_metrics.csv.{i} '\
                  f'--trb_dir {conf.trb_dir} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'af2_'+os.path.basename(conf.datadir.strip('/'))
            af2_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if af2_job > 0:
                job_ids.append(af2_job)
            print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to AF2-predict {len(filenames)} designs')


    # AF2 predictions
    if 'af2' in conf.run:
        af2_binary = get_af2_binary()
        job_fn = conf.datadir + '/jobs.score.af2.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            outcsv = f'{conf.datadir}/af2_metrics.csv.{i}'
            job = (f'{af2_binary} {script_dir}/util/af2_metrics.py --use_ptm '\
                  f'--outcsv {outcsv} '\
                  f'--trb_dir {conf.trb_dir} '\
                  f'{tmp_fn}')
            print(job, file=job_list_file)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'af2_'+os.path.basename(conf.datadir.strip('/'))
            # Cached AF2 predictions are validated and reused by af2_metrics.py,
            # but its CSV also contains metrics derived from the current input
            # backbone and TRB/template. Always rerun that inexpensive metric
            # calculation rather than pruning on prediction-cache presence.
            af2_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if af2_job > 0:
                job_ids.append(af2_job)
            print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to AF2-predict {len(filenames)} designs')

    # Rosetta metrics
    if 'pyrosetta' in conf.run:
        # pyrosetta metrics (rog, SS)
        job_fn = conf.datadir + '/jobs.score.pyr.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.pyr.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec /software/containers/pyrosetta.sif python {script_dir}/util/pyrosetta_metrics.py '\
                  f'--outcsv {conf.datadir}/pyrosetta_metrics.csv.{i} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'pyr_'+os.path.basename(conf.datadir.strip('/'))
            pyr_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if pyr_job > 0:
                job_ids.append(pyr_job)
            print(f'Submitted array job {pyr_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to get PyRosetta metrics for {len(filenames)} designs')

    # Ligand metrics (chemnet)
    if 'chemnet' in conf.run:
        job_fn = conf.datadir + '/jobs.score.chemnet.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        chemnet_script = '/net/databases/lab/chemnet/arch.22-10-28/DALigandDock_v03.py'
        for i in range(0, len(filenames), conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.chemnet.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec --nv /software/containers/users/aivan/dlchem.sif python {chemnet_script} '\
                  f'-n 10 --ifile {tmp_fn} '\
                  f'--odir {conf.datadir}/chemnet/ '\
                  f'--ocsv {conf.datadir}/chemnet_scores.csv.{i} ',
                  file=job_list_file)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                pre = 'chemnet_'
                job_name = pre + os.path.basename(conf.datadir.strip('/')) 
            cn_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if cn_job > 0:
                job_ids.append(cn_job)
            print(f'Submitted array job {cn_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to ChemNet-predict {len(filenames)} designs')

    # Ligand metrics (rosetta)
    if False:  #'rosettalig' in conf.run: No current sif file has pyrosetta and pytorch.
        job_fn = conf.datadir + '/jobs.score.rosettalig.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        rosettalig_script = script_dir+'/util/rosetta_ligand_metrics.py'
        for i in range(0, len(filenames), conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.rosettalig.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec /software/containers/mpnn_binder_design.sif python {rosettalig_script} '\
                  f'--list {tmp_fn} '\
                  f'--outdir {conf.datadir}/rosettalig/ '\
                  f'--outcsv {conf.datadir}/rosettalig_scores.csv.{i} ',
                  file=job_list_file)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                pre = 'rosetta_lig_'
                job_name = pre + os.path.basename(conf.datadir.strip('/')) 
            lig_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if lig_job > 0:
                job_ids.append(lig_job)
            print(f'Submitted array job {lig_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute Rosetta ligand metrics on {len(filenames)} designs')

    if 'rosetta_gen_ff' in conf.run:
        job_fn = conf.datadir + '/jobs.score.rosetta_gen_ff.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        rosettalig_script = script_dir+'/util/rosetta_gen_ff.py'
        for i in range(0, len(filenames), conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.rosetta_gen_ff.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'apptainer exec --bind /databases:/databases /software/containers/users/ahern/atom_diff_4.1.sif python -u {rosettalig_script} '\
                  f'--list {tmp_fn} '\
                  f'--trb_dir {conf.trb_dir} '\
                  f'--outdir {conf.datadir}/rosetta_gen_ff/ '\
                  f'--outcsv {conf.datadir}/rosetta_gen_ff.csv.{i} ',
                  file=job_list_file)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                pre = 'rosetta_gen_ff_'
                job_name = pre + os.path.basename(conf.datadir.strip('/')) 
            lig_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc)
            if lig_job > 0:
                job_ids.append(lig_job)
            print(f'Submitted array job {lig_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute Rosetta gen ff metrics on {len(filenames)} designs')

    from collections import defaultdict
    if 'chai1' in conf.run:
        job_fn = conf.datadir + '/jobs.score.chai1.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        already_ran = {}
        filenames_by_i = defaultdict(list)
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.chai1.{i}'
            input_filenames = []
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    input_filenames.append(filenames[j])
                    print(filenames[j], file=outf)
                    filenames_by_i[i].append(filenames[j])
            outdir = f'{conf.datadir}/chai1/out'
            os.makedirs(outdir, exist_ok=True)

            # Resolve chai1 path, e.g. /home/ahern/reclone/rf_diffusion_dev/lib/chai/predict.py
            chai1_script = os.path.join(script_dir, '../../lib/chai/predict.py')
            job = (f'/usr/bin/apptainer run --nv --bind /net/software/lab/chai:/net/software/lab/chai /net/software/lab/chai/chai_apptainer/chai.sif {chai1_script} '\
                  f'--output_dir {outdir} '\
                  f'--pdb_paths_file {tmp_fn}\
                  --allow_ccd_pdb_mismatch')
            print(job, file=job_list_file)
            def outputs_exist(input_filenames=filenames_by_i[i]):
                def expected_outputs(input_pdb):
                    name = os.path.basename(input_pdb).removesuffix('.pdb')
                    return [
                        f'{outdir}/pred.{name}_model_idx_{i}.pdb'
                        for i in range(5)
                    ]
                for input_pdb in input_filenames:
                    for expected in expected_outputs(input_pdb):
                        if not os.path.exists(expected):
                            return False
                return True

            already_ran[job] = copy.deepcopy(outputs_exist)

        # submit job
        if conf.slurm.submit: 
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J 
            else:
                job_name = 'chai1_'+os.path.basename(conf.datadir.strip('/'))
            chai1_job, proc = slurm_tools.array_submit(job_fn, p = conf.chai1.slurm.p, gres=None if conf.chai1.slurm.p=='cpu' else conf.chai1.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc, already_ran=already_ran, mem=conf.chai1.slurm.mem)
            if chai1_job > 0:
                job_ids.append(chai1_job)
            print(f'Submitted array job {chai1_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to chai1-predict {len(filenames)} designs')

    if 'af2_initial_guess' in conf.run:
        job_fn = conf.datadir + '/jobs.score.af2_initial_guess.list'
        job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
        already_ran = {}
        filenames_by_i = defaultdict(list)
        for i in np.arange(0,len(filenames),conf.chunk):
            tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.af2_initial_guess.{i}'
            input_filenames = []
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                    input_filenames.append(filenames[j])
                    print(filenames[j], file=outf)
                    filenames_by_i[i].append(filenames[j])
            outdir = f'{conf.datadir}/af2_initial_guess/out'
            os.makedirs(outdir, exist_ok=True)

            initial_guess_script = '/software/lab/ppi/bcov_scripts/bcov_nate_af2_early_stop/interfaceAF2predict_bcov.py'
            job = (f'/usr/bin/apptainer run --nv -B /software/lab/ppi/bcov_scripts/bcov_nate_af2_early_stop -B /projects/ml/alphafold -B /mnt/net/databases/alphafold /software/containers/users/bcov/bcov_af2.sif {initial_guess_script} '\
                  f'-output_prefix {outdir}/job{i}_ '\
                  f'-pdb_list {tmp_fn}')
            print(job, file=job_list_file)
            def outputs_exist(input_filenames=filenames_by_i[i], output_prefix=f'{outdir}/job{i}_'):
                scorefile = output_prefix + 'out.sc'
                if not os.path.exists(scorefile):
                    return False
                expected_tags = set([os.path.basename(pdb).replace('.pdb', '_af2pred') for pdb in input_filenames])
                found_tags = set([line.split()[-1] for line in open(scorefile)])

                return len(expected_tags & found_tags) == len(expected_tags)

            already_ran[job] = copy.deepcopy(outputs_exist)

        # submit job
        if conf.slurm.submit:
            job_list_file.close()
            if conf.slurm.J is not None:
                job_name = conf.slurm.J
            else:
                job_name = 'af2_initial_guess_'+os.path.basename(conf.datadir.strip('/'))
            af2_initial_guess_job, proc = slurm_tools.array_submit(job_fn, p = conf.af2_initial_guess.slurm.p, gres=None if conf.af2_initial_guess.slurm.p=='cpu' else conf.af2_initial_guess.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc, already_ran=already_ran, mem=conf.af2_initial_guess.slurm.mem)
            if af2_initial_guess_job > 0:
                job_ids.append(af2_initial_guess_job)
            print(f'Submitted array job {af2_initial_guess_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to af2_initial_guess-predict {len(filenames)} designs')


    return job_ids

if __name__ == "__main__":
    main()
