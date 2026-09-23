#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

from datetime import datetime, timedelta
import sys
import os
import glob
import numpy as np
from icecream import ic
import hydra
from hydra.core.hydra_config import HydraConfig

script_dir = os.path.dirname(os.path.realpath(__file__))
from rf_diffusion.benchmark.util import slurm_tools

def num_lines(path):
    with open(path, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines

@hydra.main(version_base=None, config_path='configs/', config_name='add_metrics')
def main(conf: HydraConfig) -> list[int]:
    '''
    ### Expected conf keys ###
    datadir:        Folder of designs to score.
    filenames:      A path to a list of PDBs to score, rather than scoring everything in datadir.
    chunk:          How many designs to score in each job.
    tmp_pre:        Name prefix of temporary files with lists of designs to score.
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

    if conf.chunk == -1:
        conf.chunk = len(filenames)

    job_ids = []

    # Filter out those missing TRBs
    filenames = [fn for fn in filenames if os.path.exists(os.path.splitext(fn)[0] + '.trb')]

    backbone_filenames = filenames
    sequence_filenames = []
    for d in conf.mpnn_dirs:
        sequence_filenames.extend(sorted(glob.glob(os.path.join(d, '*.pdb'))))
    # ic(sequence_filenames)
    # raise Exception('stop')

    # General metrics
    for cohort, in_filenames, metrics in [
        ('design', backbone_filenames, conf.design_metrics),
        ('sequence', sequence_filenames, conf.sequence_metrics),
    ]:
        ic(cohort, len(in_filenames), metrics)
        for metric in metrics:
            filenames = in_filenames
            if metric in conf.subset_metrics:
                subset_filenames_path = conf.subset_metrics[metric]
                with open(subset_filenames_path) as f:
                    subset_filenames = set(f.read().splitlines())

                filenames = [f for f in filenames if f in subset_filenames]
                print(f'{len(in_filenames)} files trimmed to {len(filenames)}')

            job_fn = conf.datadir + f'/jobs.metrics_per_{cohort}_{metric}.list'
            job_list_file = open(job_fn, 'w') if conf.slurm.submit else sys.stdout
            for i in np.arange(0,len(filenames),conf.chunk):
                tmp_fn = f'{conf.datadir}/{conf.tmp_pre}.metrics_per_{cohort}_{metric}.{i}'
                n_chunk = 0
                with open(tmp_fn,'w') as outf:
                    for j in np.arange(i,min(i+conf.chunk, len(filenames))):
                        n_chunk += 1
                        print(filenames[j], file=outf)
                out_csv_path = f'{conf.datadir}/metrics/per_{cohort}/{metric}/csv.{i}'

                compute = True
                if os.path.exists(out_csv_path):
                    if num_lines(out_csv_path)-1 == n_chunk:
                        if not conf.invalidate_cache:
                            compute = False
                        if conf.invalidate_cache_older_than:
                            if is_modified_since(out_csv_path, datetime_to_epoch_seconds(conf.invalidate_cache_older_than)):
                                compute = False

                if compute:
                    repo_root = os.path.dirname(script_dir)
                    sif_path = os.path.join(repo_root, 'exec', 'rf_diffusion_aa.sif')
                    if conf.slurm.in_proc:
                        cmd_prefix = 'WANDB_MODE=offline python'
                    else:
                        cmd_prefix = f'apptainer exec --env WANDB_MODE=offline {sif_path} python'
                    print(f'{cmd_prefix} {os.path.join(script_dir, "per_sequence_metrics.py")} '\
                            f'--metric {metric} '\
                            f'--outcsv {conf.datadir}/metrics/per_{cohort}/{metric}/csv.{i} '\
                            f'{tmp_fn}', file=job_list_file)

            # submit job
            if conf.slurm.submit: 
                job_list_file.close()
                if conf.slurm.J is not None:
                    job_name = conf.slurm.J 
                else:
                    job_name = f'{cohort}_metrics_{metric}_'+os.path.basename(conf.datadir.strip('/'))
                af2_job, proc = slurm_tools.array_submit(job_fn, p = conf.slurm.p, gres=None if conf.slurm.p=='cpu' else conf.slurm.gres, log=conf.slurm.keep_logs, J=job_name, in_proc=conf.slurm.in_proc, mem=12)
                if af2_job > 0:
                    job_ids.append(af2_job)
                print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/conf.chunk))} jobs to compute per-{cohort} metrics for {len(filenames)} designs')

    return job_ids


def is_modified_since(path, timestamp):
    """
    Check if a file at the given path was modified since the specified timestamp.

    Args:
        path (str): The path to the file.
        timestamp (float): The timestamp to compare against (in seconds since epoch).

    Returns:
        bool: True if the file was modified since the timestamp, False otherwise.
    """
    try:
        file_mod_time = os.path.getmtime(path)
        return file_mod_time > timestamp
    except FileNotFoundError:
        print(f"File not found: {path}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def datetime_to_epoch_seconds(datetime_str):
    """
    Convert a datetime string in the format 'YYYY-MM-DD_HH:MM' or 'YYYY-MM-DD' to seconds since the epoch in PST (UTC-8).

    Args:
        datetime_str (str): The datetime string to convert.

    Returns:
        float: The number of seconds since the epoch (January 1, 1970) in PST.
    """
    # Define the PST offset (UTC-8)
    pst_offset = timedelta(hours=-8)

    formats = ['%Y-%m-%d_%H:%M', '%Y-%m-%d']  # List of formats to try
    
    for fmt in formats:
        try:
            # Parse the datetime string into a naive datetime object
            naive_dt = datetime.strptime(datetime_str, fmt)
            # Apply the PST offset
            pst_dt = naive_dt + pst_offset
            # Convert the PST datetime object to a timestamp (seconds since epoch)
            epoch_seconds = pst_dt.timestamp()
            return epoch_seconds
        except ValueError:
            continue  # Try the next format
    
    print(f"Error parsing date: {datetime_str}")
    return None
