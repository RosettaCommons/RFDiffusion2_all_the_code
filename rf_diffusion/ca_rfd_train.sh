#!/bin/bash 

container="./exec/rf_diffusion_aa.sif"

export MASTER_PORT=12336
### get the first node name as master address - customized for vgg slurm
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

config='RFdiffusion_CA.yaml'
#config='ca_rfd_spoof.yaml'
apptainer exec --nv $container ./train_multi_deep.py --config-name=$config
