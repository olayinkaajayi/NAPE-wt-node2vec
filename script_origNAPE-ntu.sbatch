#!/bin/bash
#
#SBATCH --job-name=scale_origNAPE-ntu # Job name for tracking
#SBATCH --partition=sip-turing  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
# 60GB of system RAM will be allocated with 1 GPU specified

#SBATCH --time=02:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA/12.2

############
# Usage: NAPE algorithm for small graphs.
############

## sbatch codes/node2vec/script_origNAPE-ntu.sbatch


dimensions=8
file=ntu_adj_matrix.edgelist
lr=0.1
model_name=checkpoint_betaNAPE_ntu-sb.pt
epochs=400


code=src/main_NAPE.py
seed=10
k1=1
k2=1
k3=1


cd node2vec
conda activate yinka_env

python3 $code --filename=$file --epochs=$epochs --checkpoint=$model_name --lr=$lr \
--k1=$k1 --k2=$k2 --k3=$k3 --seed=$seed --d=$dimensions -i --no-scale

