#!/bin/bash
#
#SBATCH --job-name=scaleNape # Job name for tracking
#SBATCH --partition=sip-turing  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
# 60GB of system RAM will be allocated with 1 GPU specified

#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
module load CUDA/12.2

############
# Usage: scaled version of NAPE algorithm for larger graphs.
############

# bash script_scaleNAPE.sh

code=src/main.py
# file=WikiCSDataset.edgelist
# file=ogb-collab.edgelist
file=Pubmed.edgelist
# file=WikiCSDataset.json
dimensions=24
walk_len=80
no_walks=1
prob_p=0.25
prob_q=2
seed=10
neg_k=15
neg_deg_k=15
lr=0.01
k1=1
k2=1e-3
k3=1
epochs=5
model_name=Pubmed_scaledNAPE_cpu-sbatch.pt

conda activate yinka_env
cd codes/node2vec


python3 $code --filename=$file \
--walk-length=$walk_len --num-walks=$no_walks --p=$prob_p --q=$prob_q --k=$neg_k --epochs=$epochs \
--k_deg=$neg_deg_k --checkpoint=$model_name --lr=$lr --k1=$k1 --k2=$k2 --k3=$k3 --seed=$seed --d=$dimensions -i


