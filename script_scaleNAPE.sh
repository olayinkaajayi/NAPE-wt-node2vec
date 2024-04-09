#!/bin/bash


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
model_name=Pubmed_scaledNAPE_cpu.pt

tmux new -s scaleNAPE -d
tmux send-keys -t scaleNAPE "cd codes/node2vec" C-m
tmux send-keys -t scaleNAPE "conda activate yinka_env" C-m

tmux send-keys -t scaleNAPE "CUDA_VISIBLE_DEVICES=2 python3 $code --filename=$file \
--walk-length=$walk_len --num-walks=$no_walks --p=$prob_p --q=$prob_q --k=$neg_k --epochs=$epochs \
--k_deg=$neg_deg_k --checkpoint=$model_name --lr=$lr --k1=$k1 --k2=$k2 --k3=$k3 --seed=$seed --d=$dimensions -i" C-m


# tmux send-keys "
# tmux kill-session -t NAPE_PE" C-m
