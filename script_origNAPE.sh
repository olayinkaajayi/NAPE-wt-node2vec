#!/bin/bash


############
# Usage: NAPE algorithm for small graphs.
############

# bash codes/node2vec/script_origNAPE.sh


# dimensions=8
# file=ntu_adj_matrix.edgelist
# lr=0.1
# model_name=checkpoint_betaNAPE_ntu.pt
# epochs=400

# dimensions=22
# file=WikiCSDataset.edgelist
# lr=0.01
# model_name=betaNAPE_WikiCS.pt
# epochs=5

# dimensions=30
# file=ogb-collab.edgelist
# lr=0.01
# model_name=betaNAPE_ogb-collab.pt
# epochs=5

dimensions=20
file=Pubmed.edgelist
lr=0.01
model_name=Pubmed_no-scale_NAPE.pt
epochs=5

code=src/main_NAPE.py
seed=10
k1=1
k2=1e-4
k3=1


tmux new -s betaNAPE -d
tmux send-keys -t betaNAPE "cd codes/node2vec" C-m
tmux send-keys -t betaNAPE "conda activate yinka_env" C-m

tmux send-keys -t betaNAPE "CUDA_VISIBLE_DEVICES=2 python3 $code --filename=$file \
--epochs=$epochs --checkpoint=$model_name --lr=$lr --k1=$k1 --k2=$k2 --k3=$k3 \
--seed=$seed --d=$dimensions -i --no-scale " C-m


# tmux send-keys "
# tmux kill-session -t NAPE_PE" C-m
