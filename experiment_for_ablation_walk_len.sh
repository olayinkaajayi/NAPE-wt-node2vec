#!/bin/bash

home=/dcs/pg20/u2034358/codes/node2vec
code=src/main.py
# file=WikiCSDataset.edgelist
file=ogb-collab.edgelist
# file=WikiCSDataset.json
dimensions=30
no_walks=1
prob_p=0.25
prob_q=2
seed=10
neg_k=20
neg_deg_k=20
lr=0.01
k1=1
k2=1e-4
k3=1
epochs=5
exp_name="walk_len"


function run_experiments {

  walk_len=$1
  model_name="checkpoint_scaledNAPE_COLLAB_l_eq_${walk_len}.pt"

  # Run the experiments
  tmux send-keys -t scaleNAPE "CUDA_VISIBLE_DEVICES=2 python3 $code --filename=$file \
  --walk-length=$walk_len --num-walks=$no_walks --p=$prob_p --q=$prob_q --k=$neg_k --epochs=$epochs \
  --k_deg=$neg_deg_k --checkpoint=$model_name --lr=$lr --k1=$k1 --k2=$k2 --k3=$k3 --seed=$seed --d=$dimensions -i" C-m


  # Compute central Statistics
  code_avg=zz_average_of_mul_simulation.py
  folder_name=$home/zz_ogb_collab_acc_${exp_name}
  train_filename="train_l_eq_${walk_len}_@50Hits.csv"
  test_filename="test_l_eq_${walk_len}_@50Hits.csv"

  tmux send-keys -t benchmark_COLLAB_edge_classification "
  python $code_avg --dataset $dataset --folder $folder_name \
  --best_train_filename $train_filename --best_test_filename $test_filename \
  --exp $exp_name -s
  wait" C-m
}


# Create a new tmux session
tmux new -s scaleNAPE -d
# Send commands to the tmux session
tmux send-keys -t scaleNAPE "cd codes/node2vec" C-m
tmux send-keys -t scaleNAPE "conda activate yinka_env" C-m

walk_lengths=(80 100 120 140 160 180 200)

# Run the experiments for each value of walk length
for walk_len in "${walk_lengths[@]}"; do
  run_experiments $walk_len $exp_name
done

# Get the plots for the ablation
code_avg=zz_average_of_mul_simulation.py
folder_name=$(jq '.save_file.folder' $config_file_mod)

echo "Getting ablation plots!"
tmux send-keys -t benchmark_COLLAB_edge_classification "
python $code_avg --dataset $dataset --folder $folder_name --exp $exp_name --plot
wait" C-m

# remove unused file to avoid redundancy
for eta in "${etas[@]}"; do
  tmux send-keys -t benchmark_COLLAB_edge_classification "rm $home/scripts/COLLAB/modified_config_$eta.json" C-m
done

# Kill the tmux session
# tmux send-keys "tmux kill-session -t benchmark_COLLAB_edge_classification" C-m
