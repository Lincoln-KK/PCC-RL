#!/bin/bash
project="PCC-NS-rate-vs-actual-full-SB3"
output_file="progress_monitor$project.txt"
objective="Compare the original action formulation to free action range"

#  Number of rounds to test
num_rounds=3
total_timesteps=4000000

echo "Test for $project: $objective. Timesteps = $total_timesteps" | tee -a "$output_file"

for ((i=1; i<=$num_rounds; i++)); do
    echo "Round $i/$num_rounds: $(date)" | tee -a "$output_file"
    echo "Using SB3 PPO on v0 (rate adjustment)" | tee -a "$output_file"
    python sb3_solve.py --total_timesteps $total_timesteps --env_name "PccNs-v0" --wandb_project_name $project
    echo "Using SB3 PPO on v1 (actual value)" | tee -a "$output_file"
    python sb3_solve.py --total_timesteps $total_timesteps --env_name "PccNs-v1" --action_scale 1 --wandb_project_name $project
   
done
