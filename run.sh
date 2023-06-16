#!/bin/bash
env_id='Ant-v3'
for seed in `seq 10`
do
    python TrainSAC.py \
    --env_id ${env_id} \
    --reward_form 'implicit' \
    --res_dir "/home/pami/Desktop/Continuous_Algo/results/${env_id}/50cluster_8192buffer_112/$seed" \
    --device 0 \
    --learn_start_steps 50000 \
    --random_steps 50000 \
    --total_env_steps 2500000 \
    --cluster_on_policy_buffer_size 2048 \
    --cluster_off_policy_buffer_size 2048 \
    --cluster_expert_policy_buffer_size 4096 \
    --on_policy_degree 8192 \
    --seed ${seed} &
    echo ${seed}

done
