#!/usr/bin/bash  

conda activate ./env_legal2logicicl/

BATCH_SIZE=4
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"  #"Qwen/Qwen3-14B" # meta-llama/Llama-3.1-8B-Instruct "microsoft/phi-4"
ROOT_DIR="./"

for seen_data_r in 0.6  ;
do

    for lambda_sim_div in 0.6 ;
    do
        for seed in  10 11 12 13   ;
        do 
            cd ${ROOT_DIR} && \
            python \
                ${ROOT_DIR}/src/infer_llm_base.py \
                --batch_size $BATCH_SIZE \
                --prompting_data_path ${ROOT_DIR}/data/all_samples_test-c3-t3_lam${lambda_sim_div}_pool-rate-${seen_data_r}_seed${seed}.json \
                --model_name $MODEL_NAME
        done
    done 
done  

cd ${ROOT_DIR} && \
    python \
    ./src/relax_acc_eval.py