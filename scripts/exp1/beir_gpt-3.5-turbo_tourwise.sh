#!/bin/bash

cd ../..

METHOD_NAME=tourwise
EXPERIMENT_NAME=exp1
MODEL_NAME_OR_PATH=gpt-3.5-turbo

# BEIR_DATASETS=(trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus)
BEIR_DATASET=trec-covid
BASE_SAVE_DIR=outputs
RUN_PATH=run.bm25.${BEIR_DATASET}.txt
N_RUNS=1

### OpenAI API Key
export OPENAI_API_KEY=""

### Tournament Hyperparameters
BATCH_SIZE=10
NUM_TOURNAMENTS=10
TEMPERATURE=0.5

### Construct SAVE_DIR
MODEL_SAVE_NAME=$(echo $MODEL_NAME_OR_PATH | sed 's/\//_/g') # Get the model save name by replace '/' with '_'
SAVE_DIR="${BASE_SAVE_DIR}/${EXPERIMENT_NAME}/${MODEL_SAVE_NAME}/${METHOD_NAME}"

# Add tournament parameters to save dir
SAVE_DIR="${SAVE_DIR}/tournaments-${NUM_TOURNAMENTS}_batchSize-${BATCH_SIZE}_temp-${TEMPERATURE}"

# Append the BEIR_ with specific dataset name
SAVE_DIR="${SAVE_DIR}/BEIR_${BEIR_DATASET}"

# run experiments N_RUNS times
for i in $(seq 1 $N_RUNS)
do
    echo "Running experiment $i"
    SAVE_PATH="${SAVE_DIR}/run-${i}/output.txt"
    echo "SAVE_PATH: $SAVE_PATH"
    
    python run.py \
        run --model_name_or_path $MODEL_NAME_OR_PATH \
            --openai_key $OPENAI_API_KEY \
            --run_path $RUN_PATH \
            --save_path $SAVE_PATH \
            --pyserini_index beir-v1.0.0-${BEIR_DATASET} \
            --hits 100 \
            --query_length 32 \
            --passage_length 128 \
            --scoring generation \
            --shuffle_ranking random \
        tourwise \
            --batch_size $BATCH_SIZE \
            --num_tournaments $NUM_TOURNAMENTS \
            --temperature $TEMPERATURE \
        eval \
            --k_values 3,5,10,25,50,100 \
            --dataset_name $BEIR_DATASET

done
