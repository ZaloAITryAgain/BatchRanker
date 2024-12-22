cd ../..

METHOD_NAME=batchwise
EXPERIMENT_NAME=exp1
MODEL_NAME_OR_PATH=gpt-3.5-turbo


# BEIR_DATASETS=(trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus)
BEIR_DATASET=trec-covid
BASE_SAVE_DIR=outputs
RUN_PATH=run.bm25.${BEIR_DATASET}.txt
N_RUNS=1


### OpenAI API Key
export OPENAI_API_KEY=""


### Anchor Hyperparameters
USE_COT_ANCHOR=true
METHOD=top
NUM_ANCHOR=4

### LLM Hyperparameters
USE_COT_DOCUMENT=false
NUM_VOTE=5
BATCH_SIZE=10
TEMPERATURE=0.5




### Construct SAVE_DIR
MODEL_SAVE_NAME=$(echo $MODEL_NAME_OR_PATH | sed 's/\//_/g') # Get the model save name by replace '/' with '_'
SAVE_DIR="${BASE_SAVE_DIR}/${EXPERIMENT_NAME}/${MODEL_SAVE_NAME}/${METHOD_NAME}"

# If method is random, append "/random" to the SAVE_DIR, else if num_anchor=0 or method=none or method=no, append "/noAnchor" to the SAVE_DIR, else append "/topAnchor" to the SAVE_DIR
if [ "$METHOD" = "none" ] || [ "$METHOD" = "no" ] || [ "$METHOD" = "noAnchor" ] || [ "$NUM_ANCHOR" = 0 ] ; then
    SAVE_DIR="${SAVE_DIR}/noAnchor"
elif [ "$METHOD" = "random" ] ; then
    SAVE_DIR="${SAVE_DIR}/randomAnchor"
elif [ "$METHOD" = "top" ] ; then
    SAVE_DIR="${SAVE_DIR}/topAnchor"
fi

if [ "$USE_COT_ANCHOR" = true ] ; then
    SAVE_DIR="${SAVE_DIR}_useCOTAnchor"
else
    SAVE_DIR="${SAVE_DIR}_noCOTAnchor"
fi

if [ "$USE_COT_DOCUMENT" = true ] ; then
    SAVE_DIR="${SAVE_DIR}_useCOTDoc"
else
    SAVE_DIR="${SAVE_DIR}_noCOTDoc"
fi


# Append the BEIR_ with specific dataset name
SAVE_DIR="${SAVE_DIR}/BEIR_${BEIR_DATASET}"


# run experiments N_RUNS times
for i in $(seq 1 $N_RUNS)
do
    echo "Running experiment $i"
    SAVE_PATH="${SAVE_DIR}/nAnchor-${NUM_ANCHOR}_nVote-${NUM_VOTE}_batchSize-${BATCH_SIZE}_temperature-${TEMPERATURE}_run-${i}/output.txt"
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
        batchwise \
            --num_anchor $NUM_ANCHOR \
            --batch_size $BATCH_SIZE \
            --num_vote $NUM_VOTE \
            --method $METHOD \
            --temperature $TEMPERATURE \
            --use_COT $USE_COT_DOCUMENT \
            --use_COT_anchor $USE_COT_ANCHOR \
        eval \
            --k_values 3,5,10,25,50,100 \
            --dataset_name $BEIR_DATASET

done