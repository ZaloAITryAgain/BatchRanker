DATASET=trec-covid # change to: trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus for other experiments in the paper.

# Get BM25 first stage results
# python -m pyserini.search.lucene \
#   --index beir-v1.0.0-${DATASET}.flat \
#   --topics beir-v1.0.0-${DATASET}-test \
#   --output run.bm25.${DATASET}.txt \
#   --output-format trec \
#   --batch 36 --threads 12 \
#   --hits 1000 --bm25 --remove-query


python3 run.py \
  run --model_name_or_path gpt-4o-mini \
      --openai_key $OPENAI_API_KEY \
      --run_path run.bm25.${DATASET}.txt \
      --save_path run.batchwise.top.b10.noCOT.txt \
      --ir_dataset_name beir-v1.0.0-${DATASET} \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
      --shuffle_ranking random \
  batchwise --num_anchor 4 \
           --batch_size 10 \
           --num_vote 5 \
           --method random \
           --temperature 0.5 \
           --use_COT false