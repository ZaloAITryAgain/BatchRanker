python3 run.py \
  run --model_name_or_path gpt-4o-mini \
      --openai_key $OPENAI_API_KEY \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.batchwise.random.b10.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
  batchwise --num_anchor 4 \
           --batch_size 10 \
           --num_vote 5 \
           --method none \
           --temperature 0.5