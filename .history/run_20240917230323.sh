python3 run.py \
  run --model_name_or_path gpt-3.5-turbo \
      --openai_key [your key] \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.iswise.generation.openai.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
  listwise --window_size 4 \
           --step_size 2 \
           --num_repeat 5