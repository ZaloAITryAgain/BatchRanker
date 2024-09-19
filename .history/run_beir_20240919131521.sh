DATASET=trec-covid # change to: trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus for other experiments in the paper.

# Get BM25 first stage results
python -m pyserini.search.lucene \
  --index beir-v1.0.0-${DATASET}.flat \
  --topics beir-v1.0.0-${DATASET}-test \
  --output run.bm25.${DATASET}.txt \
  --output-format trec \
  --batch 36 --threads 12 \
  --hits 1000 --bm25 --remove-query