import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import json
import csv
import urllib.request
import zipfile
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker, MarkdownChunker, SemanticChunker
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# 1. Mock Dataset (Corpus, Queries, Qrels)
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. Load Real BEIR Dataset Subset
# ---------------------------------------------------------

NUM_QUERIES = 50
DISTRACTOR_DOCS = 500

dataset_name = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
data_dir = "./datasets"
dataset_path = os.path.join(data_dir, dataset_name)

# Download and extract if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")
    print(f"Downloading {dataset_name} from BEIR...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

qrels = {}
relevant_doc_ids = set()
query_ids = set()

# A. Load Qrels (Test split)
with open(os.path.join(dataset_path, "qrels", "test.tsv"), encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        qid = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score = int(row["score"])

        if score > 0:  # Only keep positive relevance
            if qid not in qrels:
                if len(qrels) >= NUM_QUERIES:
                    break
                qrels[qid] = []
            qrels[qid].append(doc_id)
            relevant_doc_ids.add(doc_id)
            query_ids.add(qid)

# B. Load Queries
queries = {}
with open(os.path.join(dataset_path, "queries.jsonl"), encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        qid = str(data["_id"])
        if qid in query_ids:
            queries[qid] = data["text"]

# C. Load Corpus (Relevant + Distractors)
corpus = {}
distractors_added = 0
with open(os.path.join(dataset_path, "corpus.jsonl"), encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        doc_id = str(data["_id"])
        text = f"{data.get('title', '')}. {data.get('text', '')}"

        if doc_id in relevant_doc_ids:
            corpus[doc_id] = text
        elif distractors_added < DISTRACTOR_DOCS:
            corpus[doc_id] = text
            distractors_added += 1

print(
    f"Dataset ready: {len(queries)} Queries, {len(corpus)} Total Documents (Relevant + Distractors)"
)

model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name
)

print(model_name)
splitters = {
    "Baseline": None,  # No splitting
    "FixedSize": FixedSizeChunker(chunk_size=500, overlap=100),
    "Sentence": SentenceChunker(),
    "Recursive": RecursiveChunker(chunk_size=500),
    "Markdown": MarkdownChunker(chunk_size=500),
    "Semantic": SemanticChunker(embedder_fn=lambda x: emb_fn([x])[0], similarity_threshold=0.5),
}


def calculate_metrics(retrieved_parent_ids, relevant_doc_ids, k=5):
    # 1. Collapse chunks into a ranked list of unique parent documents
    unique_ranked_docs = []
    for doc_id in retrieved_parent_ids:
        if doc_id not in unique_ranked_docs:
            unique_ranked_docs.append(doc_id)

    # 2. Enforce K cutoff on the document level
    top_k_docs = unique_ranked_docs[:k]

    # Calculate Metrics
    acc_1 = 1 if len(top_k_docs) > 0 and top_k_docs[0] in relevant_doc_ids else 0

    relevant_retrieved = [doc for doc in top_k_docs if doc in relevant_doc_ids]
    recall_k = (
        len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0
    )
    precision_k = len(relevant_retrieved) / k if top_k_docs else 0

    f1_k = 0
    if precision_k + recall_k > 0:
        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)

    mrr = 0
    for rank, doc_id in enumerate(top_k_docs):
        if doc_id in relevant_doc_ids:
            mrr = 1.0 / (rank + 1)
            break

    # MAP@K
    ap = 0.0
    hits = 0
    for rank, doc_id in enumerate(top_k_docs):
        if doc_id in relevant_doc_ids:
            hits += 1
            ap += hits / (rank + 1)
    map_k = ap / min(len(relevant_doc_ids), k) if relevant_doc_ids else 0

    # NDCG@K
    dcg = 0.0
    for rank, doc_id in enumerate(top_k_docs):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / np.log2(rank + 2)  # rank is 0-indexed, so we add 2

    idcg = 0.0
    for rank in range(min(len(relevant_doc_ids), k)):
        idcg += 1.0 / np.log2(rank + 2)

    ndcg_k = dcg / idcg if idcg > 0 else 0.0

    return acc_1, recall_k, precision_k, f1_k, mrr, map_k, ndcg_k


# ---------------------------------------------------------
# 4. Main Execution & Export Loop
# ---------------------------------------------------------
chroma_client = chromadb.Client()


K = 5
final_results = []  # To store data for CSV

print(f"\n--- Advanced Retrieval Evaluation (K={K}) ---")

for strategy_name, splitter in splitters.items():
    collection_name = f"eval_{strategy_name.lower()}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name, embedding_function=emb_fn
    )

    # Process and Ingest Corpus
    for doc_id, text in corpus.items():
        if splitter is None:
            chunks = [text]
        else:
            chunks = splitter.chunk(text)

        if not chunks:
            continue

        collection.add(
            documents=chunks,
            metadatas=[{"parent_doc_id": doc_id} for _ in chunks],
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        )

    # Run Queries and Evaluate
    metrics = {
        "acc_1": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "mrr": [],
        "map": [],
        "ndcg": [],
    }

    for q_id, q_text in queries.items():
        relevant_docs = qrels.get(q_id, [])
        if not relevant_docs:
            continue

        # Retrieve many chunks to ensure we can build a list of K unique parent documents
        results = collection.query(
            query_texts=[q_text], n_results=min(100, collection.count())
        )

        retrieved_parent_ids = [
            meta["parent_doc_id"] for meta in results["metadatas"][0]
        ]
        a, r, p, f, m, map_val, n = calculate_metrics(
            retrieved_parent_ids, relevant_docs, k=K
        )

        metrics["acc_1"].append(a)
        metrics["recall"].append(r)
        metrics["precision"].append(p)
        metrics["f1"].append(f)
        metrics["mrr"].append(m)
        metrics["map"].append(map_val)
        metrics["ndcg"].append(n)

    # Aggregate
    agg = {
        "Strategy": strategy_name,
        f"Accuracy@1": round(np.mean(metrics["acc_1"]), 4),
        f"Recall@{K}": round(np.mean(metrics["recall"]), 4),
        f"Precision@{K}": round(np.mean(metrics["precision"]), 4),
        f"F1@{K}": round(np.mean(metrics["f1"]), 4),
        f"MRR@{K}": round(np.mean(metrics["mrr"]), 4),
        f"MAP@{K}": round(np.mean(metrics["map"]), 4),
        f"NDCG@{K}": round(np.mean(metrics["ndcg"]), 4),
    }
    final_results.append(agg)

    # Print
    print(f"\n[ {strategy_name} ]")
    for k_metric, v_metric in agg.items():
        if k_metric != "Strategy":
            print(f"  {k_metric:<12} : {v_metric:.4f}")

# ---------------------------------------------------------
# 5. Export to CSV
# ---------------------------------------------------------
csv_filename = "retrieval_metrics_summary.csv"

with open(csv_filename, mode="w", newline="") as csv_file:
    fieldnames = final_results[0].keys()
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for row in final_results:
        writer.writerow(row)

print(f"\nAll metrics successfully evaluated and saved to '{csv_filename}'.")
