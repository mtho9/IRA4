import sys
import json
import os
import torch
from LLMRerankSearch import rerank_documents_with_llm, read_results, write_ranked_results
from transformers import pipeline
from tqdm import tqdm

# Set the environment variables for HF_HOME and TRANSFORMERS_CACHE
os.environ['HF_HOME'] = '/mnt/netstore1_home/behrooz.mansouri/HF'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/behrooz.mansouri/HF/cache'

# Set the path to the model directory on netstore
model_path = "/mnt/netstore1_home/behrooz.mansouri/HF/Meta-Llama-3.1-8B-Instruct"

# Check for GPU availability, use CPU if not available
device = 0 if torch.cuda.is_available() else -1

# Initialize the Llama-3.1-8B model using Hugging Face's pipeline, loading from the netstore path
pipeline = pipeline("text-generation", model=model_path,
                    model_kwargs={"torch_dtype": torch.bfloat16}, device=device)

# Set the pad token ID to match the tokenizer's pad token
pipeline.model.generation_config.pad_token_id = pipeline.tokenizer.pad_token_id

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)

    # Read input files
    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    with open(answers_file, 'r') as f:
        answers = json.load(f)
    with open(topics_1_file, 'r') as f:
        topics_1 = json.load(f)
    with open(topics_2_file, 'r') as f:
        topics_2 = json.load(f)

    # Read result files
    topics_1_results = read_results("bm25_1.tsv")
    topics_2_results = read_results("bm25_2.tsv")

    print(f"Topics 1 Results (Total Topics: {len(topics_1_results)}):")
    for topic_id, doc_ids in topics_1_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    print(f"Topics 2 Results (Total Topics: {len(topics_2_results)}):")
    for topic_id, doc_ids in topics_2_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    # Perform reranking for topics_1 and topics_2 using the LLM
    reranked_results_1 = {}
    with tqdm(total=len(topics_1_results), desc="Reranking Topics 1") as pbar:
        reranked_results_1 = rerank_documents_with_llm(topics_1_results, topics_1, answers, pipeline)
        pbar.update(len(topics_1_results))

    # Save the reranked results for topics 1
    write_ranked_results(reranked_results_1, "reranked_results_1.tsv")

    reranked_results_2 = {}
    with tqdm(total=len(topics_2_results), desc="Reranking Topics 2") as pbar:
        reranked_results_2 = rerank_documents_with_llm(topics_2_results, topics_2, answers, pipeline)
        pbar.update(len(topics_2_results))

    # Save the reranked results for topics 2
    write_ranked_results(reranked_results_2, "reranked_results_2.tsv")
