import sys
import json
from LLMRerankSearch import rerank_documents, read_results, write_ranked_results
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)

    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    # Load JSON files
    with open(answers_file, 'r') as f:
        answers = json.load(f)

    with open(topics_1_file, 'r') as f:
        topics_1 = json.load(f)

    with open(topics_2_file, 'r') as f:
        topics_2 = json.load(f)

    topics_1_results = read_results("result_tfidf_1.tsv")
    topics_2_results = read_results("result_tfidf_2.tsv")

    # Rerank Topics 1
    reranked_results_1 = {}
    with tqdm(total=len(topics_1_results), desc="Reranking Topics 1") as pbar:
        for query_id, documents in topics_1_results.items():
            reranked_docs = rerank_documents(topics_1_results, topics_1, answers, model, tokenizer)
            reranked_results_1[query_id] = [(doc_id, score) for doc_id, score in reranked_docs]
            pbar.update(1)

    write_ranked_results(reranked_results_1, "prompt1_1.tsv")

    # Rerank Topics 2
    reranked_results_2 = {}
    with tqdm(total=len(topics_2_results), desc="Reranking Topics 2") as pbar:
        for query_id, documents in topics_2_results.items():
            reranked_docs = rerank_documents(topics_2_results, topics_2, answers, model, tokenizer)
            reranked_results_2[query_id] = [(doc_id, score) for doc_id, score in reranked_docs]
            pbar.update(1)

    write_ranked_results(reranked_results_2, "prompt1_2.tsv")