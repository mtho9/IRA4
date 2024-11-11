import sys
import json
from LLMRerankSearch import rerank_documents, read_results, write_ranked_results
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)

    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    with open(answers_file, 'r') as f:
        answers = json.load(f)

    with open(topics_1_file, 'r') as f:
        topics_1 = json.load(f)

    with open(topics_2_file, 'r') as f:
        topics_2 = json.load(f)

    topics_1_results = read_results("result_tfidf_1.tsv")
    topics_2_results = read_results("result_tfidf_2.tsv")

    # rerank for topics_1
    reranked_results_1 = {}
    for query_id, documents in topics_1_results.items():
        reranked_docs = rerank_documents(query_id, documents, model, tokenizer)
        reranked_results_1[query_id] = [(doc_id, score) for doc_id, score in reranked_docs]
    write_ranked_results(reranked_results_1, "reranked_topics_1_results.txt")

    # Rerank for topics_2
    reranked_results_2 = {}
    for query_id, documents in topics_2_results.items():
        reranked_docs = rerank_documents(query_id, documents, model, tokenizer)
        reranked_results_2[query_id] = [(doc_id, score) for doc_id, score in reranked_docs]

    write_ranked_results(reranked_results_2, "reranked_topics_2_results.txt")

    print("Reranking complete. Results written to 'reranked_topics_1_results.txt' and 'reranked_topics_2_results.txt'.")
