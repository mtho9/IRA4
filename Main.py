import sys
import json
from LLMRerankSearch import rerank_documents, read_results, write_ranked_results
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

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

    print(f"Topics 1 Results (Total Topics: {len(topics_1_results)}):")
    for topic_id, doc_ids in topics_1_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    print(f"Topics 2 Results (Total Topics: {len(topics_2_results)}):")
    for topic_id, doc_ids in topics_2_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    reranked_results_1 = {}
    with tqdm(total=len(topics_1_results), desc="Reranking Topics 1") as pbar:
        reranked_results_1 = rerank_documents(topics_1_results, topics_1, answers, model, tokenizer)
        pbar.update(len(topics_1_results))

    write_ranked_results(reranked_results_1, "prompt1_1.tsv")

    reranked_results_2 = {}
    with tqdm(total=len(topics_2_results), desc="Reranking Topics 2") as pbar:
        reranked_results_2 = rerank_documents(topics_2_results, topics_2, answers, model, tokenizer)
        pbar.update(len(topics_2_results))

    write_ranked_results(reranked_results_2, "prompt1_2.tsv")