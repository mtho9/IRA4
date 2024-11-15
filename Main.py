import sys
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLMRerankSearch import read_results, rerank_documents_with_qg, write_ranked_results

os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
os.environ['HF_HOME'] = '/mnt/netstore1_home/mandy.ho/HF'

model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# text generation pipeline with the model and tokenizer
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},  #  mixed precision for efficiency?
)

llm_pipeline.model.generation_config.pad_token_id = tokenizer.pad_token_id
llm_pipeline.model.eval()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)  #

    # Read JSON files containing answers and topics
    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    # Load answers and topics
    with open(answers_file, 'r') as f:
        answers = json.load(f)
    with open(topics_1_file, 'r') as f:
        topics_1 = json.load(f)
    with open(topics_2_file, 'r') as f:
        topics_2 = json.load(f)

    # Read the results of BM25
    topics_1_results = read_results("bm25_1.tsv")
    topics_2_results = read_results("bm25_2.tsv")

    print(torch.cuda.memory_allocated())  # Total memory allocated on the GPU
    print(torch.cuda.memory_reserved())   # Total memory reserved by the GPU

    # Print the number of documents associated with each topic for both result sets
    print(f"Topics 1 Results (Total Topics: {len(topics_1_results)}):")
    for topic_id, doc_ids in topics_1_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    print(f"Topics 2 Results (Total Topics: {len(topics_2_results)}):")
    for topic_id, doc_ids in topics_2_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    # Perform reranking of documents for topics_1 using the pointwise query generation
    reranked_results_1 = rerank_documents_with_qg(topics_1_results, topics_1, answers, llm_pipeline, tokenizer, device)
    # Write the reranked results for topics_1 to a file
    write_ranked_results(reranked_results_1, "reranked_results_1_qg.tsv")

    # Perform reranking of documents for topics_2 using the same method
    reranked_results_2 = rerank_documents_with_qg(topics_2_results, topics_2, answers, llm_pipeline, tokenizer, device)
    # Write the reranked results for topics_2 to a file
    write_ranked_results(reranked_results_2, "reranked_results_2_qg.tsv")
